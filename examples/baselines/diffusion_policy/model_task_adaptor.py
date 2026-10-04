import torch
from torch import nn
from torch.nn import functional as F

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.conditional_unet1d import ConditionalUnet1D

class ResidualAdapter(nn.Module):
    def __init__(self, channels, rank):
        super().__init__()

        self.down = nn.Conv1d(channels, rank, kernel_size=1, bias=False)
        self.up = nn.Conv1d(rank, channels, kernel_size=1, bias=False)

        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        residual = self.up(F.mish(self.down(x)))
        return x + residual


class TaskAdapterMidBlock(nn.Module):
    def __init__(self, block, channels, rank, num_tasks):
        super().__init__()

        self.block = block

        self.adapters = nn.ModuleList([
            ResidualAdapter(channels, rank)
            for _ in range(num_tasks)
        ])

        self.task_id = None

    def forward(self, x, cond):
        if self.task_id is None:
            raise RuntimeError(
                "Select a task through Agent.condition() before using the UNet"
            )

        x = self.block(x, cond)
        x = self.adapters[self.task_id](x)
        return x

class Agent(nn.Module):
    def __init__(self, entries, args):
        super().__init__()

        # need 1-6 tasks
        if not 1 <= len(entries) <= 6:
            raise ValueError(f'Expected 1-6 entries, got {len(entries)}')

        self.entries = list(entries)
        self.num_tasks = len(entries)
        
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon

        self.act_dim = entries[0]["action_dim"]

        self.encoder_dim = args.encoder_dim
        self.task_embedding_dim = args.task_embedding_dim

        # unique encoder for each task to accept different observation dimensions
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(entry["obs_dim"], self.encoder_dim),
                nn.Mish(),
                nn.Linear(self.encoder_dim, self.encoder_dim),
            )
            for entry in entries
        ])

        # embedding table for task IDs
        self.task_embedding = nn.Embedding(
            num_embeddings=self.num_tasks,
            embedding_dim=self.task_embedding_dim,
        )

        # concatenation of encoded observations and task embedding
        self.condition_dim = (
            self.obs_horizon * self.encoder_dim
            + self.task_embedding_dim
        )

        # tasks share the same baseline diffusion model
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim, # act_horizon is not used (U-Net doesn't care)
            # global_cond_dim=np.prod(env.single_observation_space.shape), # obs_horizon * obs_dim
            global_cond_dim=self.condition_dim,
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )

        # adaptor
        channels = args.unet_dims[-1]

        rank = args.adapter_rank

        for index, block in enumerate(self.noise_pred_net.mid_modules):
            self.noise_pred_net.mid_modules[index] = TaskAdapterMidBlock(
                block=block,
                channels=channels,
                rank=rank,
                num_tasks=self.num_tasks,
            )

        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2', # has big impact on performance, try not to change
            clip_sample=True, # clip output to [-1,1] to improve stability
            prediction_type='epsilon' # predict noise (instead of denoised action)
        )
    
    def condition(self, obs_seq, task_id=None):
        # check task_id
        if not 0 <= task_id < self.num_tasks:
            raise ValueError(f'Invalid task_id {task_id}, expected 0-{self.num_tasks-1}')

        obs_dim = self.entries[task_id]["obs_dim"]

        # obs_seq: (B, obs_horizon, obs_dim)
        assert obs_seq.ndim == 3
        assert obs_seq.shape[1] == self.obs_horizon
        assert obs_seq.shape[2] == obs_dim

        for block in self.noise_pred_net.mid_modules:
            block.task_id = task_id

        # tasks use different encoders to accept different observation dimensions
        encoded = self.encoders[task_id](obs_seq) # (B, obs_horizon, encoder_dim)
        encoded = encoded.flatten(start_dim=1) # (B, obs_horizon * encoder_dim)

        # (B,)
        task_ids = torch.full(
            (obs_seq.shape[0],),
            task_id,
            dtype=torch.long,
            device=obs_seq.device,
        )

        task_features = self.task_embedding(task_ids) # (B, task_embedding_dim)

        # (B, condition_dim)
        return torch.cat(
            [encoded, task_features],
            dim=-1,
        )

    def compute_loss(self, obs_seq, action_seq, task_id=None):
        B = obs_seq.shape[0]

        # observation as FiLM conditioning
        # obs_cond = obs_seq.flatten(start_dim=1) # (B, obs_horizon * obs_dim)
        obs_cond = self.condition(obs_seq, task_id=task_id)

        # sample noise to add to actions
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=obs_seq.device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=obs_seq.device
        ).long()

        # add noise to the clean images(actions) according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action_seq = self.noise_scheduler.add_noise(
            action_seq, noise, timesteps)

        # predict the noise residual
        noise_pred = self.noise_pred_net(
            noisy_action_seq, timesteps, global_cond=obs_cond)

        return F.mse_loss(noise_pred, noise)

    def get_action(self, obs_seq, task_id=None):
        # init scheduler
        # self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        # set_timesteps will change noise_scheduler.timesteps is only used in noise_scheduler.step()
        # noise_scheduler.step() is only called during inference
        # if we use DDPM, and inference_diffusion_steps == train_diffusion_steps, then we can skip this

        # obs_seq: (B, obs_horizon, obs_dim)
        B = obs_seq.shape[0]
        with torch.no_grad():
            # obs_cond = obs_seq.flatten(start_dim=1) # (B, obs_horizon * obs_dim)
            obs_cond = self.condition(obs_seq, task_id=task_id)

            # initialize action from Guassian noise
            noisy_action_seq = torch.randn((B, self.pred_horizon, self.act_dim), device=obs_seq.device)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )

                # inverse diffusion step (remove noise)
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                ).prev_sample

        # only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end] # (B, act_horizon, act_dim)