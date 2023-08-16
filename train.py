import json
import math
import os
import sys

import torch as th
import torchvision.utils as tvu
from torch.utils.tensorboard import SummaryWriter

import data as mrdata
import nets
import optim
import sampling

exp_root = os.environ['EXPERIMENTS_ROOT']
exp_dir = os.path.join(exp_root, sys.argv[1])
config_path = './config.json'
with open(config_path) as file:
    config = json.load(file)

if os.path.exists(exp_dir):
    pass
else:
    os.makedirs(exp_dir)
for folder in ['checkpoints']:
    if os.path.exists(os.path.join(exp_dir, folder)):
        continue
    os.mkdir(os.path.join(exp_dir, folder))

imsize = config["im_sz"]
R = nets.EnergyNet(
    f_mul=config["f_mul"],
    n_c=config["im_ch"],
    n_f=config["n_f"],
    imsize=imsize,
    n_stages=config["stages"],
    pot='abs'
).cuda()
resume_from = 0

# Load checkpoint if given
if len(sys.argv) > 2:
    R.load_state_dict(state_dict=th.load(sys.argv[2]))
    resume_from = int(sys.argv[3])

writer = SummaryWriter(exp_dir)
K = config["K"]
epsilon = config["epsilon"]
batch_size = config["batch_size"]

data = mrdata.training_data()
print(data.shape)
replay_size = config["replay_size"]

# Replay buffer, initialize with 50% images, 50% uniform noise
replay = th.rand((config["replay_size"], config["im_ch"], imsize, imsize),
                 device='cuda')
ims = th.randperm(replay_size // 2)
replay[:replay_size // 2] = data[ims % data.shape[0]]
y = data[th.randperm(data.shape[0])[:replay_size // 2]]
# To keep average value in images, scramble along last two dims
rand = y.view(*y.shape[:2],
              y.shape[2] * y.shape[3])[...,
                                       th.randperm(y.shape[2] *
                                                   y.shape[3])].view(y.shape)
replay[replay_size // 2:] = rand.clone()
# Saving some memory
del rand
del y

lr = config["lr"]
optimizer = optim.AdaBelief(R.parameters(), lr=lr)
scheduler = th.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[500, 2000, 3000, 5000, 7000], gamma=0.5
)
log_freq = 20
save_freq = 50


def update_replay(indices, samples):
    # 60% chance of surviving 50 epochs
    where = th.rand((batch_size, 1, 1, 1),
                    device=data.device).repeat((1, *data.shape[1:])) < 0.99
    # 50% chance that we refill with uniform noise or sample from dataset
    where_new = th.rand((batch_size, 1, 1, 1),
                        device=data.device).repeat((1, *data.shape[1:])) < 0.5
    y = data[th.randperm(data.shape[0])[:batch_size]]
    # To keep average value in images, scramble along last two dims
    rand = y.view(*y.shape[:2], y.shape[2] *
                  y.shape[3])[..., th.randperm(y.shape[2] *
                                               y.shape[3])].view(y.shape)
    refill = th.where(where_new, rand, y)
    # update persistent image bank
    replay[indices] = th.where(where, samples, refill)


def augment(x: th.Tensor) -> th.Tensor:
    return x + th.randn_like(x) * 1.5e-2


for i in range(100_000):
    print(i + resume_from)
    with th.no_grad():
        x = augment(data[th.randperm(data.shape[0])[:batch_size]])

        indices = th.randperm(replay_size)[:batch_size]
        samples = replay[indices].clone()
        y = sampling.ula(
            samples,
            lambda x: R.grad(x)[1],
            n=math.ceil(K * (1 - math.exp(-(i + resume_from) / 1000) + 0.01)),
            epsilon=config["epsilon"],
        )
        update_replay(indices, y.clone())
    loss = R(x).mean() - R(y).mean()
    if (i + resume_from) % log_freq == 0:
        writer.add_scalar('loss', loss.item(), global_step=(i + resume_from))
        writer.add_image(
            'y',
            tvu.make_grid(y[:8]).squeeze().clip_(0, 1),
            global_step=(i + resume_from),
        )
    if (i + resume_from) % save_freq == 0:
        th.save(
            R.state_dict(),
            os.path.join(
                exp_dir, 'checkpoints', f'{i + resume_from:06d}.ckpt'
            )
        )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
