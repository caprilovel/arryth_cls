from torch.utils.tensorboard import SummaryWriter
import numpy as np


writer = SummaryWriter(log_dir = 'logs')
for x in range(1, 101):
    writer.add_scalar('y=2x', x, 2*x)
writer.close()