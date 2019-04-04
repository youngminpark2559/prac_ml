# conda activate py36gputorch100 && \
# cd /mnt/1T-5e7/mycodehtml/prac_deep_l/pytorch/Multiprocessing && \
# rm e.l && python main.py \
# 2>&1 | tee -a e.l && code e.l

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from train import train, test

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=3, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)

if __name__ == '__main__':
  args = parser.parse_args()
  # print("args",args)
  # Namespace(
  #   batch_size=64,
  #   cuda=True,
  #   epochs=10,
  #   log_interval=10,
  #   lr=0.01,
  #   momentum=0.5,
  #   num_processes=2,
  #   seed=1,
  #   test_batch_size=1000)

  # ================================================================================
  use_cuda = args.cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  
  # ================================================================================
  # print("use_cuda",use_cuda)
  # True

  if use_cuda:
    dataloader_kwargs={'pin_memory': True}
  else:
    dataloader_kwargs={}
  # print("dataloader_kwargs",dataloader_kwargs)
  # {'pin_memory': True}

  # ================================================================================
  torch.manual_seed(args.seed)

  # mp.set_start_method('spawn')
  mp.set_start_method('spawn',force=True)

  # ================================================================================
  model = Net().to(device)
  model.share_memory() # gradients are allocated lazily, so they are not shared here

  # ================================================================================
  processes = []

  # Following loop creates 2 processes
  # and they process training in parallel lazily (?)
  for rank in range(args.num_processes):
    p = mp.Process(target=train, args=(rank, args, model, device, dataloader_kwargs))
    # print("p",p)
    # <Process(Process-1, initial)>

    # We first train the model across `num_processes` processes
    p.start()

    # print("p",p)
    # <Process(Process-1, started)>

    processes.append(p)

  # ================================================================================
  # print("processes",processes)
  # [<Process(Process-1, started)>, <Process(Process-2, started)>]

  for p in processes:
    p.join()

  # Once training is complete, we can test the model
  test(args, model, device, dataloader_kwargs)
