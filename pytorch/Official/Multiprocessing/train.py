import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

def train(rank, args, model, device, dataloader_kwargs):
    torch.manual_seed(args.seed + rank)

    transformer=transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,))])

    dataset_mnist=datasets.MNIST('../data',train=True,download=True,transform=transformer)
    
    train_loader = torch.utils.data.DataLoader(
      dataset_mnist,batch_size=args.batch_size, shuffle=True, num_workers=1,**dataloader_kwargs)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # @ Iterate epoch
    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch, args, model, device, train_loader, optimizer)


def test(args, model, device, dataloader_kwargs):
    torch.manual_seed(args.seed)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.batch_size, shuffle=True, num_workers=1,
        **dataloader_kwargs)

    test_epoch(model, device, test_loader)

def train_epoch(epoch, args, model, device, data_loader, optimizer):
    # ================================================================================
    # @ Train mode
    model.train()

    # ================================================================================
    # @ Process ID
    pid = os.getpid()
    print("epoch",epoch)
    print("traing start on pid",pid)

    # ================================================================================
    # @ Iterate all batches
    for batch_idx, (data, target) in enumerate(data_loader):

        # @ Remove existing gradients
        optimizer.zero_grad()

        # @ c output: make prediction
        output = model(data.to(device))

        # @ c loss: get loss value
        loss = F.nll_loss(output, target.to(device))

        # @ Get graident
        loss.backward()

        # @ Update network
        optimizer.step()

        # ================================================================================
        # if batch_idx % args.log_interval == 0:
        if batch_idx % args.log_interval == 100:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                pid, epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))

    print("epoch",epoch)
    print("train end on pid",pid)

    # processes [<Process(Process-1, started)>, <Process(Process-2, started)>]
    # epoch 1
    # traing start on pid 19555
    # epoch 1
    # traing start on pid 19556
    # epoch 1
    # train end on pid 19555
    # epoch 2
    # traing start on pid 19555
    # epoch 1
    # train end on pid 19556
    # epoch 2
    # traing start on pid 19556
    # epoch 2
    # train end on pid 19555
    # epoch 3
    # traing start on pid 19555
    # epoch 2
    # train end on pid 19556
    # epoch 3
    # traing start on pid 19556
    # epoch 3
    # train end on pid 19555
    # epoch 3
    # train end on pid 19556

def test_epoch(model, device, data_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item() # sum up batch loss
            pred = output.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.to(device)).sum().item()

    test_loss /= len(data_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(data_loader.dataset),
    #     100. * correct / len(data_loader.dataset)))
