# https://github.com/pytorch/examples/blob/main/mnist/main.py

import argparse
import os

import torch
from torchvision import datasets, transforms

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")


def warp_main(cfg):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_accel = torch.cuda.is_available()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_kwargs = {"batch_size": args.batch_size, "shuffle": True}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_accel:
        accel_kwargs = {"num_workers": 0, "pin_memory": True}
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform),
        **train_kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_DIR, train=False, transform=transform),
        **test_kwargs,
    )

    # TODO: replace torch code with numpy

    import requilt.nn as nn
    import requilt.optim as optim
    import warp as wp

    in_dim = 28 * 28
    out_dim = 10

    # NOTE: train and test batch size must be the same

    # TILE_B = 1
    TILE_B = 2
    layers = [
        nn.Linear(in_dim, 256, tiles=(TILE_B, 16, 16)),
        nn.Activation("relu"),
        nn.Linear(256, 64, tiles=(TILE_B, 16, 16)),
        nn.Activation("relu"),
        nn.Linear(64, out_dim, tiles=(TILE_B, 16, 5)),
        nn.LogSoftmax(axis=1),
    ]
    model = nn.Sequential(layers)
    criterion = nn.NLLLoss()

    params = model.parameters()
    for i, p in enumerate(params):
        print(i, p.shape, p.dtype, p.requires_grad)

    optimizer_grads = [p.grad.flatten() for p in params]
    optimizer_inputs = [p.flatten() for p in params]
    optimizer = optim.Adam(optimizer_inputs, lr=cfg.lr)

    # TODO: graph capture
    # x = wp.zeros((cfg.batch_size, in_dim), dtype=float, device=device, requires_grad=True)
    # y = wp.zeros((cfg.batch_size, out_dim), dtype=float, device=device, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    def train(cfg, model, device, train_loader, optimizer, epoch):
        # model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.reshape(-1, 28 * 28)
            data, target = wp.from_torch(data), wp.from_torch(target)
            data.requires_grad = True

            loss.zero_()
            with wp.Tape() as tape:
                output = model(data)
                criterion(output, target, y=loss)
                tape.backward(loss)
                optimizer.step(optimizer_grads)
                tape.zero()

            if batch_idx % cfg.log_interval == 0:
                _loss = loss.numpy().item() / float(cfg.batch_size)  # NOTE: take mean since loss uses reduction="sum" currently
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {_loss:.6f}"
                )

    def test(model, device, test_loader):
        # model.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = data.reshape(-1, 28 * 28)
                data, target = wp.from_torch(data), wp.from_torch(target)

                # TODO: avoid numpy conversion
                output = model(data)
                test_loss += criterion(output, target).numpy()  # sum up batch loss
                pred = output.numpy().argmax(axis=1, keepdims=True)  # get the index of the max log-probability
                correct += (pred == target.numpy().reshape(pred.shape)).sum()

        test_loss /= float(len(test_loader.dataset))
        test_loss = test_loss.item()

        print(
            f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100.0 * correct / len(test_loader.dataset):.2f}%)\n"
        )

    for epoch in range(1, cfg.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)


def torch_main(cfg):
    torch.manual_seed(args.seed)
    device = torch.accelerator.current_accelerator()
    use_accel = torch.accelerator.is_available()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_kwargs = {"batch_size": args.batch_size, "shuffle": True}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_accel:
        accel_kwargs = {"num_workers": 0, "pin_memory": True}
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform),
        **train_kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_DIR, train=False, transform=transform),
        **test_kwargs,
    )

    import torch.nn as nn
    import torch.nn.functional as F

    class ConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            # self.dropout1 = nn.Dropout(0.25)
            # self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            # x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            # x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
        nn.LogSoftmax(dim=1),
    ).to(device)
    # model = ConvNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    print(model)

    def train(cfg, model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % cfg.log_interval == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )

    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print(
            f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100.0 * correct / len(test_loader.dataset):.2f}%)\n"
        )

    for epoch in range(1, cfg.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST Example")
    parser.add_argument(
        "--batch-size", type=int, default=100, metavar="N", help="input batch size for training (default: 32)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=100, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=5, metavar="N", help="number of epochs to train (default: 5)")
    parser.add_argument("--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)")

    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    args = parser.parse_args()

    warp_main(args)
    # torch_main(args)
