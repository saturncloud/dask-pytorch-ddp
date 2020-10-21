# dask-pytorch

<!-- ![GitHub Actions](https://github.com/saturncloud/dask-pytorch/workflows/GitHub%20Actions/badge.svg) [![PyPI Version](https://img.shields.io/pypi/v/prefect-saturn.svg)](https://pypi.org/project/prefect-saturn) -->

`dask-pytorch` is a Python package that makes it easy to train PyTorch models on dask clusters using distributed data paralllel.

```
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

device = torch.cuda()
model = ToyModel()
model.to(device)
optimizer = ...
data_loader = ...

for epoch in range(10):
    for data in data_loader():
        outputs = model(data)
        loss = loss_fn(outputs)
        loss.backwards()
        optimizer.step()
```

```
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def run():
    device = torch.cuda()
    model = ToyModel()
    model = DDP(model, device_ids=[0])
    model.to(device)

    optimizer = pass
    data_loader = pass

    for epoch in range(10):
        for data in data_loader():
            outputs = model(data)
            loss = loss_fn(outputs)
            loss.backwards()
            optimizer.step()
```
