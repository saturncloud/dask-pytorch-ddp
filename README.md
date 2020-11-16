# dask-pytorch

<!-- ![GitHub Actions](https://github.com/saturncloud/dask-pytorch/workflows/GitHub%20Actions/badge.svg) [![PyPI Version](https://img.shields.io/pypi/v/prefect-saturn.svg)](https://pypi.org/project/prefect-saturn) -->

`dask-pytorch` is a Python package that makes it easy to train PyTorch models on Dask clusters using distributed data parallel.  The intended scope of the project is
- bootstrapping PyTorch workers on top of a Dask cluster
- Using distributed data stores (e.g., S3) as normal PyTorch datasets
- mechanisms for tracking and logging intermediate results, training statistics, and checkpoints.

At this point, this library and examples provided are tailored to computer vision tasks, but this library is intended to be useful for any sort of PyTorch tasks. The only thing really specific to image processing is the `S3ImageFolder` dataset class. Implementing a PyTorch dataset (assuming map style random access) outside of images currently requires implementing `__getitem__(self, idx: int):` and `__len__(self):` We plan to add more varied examples for other use cases in the future, and welcome PRs extending functionality.

## Typical non-dask workflow

A typical example of non-dask PyTorch usage is as follows:

### Loading Data
Create an dataset (`ImageFolder`), and wrap it in a `DataLoader`

```python
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(250),
    transforms.ToTensor()
])

whole_dataset = ImageFolder(path, transform=transform)

batch_size = 100
num_workers = 64
indices = list(range(len(data)))
np.random.shuffle(indices)
train_idx = indices[:num]
test_idx = indices[num:num+num]

train_sampler = SubsetRandomSampler(train_idx)
train_loader = DataLoader(data, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers)
```

### Training a Model
Loop over the dataset, and train the model by stepping the optimizer

```python
device = torch.device(0)
net = models.resnet18(pretrained=False)
model = net.to(device)
device_ids = [0]

criterion = nn.CrossEntropyLoss().cuda()
lr = 0.001
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
count = 0
for epoch in range(n_epochs):
    model.train()  # Set model to training mode
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # zero the parameter gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += 1
```

## Now on Dask

With dask_pytorch and PyTorch Distributed Data Parallel, we can train on multiple workers as follows:

### Loading Data
Load the dataset from S3, and explicitly set the multiprocessing context (Dask defaults to spawn, but pytorch is generally configured to use fork)

```python
from dask_pytorch.data import S3ImageFolder

whole_dataset = S3ImageFolder(bucket, prefix, transform=transform)
train_loader = torch.utils.data.DataLoader(
    whole_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers, multiprocessing_context=mp.get_context('fork')
)
```

### Training in Parallel

Wrap the training loop in a function (and add metrics logging.  Not necessary, but very useful).  Convert the model into a PyTorch Distributed Data Parallel (`DDP`) model which knows how to sync gradients together across workers.

```python
import uuid
import pickle
import logging
import json


key = uuid.uuid4().hex
rh = DaskResultsHandler(key)

def run_transfer_learning(bucket, prefix, samplesize, n_epochs, batch_size, num_workers, train_sampler):
    worker_rank = int(dist.get_rank())
    device = torch.device(0)
    net = models.resnet18(pretrained=False)
    model = net.to(device)
    model = DDP(model, device_ids=[0])

    criterion = nn.CrossEntropyLoss().cuda()
    lr = 0.001
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    whole_dataset = S3ImageFolder(bucket, prefix, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(
        whole_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        multiprocessing_context=mp.get_context('fork')
    )
    
    count = 0
    for epoch in range(n_epochs):
        # Each epoch has a training and validation phase
        model.train()  # Set model to training mode
        for inputs, labels in train_loader:
            dt = datetime.datetime.now().isoformat()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += 1

            # statistics
            rh.submit_result(
                f"worker/{worker_rank}/data-{dt}.json",
                json.dumps({'loss': loss.item(), 'epoch': epoch, 'count': count, 'worker': worker_rank})
            )
            if (count % 100) == 0 and worker_rank == 0:
                rh.submit_result(f"checkpoint-{dt}.pkl", pickle.dumps(model.state_dict()))

```

## How does it work?

`dask-pytorch` is largely a wrapper around existing `pytorch` functionality.  `pytorch.distributed` provides infrastructure for [Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) (DDP).

In DDP, you create N workers, and the 0th worker is the "master", and coordinates the synchronization of buffers and gradients.  In SGD, gradients are normally averaged between all data points in a batch.  By running batches on multiple workers, and averaging the gradients, DDP enables you to run SGD with a much bigger batch size `(N * batch_size)`

`dask-pytorch` sets some environment variables to configure the "master" host and port, and then calls `init_process_group` before training, and calls `destroy_process_group` after training.  This is the same process normally done manually by the data scientist.

### Multi GPU machines
`dask_cuda_worker` automatically rotates `CUDA_VISIBLE_DEVICES` for each worker it creates (typically one per GPU).  As a result, your PyTorch code should always start with the 0th GPU.

For example, if I have an 8 GPU machine, the 3rd worker will have `CUDA_VISIBLE_DEVICES` set to `2,3,4,5,6,7,0,1`.  On that worker, if I call `torch.device(0)`, I will get GPU 2.

## What else?

`dask-pytorch` also implements an S3 based `ImageFolder`.  More distributed friendly datasets are planned.  `dask-pytorch`  also implements a basic results aggregation framework so that it is easy to collect training metrics across different workers.  Currently, only `DaskResultsHandler` which leverages [Dask pub-sub communication protocols][1] is implemented, but an S3 based result handler is planned.

[1]:https://docs.dask.org/en/latest/futures.html#publish-subscribe

## Some Notes

Dask generally spawns processes.  PyTorch generally forks.  When using a multiprocessing enabled data loader, it is a good idea to pass the `Fork` multiprocessing context to force the use of Forking in the data loader.

Some Dask deployments do not permit spawning processes.  To override this, you can change the [distributed.worker.daemon](https://docs.dask.org/en/latest/configuration-reference.html#distributed.worker.daemon) setting.

Environment variables are a convenient way to do this:

```
DASK_DISTRIBUTED__WORKER__DAEMON=False
```
