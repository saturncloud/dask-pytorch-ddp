import os
from typing import List, Callable
from dask.distributed import as_completed
from dask.distributed import Client
import torch.distributed as dist


def get_worker_info(client: Client) -> List[str]:
    workers = client.scheduler_info()["workers"]
    worker_keys = sorted(workers.keys())
    host = workers[worker_keys[0]]["host"]
    return worker_keys, host


def run(client: Client, pytorch_function: Callable, *args, **kwargs):
    sync = kwargs.pop("sync", True)
    worker_keys, host = get_worker_info(client)
    world_size = len(worker_keys)
    port = 23456  # TODO somehow pick a free port?

    index_to_fut = {}
    for idx, w in enumerate(worker_keys):
        fut = client.submit(
            dispatch_with_ddp, pytorch_function, host, port, idx, world_size, *args, **kwargs
        )
        index_to_fut[idx] = fut

    if not sync:
        return [index_to_fut[x] for x in range(len(worker_keys))]

    fut_to_result = {}

    for fut in as_completed(index_to_fut.values()):
        result = fut.result()
        fut_to_result[fut] = result

    return [fut_to_result[index_to_fut[x]] for x in range(len(worker_keys))]


def dispatch_with_ddp(
    pytorch_function, master_addr, master_port, rank, world_size, *args, **kwargs
):
    # These are the parameters used to initialize the process group
    master_addr = str(master_addr)
    master_port = str(master_port)
    rank = str(rank)
    world_size = str(world_size)

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = rank
    os.environ["WORLD_SIZE"] = world_size

    try:
        dist.init_process_group(backend="nccl")
        val = pytorch_function(*args, **kwargs)
    finally:
        dist.destroy_process_group()
    return val
