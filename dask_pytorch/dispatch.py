"""
This module contains the user-facing API to submit PyTorch jobs to a Dask cluster
"""

import os
from typing import List, Callable, Tuple, Any
from dask.distributed import Client
import torch.distributed as dist


def _get_worker_info(client: Client) -> Tuple[List[str], str]:
    """
    returns a list of workers (sorted), and the DNS name for the master host
    The master is the 0th worker's host
    """
    workers = client.scheduler_info()["workers"]
    worker_keys = sorted(workers.keys())
    host = workers[worker_keys[0]]["host"]
    return worker_keys, host


def run(client: Client, pytorch_function: Callable, *args, **kwargs):
    """
    Dispatch a pytorch function over a dask cluster, and returns a list of futures
    for the resulting tasks
    """
    worker_keys, host = _get_worker_info(client)
    world_size = len(worker_keys)
    port = 23456  # pick a free port?

    futures = [
        client.submit(
            dispatch_with_ddp,
            pytorch_function=pytorch_function,
            master_addr=host,
            master_port=port,
            rank=idx,
            world_size=world_size,
            backend="nccl",
            *args,
            **kwargs
        )
        for idx, w in enumerate(worker_keys)
    ]

    return futures


def dispatch_with_ddp(
    pytorch_function: Callable,
    master_addr: str,
    master_port: int,
    rank: int,
    world_size: int,
    backend: str = "nccl",
    *args,
    **kwargs
) -> Any:
    """
    runs a pytorch function, setting up torch.distributed before execution
    and tearing it down afterwards.
    """
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
        dist.init_process_group(backend=backend)
        val = pytorch_function(*args, **kwargs)
    finally:
        dist.destroy_process_group()
    return val
