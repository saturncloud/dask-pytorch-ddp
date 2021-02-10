"""
This module contains the user-facing API to submit PyTorch jobs to a Dask cluster
"""

import os
from typing import List, Callable, Any, Dict
from dask.distributed import Client
import torch.distributed as dist


def _get_worker_info(client: Client) -> List[Dict]:
    """
    returns a list of workers (sorted), and the DNS name for the master host
    The master is the 0th worker's host
    """
    workers = client.scheduler_info()["workers"]
    worker_keys = sorted(workers.keys())
    workers_by_host: Dict[str, List[str]] = {}
    for key in worker_keys:
        worker = workers[key]
        host = worker["host"]
        workers_by_host.setdefault(host, []).append(key)
    host = workers[worker_keys[0]]["host"]
    all_workers = []
    global_rank = 0
    for host in sorted(workers_by_host.keys()):
        local_rank = 0
        for worker in workers_by_host[host]:
            all_workers.append(
                dict(
                    worker=worker,
                    local_rank=local_rank,
                    global_rank=global_rank,
                    host=host,
                )
            )
            local_rank += 1
            global_rank += 1
    return all_workers


def run(
    client: Client,
    pytorch_function: Callable,
    *args,
    backend: str = "nccl",
    pass_local_rank: bool = False,
    **kwargs
):
    """
    Dispatch a pytorch function over a dask cluster, and returns a list of futures
    for the resulting tasks
    """
    all_workers = _get_worker_info(client)
    world_size = len(all_workers)
    port = 23456  # pick a free port?
    host = all_workers[0]["host"]
    futures = []
    for worker in all_workers:
        if pass_local_rank:
            fut = client.submit(
                dispatch_with_ddp,
                pytorch_function=pytorch_function,
                master_addr=host,
                master_port=port,
                rank=worker["global_rank"],
                world_size=world_size,
                *args,
                local_rank=worker["local_rank"],
                backend=backend,
                workers=[worker["worker"]],
                **kwargs
            )
        else:
            fut = client.submit(
                dispatch_with_ddp,
                pytorch_function=pytorch_function,
                master_addr=host,
                master_port=port,
                rank=worker["global_rank"],
                world_size=world_size,
                *args,
                backend=backend,
                workers=[worker["worker"]],
                **kwargs
            )
        futures.append(fut)
    return futures


# pylint: disable=too-many-arguments
def dispatch_with_ddp(
    pytorch_function: Callable,
    master_addr: Any,
    master_port: Any,
    rank: Any,
    world_size: Any,
    *args,
    backend: str = "nccl",
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
