import os

from dask.distributed import as_completed


def run(client, pytorch_function, *args, **kwargs):
    sync = kwargs.get('sync', True)
    workers = c.scheduler_info()['workers']
    worker_keys = sorted(workers.keys())
    world_size = len(worker_keys)
    host = workers[worker_keys[0]]['host']
    port = 23456  # TODO somehow pick a free port?
    fut_to_index = []

    for idx, w in enumerate(worker_keys):
        fut = client.submit(
            dispatch_with_ddp, pytorch_function, host, port, rank, world_size,
            *args, **kwargs
        )
        fut_to_index[fut] = idx

    if not sync:
        return [fut_to_index[x] for x in range(len(worker_keys))]

    fut_to_result = []

    for fut in as_completed(fut_to_index):
        result = fut.result()
        fut_to_result[fut] = result

    return [fut_to_result[fut_to_index[x]] for x in range(len(worker_keys))]


def dispatch_with_ddp(pytorch_function, master_addr, master_port, rank, world_size, *args, **kwargs):
    # These are the parameters used to initialize the process group
    master_addr = str(master_addr)
    master_port = str(master_port)
    rank = str(rank)
    world_size = str(world_size)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['RANK'] = rank
    os.environ['WORLD_SIZE'] = world_size

    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    dist.init_process_group(backend="nccl")

    val = pytorch_function(*args, **kwargs)

    dist.destroy_process_group()
    return val
