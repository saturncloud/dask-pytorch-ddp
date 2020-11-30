"""
Infrastructure for retrieving and logging intermediate results from pytorch training jobs.

Currently using dask pub/sub, but will create an S3 version in the future.
"""
import uuid
import logging
import os
from typing import List, Optional
from os.path import join, exists, dirname

from distributed.pubsub import Pub, Sub
from distributed.utils import TimeoutError as DistributedTimeoutError
from distributed.client import wait, FIRST_COMPLETED, Future


logger = logging.getLogger(__name__)


class DaskResultsHandler:
    """
    This class use Dask pubsub infra to pass intermediate results back from PyTorch
    jobs to the client.
    """

    def __init__(self, pub_sub_key: Optional[str] = None):
        """
        pub_sub_key is an arbitrary string (topic) for the pub sub channel.
        It's a good idea to change it.  Sometimes old topics can get "clogged"
        """
        if pub_sub_key is None:
            pub_sub_key = uuid.uuid4().hex
        self.pub_sub_key = pub_sub_key

    @classmethod
    def _get_all(cls, sub: Sub):
        while True:
            try:
                yield sub.get(timeout=1.0)
            except DistributedTimeoutError:
                break

    def _get_results(self, futures: List[Future], raise_errors: bool = True):
        sub = Sub(self.pub_sub_key)
        while True:
            for obj in self._get_all(sub):
                yield obj
            if not futures:
                break
            try:
                result = wait(futures, 0.1, FIRST_COMPLETED)
            except DistributedTimeoutError:
                continue

            for fut in result.done:
                try:
                    fut.result()
                except Exception as e:  # pylint: disable=broad-except
                    logging.exception(e)
                    if raise_errors:
                        raise
            futures = result.not_done

    def process_results(
        self, prefix: str, futures: List[Future], raise_errors: bool = True
    ) -> None:
        """
        Process the intermediate results:
        result objects will be dictionaries of the form {'path': path, 'data': data}
        As results come in, data will be written to f"prefix/{path}"

        prefix:  directory where you want results to be written
        futures:  list of futures for your jobs (output of dask_pytorch_ddp.dispatch.run)
        raise_errors:  If any of the jobs fail, either raise an exception, or log it and continue.
        """
        for result in self._get_results(futures, raise_errors=raise_errors):
            path = result["path"]
            data = result["data"]
            fpath = join(prefix, path)
            if not exists(dirname(fpath)):
                os.makedirs(dirname(fpath))
            if isinstance(data, str):
                data = data.encode("utf-8")
            with open(fpath, "wb+") as f:
                f.write(data)

    def submit_result(self, path: str, data: str):
        """
        To be used in jobs.  Call this function with a path, and some data.
        Client will write {data} to a file at {path}
        """
        pub = Pub(self.pub_sub_key)
        pub.put({"path": path, "data": data})
