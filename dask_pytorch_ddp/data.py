"""
Utilities for loading PyTorch data in distributed environments
"""


import tempfile
from os.path import basename, dirname
from typing import List, Callable, Optional
from PIL import Image
from torch.utils.data import Dataset


"""
In the following, we are explicitly avoiding s3fs because it does not behave well with
multiprocessing (which is commonly used in PyTorch dataloaders).

https://github.com/dask/s3fs/issues/369
"""  # pylint: disable=pointless-string-statement


def _list_all_files(bucket: str, prefix: str, s3_client=None, anon=False) -> List[str]:
    """
    Get list of all files from an s3 bucket matching a certain prefix
    """
    import boto3  # pylint: disable=import-outside-toplevel
    from botocore import UNSIGNED  # pylint: disable=import-outside-toplevel
    from botocore.client import Config  # pylint: disable=import-outside-toplevel

    if s3_client is None:
        if anon:
            s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        else:
            s3_client = boto3.client("s3")

    paginator = s3_client.get_paginator("list_objects")
    all_files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        files = [x["Key"] for x in page["Contents"]]
        all_files.extend(files)
    return all_files


def _read_s3_fileobj(bucket, path, fileobj, anon=False):
    """
    read an obj from s3 to a file like object
    """
    import boto3  # pylint: disable=import-outside-toplevel
    from botocore import UNSIGNED  # pylint: disable=import-outside-toplevel
    from botocore.client import Config  # pylint: disable=import-outside-toplevel

    if anon:
        s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
    else:
        s3 = boto3.resource("s3")

    bucket = s3.Bucket(bucket)
    bucket.download_fileobj(path, fileobj)
    fileobj.seek(0)
    return fileobj


def _load_image_obj(fileobj):
    """
    turn a file like object into an image
    """
    return Image.open(fileobj).convert("RGB")


class S3ImageFolder(Dataset):
    """
    An image folder that lives in S3.  Directories containing the image are classes.
    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments

    def __init__(
        self,
        s3_bucket: str,
        s3_prefix: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        anon: Optional[bool] = False,
    ):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.anon = anon
        self.all_files = _list_all_files(s3_bucket, s3_prefix, anon=anon)
        self.classes = sorted({self._get_class(x) for x in self.all_files})
        self.class_to_idx = {k: idx for idx, k in enumerate(self.classes)}
        self.transform = transform
        self.target_transform = target_transform

    @classmethod
    def _get_class(cls, path):
        """
        parse the path to extract the class name
        """
        return basename(dirname(path))

    def __getitem__(self, idx):
        """
        get the nth (idx) image and label
        """
        path = self.all_files[idx]
        label = self.class_to_idx[self._get_class(path)]
        with tempfile.TemporaryFile() as f:
            f = _read_s3_fileobj(self.s3_bucket, path, f, self.anon)
            img = _load_image_obj(f)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        """
        total number of images
        """
        return len(self.all_files)
