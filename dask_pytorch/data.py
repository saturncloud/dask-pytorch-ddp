import tempfile
import logging
from os.path import basename, dirname

from torch.utils.data import Dataset


def get_all_files(bucket, prefix, s3_client=None):
    import boto3

    if s3_client is None:
        s3_client = boto3.client("s3")
    paginator = s3.get_paginator("list_objects")
    all_files = []
    for page in paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix):
        files = [x["Key"] for x in page["Contents"]]
        all_files.extend(files)
    return all_files


def read_s3_fileobj(bucket, path, fileobj):
    import boto3

    s3 = boto3.resource("s3")
    bucket = s3.Bucket(self.s3_bucket)
    bucket.download_fileobj(path, fileobj)
    fileobj.seek(0)
    return fileobj


def load_image_obj(fileobj):
    return Image.open(fileobj).convert("RGB")


class BOTOS3ImageFolder(Dataset):
    def __init__(self, s3_bucket, s3_prefix, transform=None, target_transform=None):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.all_files = get_all_files(s3_bucket, s3_prefix)
        self.classes = sorted(set([self.get_class(x) for x in self.all_files]))
        self.class_to_idx = {k: idx for idx, k in enumerate(self.classes)}
        self.transform = transform
        self.target_transform = target_transform

    @classmethod
    def get_class(cls, path):
        return basename(dirname(path))

    def __getitem__(self, idx):
        # pid = os.getpid()
        # logging.error(f'GET {idx}: {pid}')
        path = self.all_files[idx]
        label = self.class_to_idx[self.get_class(path)]
        with tempfile.TemporaryFile() as f:
            f = read_s3_fileobj(self.s3_bucket, path, f)
            img = load_image_obj(f)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, label

    def __len__(self):
        return len(self.all_files)
