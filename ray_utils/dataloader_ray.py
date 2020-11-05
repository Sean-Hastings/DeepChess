import torch
from torch.utils.data import SequentialSampler, RandomSampler
import ray
ray.init()


class DataLoader():
    """
    DataLoader

    a (simplified) ray-based DataLoader for PyTorch
    """
    __init__(self,
             dataset: torch.utils.data.dataset.Dataset[T_co],
             batch_size: Optional[int] = 1,
             shuffle: bool = False,
             pin_memory: bool = False,
             queue_size: Optional[int] = 1):

        self.dataset    = dataset
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.pin_memory = pin_memory
        self.queue_size = queue_size

        # Copied & modified from torch.utils.data.DataLoader
        if isinstance(dataset, IterableDataset) and shuffle is not False:
            raise ValueError(
                "DataLoader with IterableDataset: expected unspecified "
                "shuffle option, but got shuffle={}".format(shuffle))

        if shuffle:
            self.sampler = RandomSampler(dataset)
        else:
            self.sampler = SequentialSampler(dataset)
