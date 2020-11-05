import torch
from torch.utils.data import SequentialSampler, RandomSampler
import ray
import ray_utils.queue_ray as q


def stack(values):
    print(values[0].type)
    if isinstance(values[0], (list, tuple)):
        values = list(zip(*values))
        values = tuple([torch.stack(v, dim=0) for v in values])
    else:
        values = torch.stack(values, dim=0)
    return values


def batch_gen(gen, batch_size):
    cur_batch = []
    try:
        cur_batch += [next(gen)]
        print(cur_batch[-1])
        if len(cur_batch) == batch_size:
            yield stack(cur_batch)
            cur_batch = []
    except:
        if len(cur_batch) > 0:
            return stack(cur_batch)


def f_pin(tensor):
    return tensor.pin_memory()


def shuffle_gen(dataset):
    p = torch.random.permutation(len(dataset))
    for i in p:
        yield dataset[i]


class DataLoader():
    """
    DataLoader

    a (simplified) ray-based DataLoader for PyTorch
    """
    def __init__(self, dataset, batch_size=1, pin_memory=False):
        self.dataset    = dataset
        self.batch_size = batch_size
        self.pin_memory = pin_memory

        self.pin_fun = f_pin if pin_memory else q.pass_through


    def __iter__(self):
        queue = q.Queue()
        q.put_queue.remote(shuffle_gen, self.dataset, queue, self.pin_fun)
        return batch_gen(q.get_queue(queue, 5), self.batch_size)
