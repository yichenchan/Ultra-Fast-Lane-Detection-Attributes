import torch, os
import numpy as np

import torchvision.transforms as transforms
import data.mytransforms as mytransforms
from data.constant import tusimple_row_anchor, culane_row_anchor
from data.dataset import LaneDataset

def get_dataloader(batch_size, data_root, num_x_grid, dataset, augment, use_aux, distributed, num_lanes, resized_width, resized_height):

    train_dataset = LaneDataset(data_root,
                                os.path.join(data_root, dataset),
                                augment=augment,
                                row_anchor=culane_row_anchor,
                                num_x_grid=num_x_grid, 
                                num_lanes=num_lanes,
                                resized_width=resized_width,
                                resized_height=resized_height,
                                use_aux=use_aux)

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = torch.utils.data.RandomSampler(train_dataset)

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler = sampler, num_workers=8)

    return dataloader

class SeqDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    '''
    Change the behavior of DistributedSampler to sequential distributed sampling.
    The sequential sampling helps the stability of multi-thread testing, which needs multi-thread file io.
    Without sequentially sampling, the file io on thread may interfere other threads.
    '''
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas, rank, shuffle)
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))


        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size


        num_per_rank = int(self.total_size // self.num_replicas)

        # sequential sampling
        indices = indices[num_per_rank * self.rank : num_per_rank * (self.rank + 1)]

        assert len(indices) == self.num_samples

        return iter(indices)