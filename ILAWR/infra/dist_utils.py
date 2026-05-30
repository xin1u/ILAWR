import os
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler


def setup_distributed(backend='nccl'):
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    if world_size > 1:
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_local_rank():
    return int(os.environ.get('LOCAL_RANK', 0))


def get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process():
    return get_rank() == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_tensor(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def make_distributed_sampler(dataset, shuffle=True):
    if get_world_size() > 1:
        return DistributedSampler(dataset, shuffle=shuffle)
    return None


if __name__ == '__main__':
    rank, local_rank, world_size = setup_distributed()
    print(f'rank={rank}, local_rank={local_rank}, world_size={world_size}')
    print(f'is_main={is_main_process()}')
