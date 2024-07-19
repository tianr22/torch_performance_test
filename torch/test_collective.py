import numpy as np
import torch
import torch.distributed as dist
from utils import test_bw
from platforms import DEVICE, BACKEND, set_device


def all_reduce_test(x, pg):
    def run():
        dist.all_reduce(x, group=pg)
    def bandwidth(t):
        algbw = x.numel() * 4 / t
        n = pg.size()
        busbw = algbw * 2 * (n - 1) / n
        return f' algbw {algbw * 1e-9:8.3f} GB/s, busbw {busbw * 1e-9:8.3f} GB/s'
    return run, bandwidth


def all_gather_test(y, x, pg):
    def run():
        dist.all_gather_into_tensor(y, x, group=pg)
    def bandwidth(t):
        algbw = y.numel() * 4 / t
        n = pg.size()
        busbw = algbw * (n - 1) / n
        return f' algbw {algbw * 1e-9:8.3f} GB/s, busbw {busbw * 1e-9:8.3f} GB/s'
    return run, bandwidth


def reduce_scatter_test(y, x, pg):
    def run():
        dist.reduce_scatter_tensor(y, x, group=pg)
    def bandwidth(t):
        algbw = x.numel() * 4 / t
        n = pg.size()
        busbw = algbw * (n - 1) / n
        return f' algbw {algbw * 1e-9:8.3f} GB/s, busbw {busbw * 1e-9:8.3f} GB/s'
    return run, bandwidth


def all_to_all_test(x, pg):
    def run():
        dist.all_to_all_single(x, x, group=pg)
    def bandwidth(t):
        algbw = x.numel() * 4 / t
        return f' bw {algbw * 1e-9:8.3f} GB/s'
    return run, bandwidth


GROUPS = [
    [[0, 1, 2, 3, 4, 5, 6, 7]],
    [[0, 1, 2, 3], [4, 5, 6, 7]],
    [[0, 1, 4, 5], [2, 3, 6, 7]],
    [[0, 2, 4, 6], [1, 3, 5, 7]],
    [[0, 1], [2, 3], [4, 5], [6, 7]],
    [[0, 2], [1, 3], [4, 6], [5, 7]],
    [[0, 3], [1, 2], [4, 7], [5, 6]],
    [[0, 4], [1, 5], [2, 6], [3, 7]],
]


def main(dim):
    dist.init_process_group(backend=BACKEND)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    set_device(rank % 8)
    x = torch.randn(dim, device=DEVICE)

    for pgs in GROUPS:
        pg = None
        for gi in pgs:
            gin = np.array(gi)
            ranks = []
            for i in range(world_size // 8):
                ranks += (gin + (i * 8)).tolist()
            group = dist.new_group(ranks)
            if rank in ranks:
                pg = group
        y = torch.randn(dim // pg.size(), device=DEVICE)
        log_string = f'Group {pgs} x {world_size // 8}'
        log_string += ' ' * (50 - len(log_string))
        tm, bw = test_bw(*all_reduce_test(x, pg))
        log_string += f' | all_reduce mean time {tm * 1e3:6.1f} ms, ' + bw
        tm, bw = test_bw(*all_gather_test(x, y, pg))
        log_string += f' | all_gather mean time {tm * 1e3:6.1f} ms, ' + bw
        tm, bw = test_bw(*reduce_scatter_test(y, x, pg))
        log_string += f' | reduce_scatter mean time {tm * 1e3:6.1f} ms, ' + bw
        tm, bw = test_bw(*all_to_all_test(x, pg))
        log_string += f' | a2a mean time {tm * 1e3:6.1f} ms, ' + bw
        if rank == 0:
            print(log_string)


if __name__ == "__main__":
    main(512 * 1024 * 1024)
