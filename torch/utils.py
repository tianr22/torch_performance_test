import time
import torch
from platforms import synchronize
import argparse


def test_bw(e, b, nt=8, nc=5):
    ts = []
    for _ in range(nt):
        synchronize()
        t0 = time.time()
        e()
        synchronize()
        t1 = time.time()
        ts.append(t1 - t0)
    tm = sum(ts[-nc:]) / nc
    bw = b(tm)
    return tm, bw


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--model_size", type=int, default=7)
    args = parser.parse_args()
    return args
