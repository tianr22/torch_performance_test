import torch
from utils import test_bw, parse_input
from platforms import DEVICE
from model_info import ModelInfo


use_native_rms = hasattr(torch, 'rms_norm')


# TODO not able to build apex cpp extention for Fused cuda kernel RMSNorm
# Steps performed, 1. copy https://github.com/NVIDIA/apex/blob/master/apex/normalization/fused_layer_norm.py, https://github.com/NVIDIA/apex/blob/master/csrc/layer_norm_cuda.cpp, https://github.com/NVIDIA/apex/blob/master/csrc/layer_norm_cuda_kernel.cu to ./megatron/model/fused_layer_norm.py, ./megatron/fused_kernels/layer_norm_cuda.cpp, ./megatron/fused_kernels/layer_norm_cuda_kernel.cu, and update ./megatron/fused_kernels/__init__.py accordingly 2. use below line to import MixedFusedRMSNorm
# torch.nn.LayerNorm is slower than apex.FusedLayerNorm for shapes typical in NLP models. For example: (512, 16, 1024) with normalization over the last dimension is slower using torch.nn.LayerNorm
# from megatron.model.fused_layer_norm import MixedFusedRMSNorm as RMSNorm # for cuda
class RMSNorm(torch.nn.Module):  # for cpu
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        """
        BaichuanRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, **kwargs))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        if use_native_rms:
            return torch.rms_norm(hidden_states, self.weight,
                                  self.variance_epsilon)[0]
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        hidden_states = self.weight * hidden_states

        return hidden_states




def test_rms(l, x):
    def e():
        y = l(x)
        y.sum().backward()

    def t(elapsed_time):
        bw = x.numel() / elapsed_time
        return f'alg bw {bw * 2 * 1e-9:5.1f} GB/s'

    return e, t


def main(ms, bs, sl, dtype=torch.bfloat16):
    info = ModelInfo(ms)
    hs = info.hidden_dim
    l = RMSNorm(hs, dtype=dtype, device=DEVICE)
    x = torch.randn(bs, sl, hs, device=DEVICE, dtype=dtype)

    log_string = f'RMS {bs:4}x{sl:4}x{hs:5}'
    tm, bw = test_bw(*test_rms(l, x))
    log_string += f' | mean time {tm * 1e3:6.1f} ms, ' + bw

    print(log_string)


if __name__ == '__main__':
    args = parse_input()

    main(args.model_size, args.batch_size, args.seq_len)
