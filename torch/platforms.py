import torch
DEVICE = torch.device('cpu')


try:
    import torch_musa
    DEVICE = torch.device('musa')
    BACKEND = 'mccl'
    def set_device(rank):
        torch.musa.set_device(rank)
    def synchronize():
        torch.musa.synchronize()
    PEAK = 98
except Exception:
    pass



try:
    if not torch.cuda.is_available():
        raise NotImplementedError
    DEVICE = torch.device('cuda')
    BACKEND = 'nccl'
    def set_device(rank):
        torch.cuda.set_device(rank)
    def synchronize():
        torch.cuda.synchronize()
    mapping = {
            "Tesla V100": 112,
            "NVIDIA A100": 312,
            "NVIDIA H100": 1513,
            "Device 4001": 240, # MXC500
    }
    PEAK = 0.1
    name = torch.cuda.get_device_name()
    for n in mapping:
        if name.startswith(n):
            PEAK = mapping[n]
except Exception:
    pass


if torch.__version__.find('SWT') > -1:
    PEAK = 55
    BACKEND = 'mpi'
    def synchronize():
        pass
    def set_device(_):
        pass
