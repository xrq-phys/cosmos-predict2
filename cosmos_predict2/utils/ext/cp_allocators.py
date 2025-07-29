from functools import reduce
import torch

class BaseAllocator:
    def __call__(self, shape, dtype : torch.dtype = torch.float32, device=None):
        return torch.empty(shape, dtype=dtype, device=device)

class FromPoolAllocator(BaseAllocator):
    def __init__(self, pool: torch.Tensor):
        self.pool = pool.view(torch.uint8).flatten()
        self.alloc = 0

    def __call__(self, shape, dtype : torch.dtype = torch.float32, device=None):
        device_req = torch.device(device)
        device_pool = torch.device(self.pool.device)
        assert device_req == device_pool, "Requested device is different from pool device"

        nbytes = reduce(lambda x, y: x * y, tuple(shape)) * dtype.itemsize
        alloc = self.alloc
        self.alloc = alloc + nbytes
        return self.pool[alloc:self.alloc].view(dtype).view(shape)

    @staticmethod
    def cdiv(a: int, b: int):
        return (a + b - 1) // b

    def allocated_buffer(self, alignment: int = 1):
        return self.pool[:FromPoolAllocator.cdiv(self.alloc, alignment) * alignment]
