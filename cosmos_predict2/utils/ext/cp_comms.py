import torch
import torch.distributed as dist
from types import SimpleNamespace
from typing import List

_sageattn_context = SimpleNamespace()
_sageattn_context.sm_version = {}
_sageattn_context.cp_procgrp = {}
_sageattn_context.cp_scomm = torch.cuda.Stream(priority=-1)

def get_cp_stream() -> torch.cuda.Stream:
    return _sageattn_context.cp_scomm

def set_cp_procgrp(cp_group_list: List[int], procgrp: dist.ProcessGroup) -> None:
    _sageattn_context.cp_procgrp[tuple(cp_group_list)] = procgrp

def get_cp_procgrp(cp_group_list: List[int]) -> dist.ProcessGroup:
    cp_group = tuple(cp_group_list)
    if cp_group not in _sageattn_context.cp_procgrp:
        raise RuntimeError(f"CP group {cp_group} not found. Forbidding creation for safety. Only comment out this error when you know what you are doing.")
        _sageattn_context.cp_procgrp[cp_group] = dist.new_group(ranks=cp_group)
    return _sageattn_context.cp_procgrp[cp_group]

def get_sm_version(rank: int) -> int:
    if rank not in _sageattn_context.sm_version:
        gpu_props = torch.cuda.get_device_properties(rank)
        sm_version = gpu_props.major * 10 + gpu_props.minor
        _sageattn_context.sm_version[rank] = sm_version
    return _sageattn_context.sm_version[rank]
