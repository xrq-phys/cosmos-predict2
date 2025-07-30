from typing import Optional, List

import torch
import torch.nn.functional as F
import torch.distributed as dist

from .natten_interface import NeighborhoodAttentionRunner
from .cp_comms import get_cp_procgrp, get_cp_stream


# Util function to perform weighted sum according to LSE
# LSE tensors will have shape (B, S, H)
# Outputs will always be in float32
@torch.jit.script
def _reaccum_o_according_to_lse(out1, out1_lse, out2, out2_lse):
    log_sumall_exp = out2_lse + F.softplus(out1_lse - out2_lse)
    out1_qse = torch.exp(out1_lse - log_sumall_exp)
    out2_qse = torch.exp(out2_lse - log_sumall_exp)
    out1 = out1.to(torch.float32) * out1_qse.unsqueeze(-1)
    out2 = out2.to(torch.float32) * out2_qse.unsqueeze(-1)
    return out1 + out2, log_sumall_exp

def ring_netten_exec(
    query, key, value, stream,
    cp_rank: int = 0,
    cp_size: int = 1,
    cp_group: Optional[List[int]] = None,
    **natten_kwargs,
):
    na_dim = query.dim() - 3  # batch, heads, head_dim
    latent_shape = query.shape[1:na_dim+1]

    assert na_dim in [1, 2, 3], "Not 1D/2D/3D latent space."
    if query.shape[0] != 1:
        raise NotImplementedError("Batching unsupported yet.")
    if cp_size > 1 and cp_size % 2 != 0:
        raise NotImplementedError("Ring attention not implemented for odd number of participants.")

    # Communicator & process groups
    scomm = get_cp_stream()
    scomm.wait_stream(stream)
    cp_active = cp_size > 1 and cp_group is not None
    procgrp = get_cp_procgrp(cp_group) if cp_active else None

    # Determine the neighbouring nodes to send / receive tensors
    recv_from = (cp_rank + 1) % cp_size
    send_to = (cp_rank - 1 + cp_size) % cp_size

    # Allocate transceiving buffers
    key_remote = torch.empty_like(key)
    value_remote = torch.empty_like(value)

    # CUDA events
    compute_finish_events = []

    with torch.cuda.stream(stream), torch.cuda.nvtx.range('Ring Attention local chunks'):
        out_local_info = NeighborhoodAttentionRunner.run(
            query=query, key=key, value=value,
            return_lse=cp_active,
            **natten_kwargs,
        )
        if cp_active:
            out_local, out_local_lse = out_local_info
        else:
            out_local, out_local_lse = out_local_info, torch.empty((0, 0, 0, 0, 0, 0))
        # Merging sometimes casts float32
        out_dtype = out_local.dtype

        # Pin the event when each attention finishes
        compute_finish_events.append(torch.cuda.Event())
        compute_finish_events[-1].record()

    if cp_active:
        for ipeer in range(cp_size-1):

            with torch.cuda.stream(scomm), torch.cuda.nvtx.range('Exchange peer KV blocks'):
                # Delayed event wait (Pingpong)
                if len(compute_finish_events) > 1:
                    compute_finish_events[0].wait()
                    compute_finish_events.pop(0)

                # Send to / receive from peer procs
                req_r = dist.batch_isend_irecv([
                    dist.P2POp(dist.isend, key, send_to, group=procgrp),
                    dist.P2POp(dist.isend, value, send_to, group=procgrp),
                    dist.P2POp(dist.irecv, key_remote, recv_from, group=procgrp),
                    dist.P2POp(dist.irecv, value_remote, recv_from, group=procgrp),
                ])
                [ r.wait() for r in req_r ]

            with torch.cuda.stream(stream), torch.cuda.nvtx.range('Ring Attention remote chunks'):
                # Wait for peer KV chunks
                stream.wait_stream(scomm)

                # Received buffer will be sent in the next iteration
                # Consumed buffer will be reused as receiving buffer
                key, key_remote = key_remote, key
                value, value_remote = value_remote, value

                # Compute contributions from peer KV chunks
                out_remote, out_remote_lse = NeighborhoodAttentionRunner.run(
                    query=query, key=key, value=value,
                    return_lse=True,
                    **natten_kwargs,
                )

                # Pin the event when each attention finishes
                compute_finish_events.append(torch.cuda.Event())
                compute_finish_events[-1].record()

                out_merge, out_merge_lse = _reaccum_o_according_to_lse(
                    out_local.flatten(1, na_dim), out_local_lse.flatten(1, na_dim),
                    out_remote.flatten(1, na_dim), out_remote_lse.flatten(1, na_dim)
                )
                out_local = out_merge.unflatten(1, latent_shape)
                out_local_lse = out_merge_lse.unflatten(1, latent_shape)

    with torch.cuda.stream(stream):
        return out_local.to(out_dtype)