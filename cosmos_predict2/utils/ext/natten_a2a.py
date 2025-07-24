from typing import Optional, List
import torch

from .cp_comms import get_cp_procgrp, get_cp_stream
from .natten_interface import NeighborhoodAttentionRunner
from cosmos_predict2.networks.a2a_cp import _SeqAllToAllQKV, _SeqAllToAll

def a2a_netten_exec(
    query, key, value, stream,
    cp_rank: int = 0,
    cp_size: int = 1,
    cp_group: Optional[List[int]] = None,
    **natten_kwargs,
):
    na_dim = query.dim() - 3  # batch, heads, head_dim

    assert na_dim in [1, 2, 3], "Not 1D/2D/3D latent space."
    if query.shape[0] != 1:
        raise NotImplementedError("Batching unsupported yet.")
    if cp_size > 1 and cp_size % 2 != 0:
        raise NotImplementedError("Ring attention not implemented for odd number of participants.")

    # Communicator & process groups
    scomm = get_cp_stream()
    cp_active = cp_size > 1 and cp_group is not None
    procgrp = get_cp_procgrp(cp_group) if cp_active else None

    with torch.cuda.stream(stream):
        if not cp_active:
            return NeighborhoodAttentionRunner.run(
                query=query, key=key, value=value,
                **natten_kwargs,
            )

        latent_shape = list(query.shape[1:na_dim+1])
        latent_shape[0] = -1

        query, key, value = map(lambda t: t.flatten(1, na_dim), (query, key, value))
        query_layer, key_layer, value_layer = _SeqAllToAllQKV.apply(
            procgrp, query, key, value, cp_size, scomm, True
        )
        query_layer, key_layer, value_layer = map(
            lambda t: t.unflatten(1, tuple(latent_shape)),
            (query_layer, key_layer, value_layer)
        )
        context_layer = NeighborhoodAttentionRunner.run(
            query=query_layer, key=key_layer, value=value_layer,
            **natten_kwargs,
        )

        output = _SeqAllToAll.apply(procgrp, context_layer.flatten(1, na_dim), False)
        return output.unflatten(1, tuple(latent_shape))