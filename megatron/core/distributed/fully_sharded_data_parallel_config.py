# from dataclasses import dataclass
# from typing import Optional

# import torch
# from torch.distributed.fsdp.fully_sharded_data_parallel import (
#     BackwardPrefetch,
#     CPUOffload,
#     MixedPrecision,
#     ShardingStrategy,
# )

# @dataclass
# class FullyShardedDataParallelConfig:
#     """Configuration for FullyShardedDataParallel."""

#     sharding_strategy: ShardingStrategy = ShardingStrategy.HYBRID_SHARD
#     """The sharding strategy to use for FSDP."""

#     mixed_precision: Optional[MixedPrecision] = None
#     """Mixed precision settings for FSDP."""

#     cpu_offload: Optional[CPUOffload] = None
#     """CPU offloading settings for FSDP."""

#     backward_prefetch: BackwardPrefetch = BackwardPrefetch.BACKWARD_PRE
#     """The backward prefetch strategy to use."""

#     limit_all_gathers: bool = True
#     """Whether to limit all-gathers (potentially saves memory)."""

#     def __post_init__(self):
#         if self.mixed_precision is None:
#             self.mixed_precision = MixedPrecision(
#                 param_dtype=torch.bfloat16,
#                 reduce_dtype=torch.bfloat16,
#                 buffer_dtype=torch.bfloat16,
#             )
#         if self.cpu_offload is None:
#             self.cpu_offload = CPUOffload(offload_params=False)

# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Optional

import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)

@dataclass
class FullyShardedDataParallelConfig:
    """Configuration for FullyShardedDataParallel."""

    sharding_strategy: ShardingStrategy = ShardingStrategy.HYBRID_SHARD
    """The sharding strategy to use for FSDP."""

    mixed_precision: Optional[MixedPrecision] = None
    """Mixed precision settings for FSDP."""

    cpu_offload: Optional[CPUOffload] = None
    """CPU offloading settings for FSDP."""

    backward_prefetch: BackwardPrefetch = BackwardPrefetch.BACKWARD_PRE
    """The backward prefetch strategy to use."""

    limit_all_gathers: bool = True
    """Whether to limit all-gathers (potentially saves memory)."""

    def __post_init__(self):
        if self.mixed_precision is None:
            self.mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        if self.cpu_offload is None:
            self.cpu_offload = CPUOffload(offload_params=False)