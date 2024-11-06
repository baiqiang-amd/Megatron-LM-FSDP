from .. import parallel_state 
import logging
from typing import Optional

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    ShardingStrategy,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch
    )
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from ..transformer.module import MegatronModule
from ..transformer.transformer_config import TransformerConfig
from ..utils import log_single_rank
from .fully_sharded_data_parallel_config import FullyShardedDataParallelConfig

logger = logging.getLogger(__name__)

def wrap_model_with_fsdp(
    model: torch.nn.Module,
    config: TransformerConfig,
    fsdp_config: FullyShardedDataParallelConfig,
) -> FSDP:
    """
    Wraps a PyTorch model with Fully Sharded Data Parallel (FSDP).

    Args:
        model (torch.nn.Module): The model to be wrapped.
        config (TransformerConfig): Transformer config object.
        fsdp_config (FullyShardedDataParallelConfig): FSDP config object.

    Returns:
        FSDP: The wrapped model.
    """
    fsdp_kwargs = {
        # "sharding_strategy": fsdp_config.sharding_strategy,
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "mixed_precision": MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16),
        # "mixed_precision": fsdp_config.mixed_precision,
        "cpu_offload": fsdp_config.cpu_offload,
        "backward_prefetch": fsdp_config.backward_prefetch,
        "limit_all_gathers": fsdp_config.limit_all_gathers,
    }

    log_single_rank(
        logger,
        logging.INFO,
        f"Wrapping model with FSDP using config: {fsdp_kwargs}",
    )
    data_parallel_group = parallel_state.get_data_parallel_group(
        with_context_parallel=False # Now just playing it simple
    )
    # wrapped_model = FSDP(model, process_group=(data_parallel_group, data_parallel_group), **fsdp_kwargs)
    wrapped_model = FSDP(model, **fsdp_kwargs)

    return wrapped_model

class FullyShardedDataParallel(MegatronModule):
    """
    A wrapper class for Fully Sharded Data Parallel (FSDP) in Megatron-Core.

    This class provides a simple interface to wrap a model with FSDP using good defaults.
    It also includes methods for saving and loading checkpoints compatible with FSDP.

    Args:
        config (TransformerConfig): Transformer config object.
        model (torch.nn.Module): The model to be wrapped with FSDP.
        fsdp_config (FullyShardedDataParallelConfig): FSDP config object.
    """

    def __init__(
        self,
        config: TransformerConfig,
        model: torch.nn.Module,
        fsdp_config: Optional[FullyShardedDataParallelConfig] = None,
    ):
        super().__init__(config=config)
        self.config = config
        self.config.fsdp = True
        if fsdp_config is None:
            fsdp_config = FullyShardedDataParallelConfig()
        self.module = wrap_model_with_fsdp(model, config, fsdp_config)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self.module.load_state_dict(state_dict, strict)

    @staticmethod
    def save_model_checkpoint(model, checkpoint_path):
        """
        Save the FSDP model checkpoint.

        Args:
            model (FullyShardedDataParallel): The FSDP-wrapped model.
            checkpoint_path (str): Path to save the checkpoint.
        """
        with FSDP.state_dict_type(model.model, StateDictType.FULL_STATE_DICT):
            state_dict = model.state_dict()
        torch.save(state_dict, checkpoint_path)

    @staticmethod
    def load_model_checkpoint(model, checkpoint_path):
        """
        Load the FSDP model checkpoint.

        Args:
            model (FullyShardedDataParallel): The FSDP-wrapped model.
            checkpoint_path (str): Path to load the checkpoint from.
        """
        state_dict = torch.load(checkpoint_path)
        with FSDP.state_dict_type(model.model, StateDictType.FULL_STATE_DICT):
            model.load_state_dict(state_dict)


    def finish_grad_sync(self):
        pass

# #Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

# import logging
# from typing import Optional
# from abc import ABC, abstractmethod
# import torch
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# from torch.distributed.fsdp import (
#     ShardingStrategy,
#     CPUOffload,
#     MixedPrecision,
#     BackwardPrefetch,

# )
# from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

# # from .module import MegatronModule
# from ..transformer.module import MegatronModule
# from .fully_sharded_data_parallel_config import FullyShardedDataParallelConfig

# logger = logging.getLogger(__name__)

# def print_rank_0(message):
#     """If distributed is initialized, print only on rank 0."""
#     if torch.distributed.is_initialized():
#         if torch.distributed.get_rank() == 0:
#             print(message, flush=True)
#     else:
#         print(message, flush=True)

# def wrap_model_with_fsdp(
#     model: torch.nn.Module,
#     args,
#     data_parallel_group,
#     # fsdp_config: FullyShardedDataParallelConfig,
# ) -> FSDP:
#     """
#     Wraps a PyTorch model with Fully Sharded Data Parallel (FSDP).

#     Args:
#         model (torch.nn.Module): The model to be wrapped.
#         config (TransformerConfig): Transformer config object.
#         fsdp_config (FullyShardedDataParallelConfig): FSDP config object.

#     Returns:
#         FSDP: The wrapped model.
#     # """
#     fsdp_kwargs = {
#         "sharding_strategy": ShardingStrategy.HYBRID_SHARD,
#         "mixed_precision": MixedPrecision(
#             param_dtype=torch.bfloat16,
#             reduce_dtype=torch.bfloat16,
#             buffer_dtype=torch.bfloat16),
#         "cpu_offload": None,
#         "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
#         "limit_all_gathers": False,
#     }

#     print_rank_0(f"Wrapping model with FSDP using config")
#     print_rank_0(f"Data_parallel_group: {data_parallel_group}")
#     data_parallel_world_size = torch.distributed.get_world_size(group=data_parallel_group)
#     print(f"Data_parallel_world_size: {data_parallel_world_size}")
#     wrapped_model = FSDP(model, process_group=(data_parallel_group, data_parallel_group), **fsdp_kwargs)

#     return wrapped_model



# class FullyShardedDataParallelBase(MegatronModule, ABC):
#     """Abstract class for DDP."""

#     def __init__(self, module):
#         super(FullyShardedDataParallelBase, self).__init__()
#         # Keep a pointer to the model.
#         self.module = module

#     # @abstractmethod
#     # def sync_gradients(self):
#     #     pass

#     def forward(self, *inputs, **kwargs):
#         return self.module(*inputs, **kwargs)

#     def state_dict(self, prefix='', keep_vars=False):
#         return self.module.state_dict(prefix=prefix, keep_vars=keep_vars)

#     # def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
#     #     return self.module.state_dict_for_save_checkpoint(prefix=prefix, keep_vars=keep_vars)

#     def load_state_dict(self, state_dict, strict=True):
#         self.module.load_state_dict(state_dict, strict=strict)



# class FullyShardedDataParallel(FullyShardedDataParallelBase):
#     """
#     A wrapper class for Fully Sharded Data Parallel (FSDP) in Megatron-Core.

#     This class provides a simple interface to wrap a model with FSDP using good defaults.
#     It also includes methods for saving and loading checkpoints compatible with FSDP.

#     Args:
#         config (TransformerConfig): Transformer config object.
#         model (torch.nn.Module): The model to be wrapped with FSDP.
#         fsdp_config (FullyShardedDataParallelConfig): FSDP config object.
#     """

#     def __init__(
#         self,
#         module: torch.nn.Module,
#         args,
#         data_parallel_group,
#         accumulate_allreduce_grads_in_fp32,
#         overlap_grad_reduce,
#         use_distributed_optimizer,
#     ):
#         super(FullyShardedDataParallel, self).__init__(module)
#         self.args = args
#         # fsdp_config = FullyShardedDataParallelConfig()

#         # self.model = wrap_model_with_fsdp(model, args, fsdp_config)
#         self.model = wrap_model_with_fsdp(module, args, data_parallel_group)

#     def forward(self, *args, **kwargs):
#         return self.model(*args, **kwargs)

#     def state_dict(self, destination=None, prefix='', keep_vars=False):
#         return self.model.state_dict(destination, prefix, keep_vars)

#     def load_state_dict(self, state_dict, strict=True):
#         return self.model.load_state_dict(state_dict, strict)

#     @staticmethod
#     def save_model_checkpoint(model, checkpoint_path):
#         """
#         Save the FSDP model checkpoint.

#         Args:
#             model (FullyShardedDataParallel): The FSDP-wrapped model.
#             checkpoint_path (str): Path to save the checkpoint.
#         """
#         with FSDP.state_dict_type(model.model, StateDictType.FULL_STATE_DICT):
#             state_dict = model.state_dict()
#         torch.save(state_dict, checkpoint_path)

#     @staticmethod
#     def load_model_checkpoint(model, checkpoint_path):
#         """
#         Load the FSDP model checkpoint.

#         Args:
#             model (FullyShardedDataParallel): The FSDP-wrapped model.
#             checkpoint_path (str): Path to load the checkpoint from.
#         """
#         state_dict = torch.load(checkpoint_path)
#         with FSDP.state_dict_type(model.model, StateDictType.FULL_STATE_DICT):
#             model.load_state_dict(state_dict)

# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.