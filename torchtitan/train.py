# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os
import time
from datetime import timedelta
from typing import Any, Generator, Iterable, Optional

import torch
from torch.distributed.elastic.multiprocessing.errors import record

import torchtitan.components.ft as ft
import torchtitan.protocols.train_spec as train_spec_module

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.metrics import (
    build_metrics_processor,
    ensure_pp_loss_visible,
)
from torchtitan.config_manager import ConfigManager, JobConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.profiling import (
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
)


class Trainer(torch.distributed.checkpoint.stateful.Stateful):
    job_config: JobConfig
    gc_handler: utils.GarbageCollection

    parallel_dims: ParallelDims
    train_spec: train_spec_module.TrainSpec
    world_mesh: torch.distributed.DeviceMesh

    dataloader: train_spec_module.BaseDataLoader
    metrics_processor: train_spec_module.MetricsProcessor
    checkpointer: CheckpointManager
    train_context: Generator[None, None, None]

    model_parts: list[torch.nn.Module]
    loss_fn: train_spec_module.LossFunction
    optimizers: train_spec_module.OptimizersContainer
    lr_schedulers: train_spec_module.LRSchedulersContainer

    pp_has_first_stage: bool
    pp_has_last_stage: bool

    device: torch.device

    # states
    step: int

    # Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
    @record
    def __init__(self, job_config: JobConfig):
        self.job_config = job_config

        logger.info(f"Starting job: {job_config.job.description}")

        if job_config.experimental.custom_import:
            importlib.import_module(job_config.experimental.custom_import)

        if job_config.job.print_args:
            logger.info(f"Running with args: {job_config.to_dict()}")

        # take control of garbage collection to avoid stragglers
        self.gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

        device_module, device_type = utils.device_module, utils.device_type
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        # Device has to be set before creating TorchFT manager.
        device_module.set_device(self.device)

        # init distributed
        world_size = int(os.environ["WORLD_SIZE"])
        parallelism_config = job_config.parallelism
        self.parallel_dims = parallel_dims = ParallelDims(
            dp_shard=parallelism_config.data_parallel_shard_degree,
            dp_replicate=parallelism_config.data_parallel_replicate_degree,
            cp=parallelism_config.context_parallel_degree,
            tp=parallelism_config.tensor_parallel_degree,
            pp=parallelism_config.pipeline_parallel_degree,
            world_size=world_size,
            enable_loss_parallel=not parallelism_config.disable_loss_parallel,
        )
        dist_utils.init_distributed(job_config)

        # build meshes
        self.world_mesh = world_mesh = parallel_dims.build_mesh(device_type=device_type)
        if parallel_dims.dp_enabled:
            dp_mesh = world_mesh["dp"]
            dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
        else:
            dp_degree, dp_rank = 1, 0

        self.ft_manager = ft.init_ft_manager(job_config)
        # If TorchFT is enabled, the dp_rank and dp_degree, which are used for
        # dataloader must be changed.
        if self.ft_manager.enabled:
            dp_degree, dp_rank = self.ft_manager.get_dp_info(dp_degree, dp_rank)

        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        dist_utils.set_determinism(
            world_mesh,
            self.device,
            job_config.training.seed,
            job_config.training.deterministic,
        )
        self.train_spec = train_spec_module.get_train_spec(job_config.model.name)

        # build dataloader
        tokenizer = (
            self.train_spec.build_tokenizer_fn(job_config)
            if self.train_spec.build_tokenizer_fn is not None
            else None
        )

        self.dataloader = self.train_spec.build_dataloader_fn(
            dp_world_size=dp_degree,
            dp_rank=dp_rank,
            tokenizer=tokenizer,
            job_config=job_config,
        )

        # build model (using meta init)
        model_cls = self.train_spec.cls
        model_args = self.train_spec.config[job_config.model.flavor]
        # set the model args from training job configs
        model_args.update_from_config(job_config, tokenizer)

        logger.info(
            f"Building {self.train_spec.name} {job_config.model.flavor} with {model_args}"
        )
        with torch.device("meta"):
            model = model_cls.from_model_args(model_args)

        # Build the collection of model converters. No-op if `model.converters` empty
        model_converters = build_model_converters(job_config, parallel_dims)
        model_converters.convert(model)

        # metrics logging
        build_metrics_processor_fn = (
            build_metrics_processor
            if self.train_spec.build_metrics_processor_fn is None
            else self.train_spec.build_metrics_processor_fn
        )
        self.metrics_processor = build_metrics_processor_fn(
            job_config, parallel_dims, model_args
        )
        color = self.metrics_processor.color

        # calculate model size and flops per token
        (
            model_param_count,
            self.metrics_processor.num_flops_per_token,
        ) = model_args.get_nparams_and_flops(model, job_config.training.seq_len)

        logger.info(
            f"{color.blue}Model {self.train_spec.name} {job_config.model.flavor} "
            f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
        )

        # move sharded model to CPU/GPU and initialize weights via DTensor
        if job_config.checkpoint.create_seed_checkpoint:
            init_device = "cpu"
            buffer_device = None
        elif job_config.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = device_type
        else:
            init_device = device_type
            buffer_device = None

        self.loss_fn = self.train_spec.build_loss_fn(job_config)

        # apply parallelisms and initialization
        if parallel_dims.pp_enabled:
            if not self.train_spec.pipelining_fn:
                raise RuntimeError(
                    f"Pipeline Parallel is enabled but {self.train_spec.name} "
                    f"does not support pipelining"
                )

            # apply both PT-D Pipeline Parallel and SPMD-style PT-D techniques
            (
                self.pp_schedule,
                self.model_parts,
                self.pp_has_first_stage,
                self.pp_has_last_stage,
            ) = self.train_spec.pipelining_fn(
                model,
                world_mesh,
                parallel_dims,
                job_config,
                self.device,
                model_args,
                self.train_spec.parallelize_fn,
                self.loss_fn,
            )
            # when PP is enabled, `model` obj is no longer used after this point,
            # model_parts is used instead
            del model

            for m in self.model_parts:
                m.to_empty(device=init_device)
                with torch.no_grad():
                    m.init_weights(buffer_device=buffer_device)
                m.train()

            # confirm that user will be able to view loss metrics on the console
            ensure_pp_loss_visible(parallel_dims, job_config, color)
        else:
            # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
            model = self.train_spec.parallelize_fn(
                model, world_mesh, parallel_dims, job_config
            )

            model.to_empty(device=init_device)
            with torch.no_grad():
                model.init_weights(buffer_device=buffer_device)
            model.train()

            self.model_parts = [model]

        if self.ft_manager.enabled:
            self.ft_manager.set_all_reduce_hook(self.model_parts)

        # initialize device memory monitor and get peak flops for MFU calculation
        device_memory_monitor = self.metrics_processor.device_memory_monitor
        gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
        logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
        device_mem_stats = device_memory_monitor.get_peak_stats()
        logger.info(
            f"{device_type.upper()} memory usage for model: "
            f"{device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)"
        )

        # build optimizer after applying parallelisms to the model
        self.optimizers = self.train_spec.build_optimizers_fn(
            self.model_parts, job_config, self.ft_manager
        )
        self.lr_schedulers = self.train_spec.build_lr_schedulers_fn(
            self.optimizers, job_config
        )
        # Post optimizer step model converters hook.
        # e.g. calculate float8 dynamic amax/scale for all-parameter for FSDP2
        # where it issues a single all-reduce for all parameters at once for better performance
        self.optimizers.register_step_post_hook(
            lambda *args, **kwargs: model_converters.post_optimizer_hook(
                self.model_parts
            )
        )
        self.metrics_processor.optimizers = self.optimizers

        # Initialize trainer states that will be saved in checkpoint.
        # These attributes must be initialized before checkpoint loading.
        self.step = 0

        self.checkpointer = CheckpointManager(
            dataloader=self.dataloader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},
            job_config=job_config,
            ft_manager=self.ft_manager,
        )

        self.train_context = dist_utils.get_train_context(
            parallel_dims.loss_parallel_enabled,
            parallelism_config.enable_compiled_autograd,
        )

        logger.info(
            "Trainer is initialized with "
            f"local batch size {job_config.training.batch_size}, "
            f"global batch size {job_config.training.batch_size * dp_degree * job_config.training.gradient_accumulation_steps}, "
            f"sequence length {job_config.training.seq_len}, "
            f"total steps {job_config.training.steps} "
            f"(warmup {job_config.lr_scheduler.warmup_steps})."
        )

    def next_batch(
        self, data_iterator: Iterable
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        data_load_start = time.perf_counter()
        batch = next(data_iterator)
        input_dict, labels = batch
        self.metrics_processor.ntokens_since_last_log += labels.numel()
        self.metrics_processor.data_loading_times.append(
            time.perf_counter() - data_load_start
        )

        device_type = utils.device_type
        for k, _ in input_dict.items():
            input_dict[k] = input_dict[k].to(device_type)
        labels = labels.to(device_type)
        return input_dict, labels

    def gradient_accumulate_train_step(self, data_iterator, grad_accum_steps: int):
        model_parts = self.model_parts
        world_mesh = self.world_mesh
        parallel_dims = self.parallel_dims
        assert not parallel_dims.pp_enabled, 'PP is not supported for gradient accumulation for now'


        self.optimizers.zero_grad()
        
        losses = [] # running loss for display
        channel_loss_dict = {}
        boundary_tensor = None
        if self.job_config.metrics.log_target_loss >= 0: # 0 for text, 1 for vq0, 2 for vq1, etc.
            boundaries = [0, self.job_config.model.text_token_cnt]
            for n_q in range(self.job_config.metrics.log_target_loss):
                boundaries.append(self.job_config.model.text_token_cnt + self.job_config.model.audio_codebook_size * (n_q+1))
            boundary_tensor = torch.tensor(boundaries, device=self.device)

        for ga_id in range(grad_accum_steps):
            inputs, labels = self.next_batch(data_iterator)
            # accumulate gradients for grad_accum_steps
            for m in model_parts:
                m.set_requires_gradient_sync(ga_id == grad_accum_steps - 1)
                # m.set_requires_gradient_sync(True)
            batch_loss = self.train_step(inputs, labels).view(labels.shape) # [micro_batch_size, seq_len]

            if boundary_tensor is not None:
                label_range_flat = torch.bucketize(labels, boundaries=boundary_tensor, right=True).view(-1) # [micro_batch_size, seq_len]
                batch_loss_flat = batch_loss.view(-1).detach() 

                range_losses = torch.zeros(len(boundaries)+1, device=self.device)
                range_valid_token_count = torch.zeros(len(boundaries)+1, device=self.device)

                range_losses.scatter_add_(0, label_range_flat, batch_loss_flat) # sum batch_loss according to label range to range_losses
                range_valid_token_count.scatter_add_(0, label_range_flat, torch.ones_like(batch_loss_flat))

                for range_id in range(1, len(boundaries)):
                    n_q = range_id - 1
                    cnt = range_valid_token_count[range_id].item()
                    if cnt > 0:
                        key_n_q = -n_q - 1 # -1 for text, -2 for vq 0, -3 for vq 1, etc.
                        range_loss = range_losses[range_id].item()
                        if key_n_q not in channel_loss_dict:
                            channel_loss_dict[key_n_q] = [range_loss, cnt]
                        else:
                            channel_loss_dict[key_n_q][0] += range_loss
                            channel_loss_dict[key_n_q][1] += cnt

            # for n_q, (start, end) in enumerate(zip(range_starts, range_ends)):
            #     range_mask = (labels >= start) & (labels < end)
            #     if range_mask.any():
            #         range_loss = batch_loss[range_mask].sum().item()
            #         range_valid_token_count = range_mask.sum().item()
            #         key_n_q = -n_q - 1 # -1 for text, -2 for vq 0, -3 for vq 1, etc.
            #         if key_n_q not in channel_loss_dict:
            #             channel_loss_dict[key_n_q] = [range_loss, range_valid_token_count]
            #         else:
            #             channel_loss_dict[key_n_q][0] += range_loss
            #             channel_loss_dict[key_n_q][1] += range_valid_token_count


            batch_loss_mean = []
            for sample_loss, channel, label in zip(batch_loss, inputs['channel'], labels): # handle each sample's channel within a batch
                sample_valid_token_count = label.ne(-1).sum().item()
                sample_loss_mean = torch.sum(sample_loss) / sample_valid_token_count # for backward calculation, has gradient
                batch_loss_mean.append(sample_loss_mean)

                channel_loss_sum = sample_loss_mean.item() * sample_valid_token_count # for channel loss display, no gradient
                channel_item = channel.item()
                if channel_item not in channel_loss_dict:
                    channel_loss_dict[channel_item] = [channel_loss_sum, sample_valid_token_count]
                else:
                    channel_loss_dict[channel_item][0] += channel_loss_sum
                    channel_loss_dict[channel_item][1] += sample_valid_token_count
            
            batch_loss_mean = torch.stack(batch_loss_mean).mean()
            batch_loss_for_backward = batch_loss_mean / grad_accum_steps #TODO: we currently consider "every sample's" weight the same, instead of "every token's"
            batch_loss_for_backward.backward()
            
            # cur_loss = self.train_step(inputs, labels) / grad_accum_steps #TODO: we currently consider "every sample's" weight the same, instead of "every token's"
            # cur_loss.backward()
            # cur_valid_token_count = labels.ne(-1).sum()

            losses.append(batch_loss_for_backward.detach()) # for general loss display, no gradient
        loss = torch.sum(torch.stack(losses)).to(self.device)

        dist_utils.clip_grad_norm_(
            [p for m in model_parts for p in m.parameters()],
            self.job_config.training.max_norm,
            foreach=True,
            pp_mesh=self.world_mesh["pp"] if parallel_dims.pp_enabled else None,
        )
        self.checkpointer.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()

        # log metrics
        if not self.metrics_processor.should_log(self.step):
            return

        extra_losses = {}
        if (
            parallel_dims.dp_replicate_enabled
            or parallel_dims.dp_shard_enabled
            or parallel_dims.cp_enabled
            or self.ft_manager.enabled
        ):
            # loss = loss.detach()
            ft_pg = self.ft_manager.replicate_pg if self.ft_manager.enabled else None
            global_avg_loss, global_max_loss = (
                dist_utils.dist_mean(loss, world_mesh["dp_cp"], ft_pg),
                dist_utils.dist_max(loss, world_mesh["dp_cp"], ft_pg),
            )
            assert ft_pg is None, "fault tolerance not supported for global channel loss"
            global_channel_loss = dist_utils.dist_dict(channel_loss_dict, world_mesh["dp_cp"])
            merged_channel_loss = {}
            for rnk in global_channel_loss:
                for channel, (loss_sum, valid_token_count) in rnk.items():
                    if channel not in merged_channel_loss:
                        merged_channel_loss[channel] = [loss_sum, valid_token_count]
                    else:
                        merged_channel_loss[channel][0] += loss_sum
                        merged_channel_loss[channel][1] += valid_token_count

        else:
            global_avg_loss = global_max_loss = loss.item()
            merged_channel_loss = channel_loss_dict
        
        # some range may not be logged, so we use channels to calculate total tokens
        global_total_tokens = sum([merged_channel_loss[x][1] for x in merged_channel_loss if x >= 0]) 
        extra_losses["misc/global_valid_tokens"] = global_total_tokens
        for channel, (loss_sum, valid_token_count) in merged_channel_loss.items():
            if channel >= 0:
                avg_channel_loss = loss_sum / valid_token_count if valid_token_count > 0 else 0.0
                extra_losses[f'loss_metrics/channel_loss_{channel}'] = avg_channel_loss
                extra_losses[f'misc/channel_token_perc_{channel}'] = valid_token_count / global_total_tokens
            else:
                n_q = -1 - channel
                avg_channel_loss = loss_sum / valid_token_count if valid_token_count > 0 else 0.0
                extra_losses[f'loss_metrics/target_loss_{n_q}'] = avg_channel_loss
                extra_losses[f'misc/target_token_perc_{n_q}'] = valid_token_count / global_total_tokens

        self.metrics_processor.log(self.step, global_avg_loss, global_max_loss, extra_metrics=extra_losses)



    def train_step(self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor):
        # self.optimizers.zero_grad()

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        model_parts = self.model_parts
        world_mesh = self.world_mesh
        parallel_dims = self.parallel_dims

        # apply context parallelism if cp is enabled
        # ensure CP handles the separate freqs_cis buffer for each pp stage
        inputs = input_dict["input"]
        optional_context_parallel_ctx = (
            dist_utils.create_context_parallel_ctx(
                cp_mesh=world_mesh["cp"],
                cp_buffers=[inputs, labels] + [m.freqs_cis for m in model_parts],
                cp_seq_dims=[1, 1] + [0 for _ in model_parts],
                cp_no_restore_buffers={inputs, labels},
                cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
            )
            if parallel_dims.cp_enabled
            else None
        )

        if parallel_dims.pp_enabled:
            # Pipeline Parallel forward / backward inside step() call
            with self.train_context(optional_context_parallel_ctx):
                targets, losses = (
                    (labels, []) if self.pp_has_last_stage else (None, None)
                )
                if self.pp_has_first_stage:
                    self.pp_schedule.step(
                        inputs, target=targets, losses=losses, input_batch=inputs
                    )
                else:
                    self.pp_schedule.step(
                        target=targets, losses=losses, input_batch=inputs
                    )

            # accumulate losses across pipeline microbatches
            # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
            loss = (
                torch.mean(torch.stack(losses)).to(self.device)
                if self.pp_has_last_stage
                else torch.tensor([-1.0], device=self.device)
            )
        else:
            # Non-PP forward / backward
            with self.train_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                pred = model_parts[0](inputs)
                loss = self.loss_fn(pred, labels)
                # need to free to before bwd to avoid peaking memory
                del pred
                # loss.backward()
        return loss
        # dist_utils.clip_grad_norm_(
        #     [p for m in model_parts for p in m.parameters()],
        #     self.job_config.training.max_norm,
        #     foreach=True,
        #     pp_mesh=self.world_mesh["pp"] if parallel_dims.pp_enabled else None,
        # )
        # self.checkpointer.maybe_wait_for_staging()
        # self.optimizers.step()
        # self.lr_schedulers.step()

        # log metrics
        # if not self.metrics_processor.should_log(self.step):
        #     return

        # if (
        #     parallel_dims.dp_replicate_enabled
        #     or parallel_dims.dp_shard_enabled
        #     or parallel_dims.cp_enabled
        #     or self.ft_manager.enabled
        # ):
        #     loss = loss.detach()
        #     ft_pg = self.ft_manager.replicate_pg if self.ft_manager.enabled else None
        #     global_avg_loss, global_max_loss = (
        #         dist_utils.dist_mean(loss, world_mesh["dp_cp"], ft_pg),
        #         dist_utils.dist_max(loss, world_mesh["dp_cp"], ft_pg),
        #     )
        # else:
        #     global_avg_loss = global_max_loss = loss.detach().item()

        # self.metrics_processor.log(self.step, global_avg_loss, global_max_loss)

    @record
    def train(self):
        job_config = self.job_config

        self.checkpointer.load(step=job_config.checkpoint.load_step)
        logger.info(f"Training starts at step {self.step + 1}.")

        with maybe_enable_profiling(
            job_config, global_step=self.step
        ) as torch_profiler, maybe_enable_memory_snapshot(
            job_config, global_step=self.step
        ) as memory_profiler, ft.maybe_semi_sync_training(
            job_config,
            ft_manager=self.ft_manager,
            model=self.model_parts[0],
            optimizer=self.optimizers,
            sync_every=job_config.fault_tolerance.sync_steps,
        ):
            data_iterator = iter(self.dataloader)
            while self.step < job_config.training.steps:
                self.step += 1
                self.gc_handler.run(self.step)
                # inputs, labels = self.next_batch(data_iterator)
                # self.train_step(inputs, labels)
                self.gradient_accumulate_train_step(
                    data_iterator, job_config.training.gradient_accumulation_steps
                )
                self.checkpointer.save(
                    self.step, force=(self.step == job_config.training.steps)
                )

                # signal the profiler that the next profiling step has started
                if torch_profiler:
                    torch_profiler.step()
                if memory_profiler:
                    memory_profiler.step()

                # reduce timeout after first train step for faster signal
                # (assuming lazy init and compilation are finished)
                if self.step == 1:
                    dist_utils.set_pg_timeouts(
                        timeout=timedelta(
                            seconds=job_config.comm.train_timeout_seconds
                        ),
                        world_mesh=self.world_mesh,
                    )

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        self.metrics_processor.close()
        logger.info("Training completed")

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step}

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.step = state_dict["step"]

    def close(self) -> None:
        if self.checkpointer:
            self.checkpointer.close()


if __name__ == "__main__":
    init_logger()
    config_manager = ConfigManager()
    config = config_manager.parse_args()
    trainer: Optional[Trainer] = None

    try:
        trainer = Trainer(config)

        if config.checkpoint.create_seed_checkpoint:
            assert (
                int(os.environ["WORLD_SIZE"]) == 1
            ), "Must create seed checkpoint using a single device, to disable sharding."
            assert (
                config.checkpoint.enable_checkpoint
            ), "Must enable checkpointing when creating a seed checkpoint."
            trainer.checkpointer.save(curr_step=0, force=True)
            logger.info("Created seed checkpoint")
        else:
            trainer.train()
    finally:
        if trainer:
            trainer.close()

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            logger.info("Process group destroyed.")
