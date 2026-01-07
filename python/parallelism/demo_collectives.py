# demo_collectives.py
import os, sys
import torch
import torch.distributed as dist
import argparse
from torchcomms import new_comm, ReduceOp


def get_device_and_backend(device_str: str, rank: int):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if device_str == "xpu":
        num_devices = torch.xpu.device_count()
        if num_devices == 0:
            if rank == 0:
                print("[ERROR] No XPU devices found.")
            sys.exit(1)
        device_id = local_rank % num_devices
        dev = torch.device(f"xpu:{device_id}")
        torch.xpu.set_device(dev)

        def sync():
            torch.xpu.synchronize()
        backend = "xccl"

    elif device_str == "cuda":
        num_devices = torch.cuda.device_count()
        if num_devices == 0:
            if rank == 0:
                print("[ERROR] No CUDA devices found.")
            sys.exit(1)
        device_id = local_rank % num_devices
        dev = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(dev)

        def sync():
            torch.cuda.synchronize()
        backend = "nccl"

    else:
        dev = torch.device("cpu")
        backend = "gloo"
        def sync():
            pass

    return dev, sync, backend


def ordered_print(rank, world, msg):
    # Print in rank order to keep output readable; fall back if no process group.
    if not dist.is_available() or not dist.is_initialized():
        print(f"[rank {rank}] {msg}", flush=True)
        return
    for r in range(world):
        dist.barrier()
        if rank == r:
            print(f"[rank {rank}] {msg}", flush=True)
    dist.barrier()


def demo_comms(args):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device, _device_sync, backend = get_device_and_backend(args.device, rank)
    comm = new_comm(backend, device, name="main_comm")
    world = world_size
    def run_collective(name, fn):
        try:
            fn()
            ordered_print(rank, world, f"[OK] {name}")
        except Exception as exc:
            ordered_print(rank, world, f"[FAIL] {name}: {exc}")

    try:
        # 1) broadcast: src -> everyone
        def _broadcast():
            x = torch.tensor([42], device=device) if rank == 0 else torch.tensor([-1], device=device)
            comm.broadcast(x, 0, False)
            ordered_print(rank, world, f"broadcast result x={x.tolist()}")

        run_collective("broadcast", _broadcast)

        # 2) all_reduce: everyone contributes, everyone gets reduced result
        def _allreduce():
            x = torch.tensor([rank], dtype=torch.int64, device=device)
            comm.all_reduce(x, ReduceOp.SUM, False)
            ordered_print(rank, world, f"all_reduce(SUM) on [rank] => x={x.tolist()} (should be [0+1+2+3]=[6])")

        run_collective("allreduce", _allreduce)

        # 3) all_gather: everyone contributes, everyone gets all tensors
        def _allgather():
            x = torch.tensor([rank], dtype=torch.int64, device=device)
            gathered = [torch.empty_like(x, device=device) for _ in range(world)]
            comm.all_gather(gathered, x, False)
            ordered_print(rank, world, f"all_gather of [rank] => {[t.item() for t in gathered]}")

        run_collective("allgather", _allgather)

        # 4) reduce_scatter: reduce then shard result (each rank gets one chunk)
        # Each rank makes 8 numbers, split into 4 chunks of size 2.
        def _reducescatter():
            inp = torch.arange(8, dtype=torch.int64, device=device) + rank * 100
            inp_chunks = list(inp.chunk(world))  # 4 chunks of length 2
            out = torch.empty(2, dtype=torch.int64, device=device)
            comm.reduce_scatter(out, inp_chunks, ReduceOp.SUM, False)
            ordered_print(rank, world, f"reduce_scatter(SUM) inp={inp.tolist()} => out(chunk for rank {rank})={out.tolist()}")

        run_collective("reducescatter", _reducescatter)

        # 5) reduce: everyone contributes, only dst gets reduced result
        def _reduce():
            x = torch.tensor([rank], dtype=torch.int64, device=device)
            comm.reduce(x, 0, ReduceOp.SUM, False)
            ordered_print(rank, world, f"reduce(SUM,dst=0) => x={x.tolist()} (dst=0 should be [6])")

        run_collective("reduce", _reduce)

        # 6) gather: many-to-one collect (only dst gets list)
        def _gather():
            x = torch.tensor([rank], dtype=torch.int64, device=device)
            gather_list = [torch.empty_like(x, device=device) for _ in range(world)] if rank == 0 else []
            comm.gather(gather_list, x, 0, False)
            if rank == 0:
                ordered_print(rank, world, f"gather(dst=0) => {[t.item() for t in gather_list]}")
            else:
                ordered_print(rank, world, f"gather(dst=0) => (sent {x.item()})")

        run_collective("gather", _gather)

        # 7) scatter: one-to-many distribute (only src provides list)
        def _scatter():
            out = torch.empty(1, dtype=torch.int64, device=device)
            scatter_list = []
            if rank == 0:
                scatter_list = [torch.tensor([10 + r], dtype=torch.int64, device=device) for r in range(world)]
            comm.scatter(out, scatter_list, 0, False)
            ordered_print(rank, world, f"scatter(src=0) => out={out.tolist()}")

        run_collective("scatter", _scatter)

        # 8) all-to-all: everyone sends a distinct piece to everyone
        # all_to_all_single: input is length 4, output is length 4.
        # Each rank r sends input[r->k] = (r*10 + k) to rank k.
        def _alltoall():
            inp = torch.tensor([rank * 10 + k for k in range(world)], dtype=torch.int64, device=device)
            out = torch.empty(world, dtype=torch.int64, device=device)
            comm.all_to_all_single(out, inp, False)
            ordered_print(rank, world, f"all_to_all_single inp={inp.tolist()} => out={out.tolist()}")

        run_collective("all-to-all", _alltoall)
    finally:
        comm.finalize()

def demo_c10d(args):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device, _device_sync, backend = get_device_and_backend(args.device, rank)
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    rank = dist.get_rank()
    world = dist.get_world_size()

    # 1) broadcast: src -> everyone
    x = torch.tensor([42], device=device) if rank == 0 else torch.tensor([-1], device=device)
    dist.broadcast(x, src=0)
    ordered_print(rank, world, f"broadcast result x={x.tolist()}")

    # 2) all_reduce: everyone contributes, everyone gets reduced result
    x = torch.tensor([rank], dtype=torch.int64, device=device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    ordered_print(rank, world, f"all_reduce(SUM) on [rank] => x={x.tolist()} (should be [0+1+2+3]=[6])")

    # 3) all_gather: everyone contributes, everyone gets all tensors
    x = torch.tensor([rank], dtype=torch.int64, device=device)
    gathered = [torch.empty_like(x, device=device) for _ in range(world)]
    dist.all_gather(gathered, x)
    ordered_print(rank, world, f"all_gather of [rank] => {[t.item() for t in gathered]}")

    # 4) reduce_scatter: reduce then shard result (each rank gets one chunk)
    # Each rank makes 8 numbers, split into 4 chunks of size 2.
    inp = torch.arange(8, dtype=torch.int64, device=device) + rank * 100
    inp_chunks = list(inp.chunk(world))  # 4 chunks of length 2
    out = torch.empty(2, dtype=torch.int64, device=device)
    dist.reduce_scatter(out, inp_chunks, op=dist.ReduceOp.SUM)
    # Concept: out on rank k is SUM over ranks of chunk k
    ordered_print(rank, world, f"reduce_scatter(SUM) inp={inp.tolist()} => out(chunk for rank {rank})={out.tolist()}")

    # 5) reduce: everyone contributes, only dst gets reduced result
    x = torch.tensor([rank], dtype=torch.int64, device=device)
    dist.reduce(x, dst=0, op=dist.ReduceOp.SUM)
    # On dst, x becomes sum; on others, x is typically unchanged (backend-dependent)
    ordered_print(rank, world, f"reduce(SUM,dst=0) => x={x.tolist()} (dst=0 should be [6])")

    # 6) gather: many-to-one collect (only dst gets list)
    x = torch.tensor([rank], dtype=torch.int64, device=device)
    gather_list = [torch.empty_like(x, device=device) for _ in range(world)] if rank == 0 else None
    dist.gather(x, gather_list=gather_list, dst=0)
    if rank == 0:
        ordered_print(rank, world, f"gather(dst=0) => {[t.item() for t in gather_list]}")
    else:
        ordered_print(rank, world, f"gather(dst=0) => (sent {x.item()})")

    # 7) scatter: one-to-many distribute (only src provides list)
    out = torch.empty(1, dtype=torch.int64, device=device)
    scatter_list = None
    if rank == 0:
        scatter_list = [torch.tensor([10 + r], dtype=torch.int64, device=device) for r in range(world)]
    dist.scatter(out, scatter_list=scatter_list, src=0)
    ordered_print(rank, world, f"scatter(src=0) => out={out.tolist()}")

    # 8) all-to-all: everyone sends a distinct piece to everyone
    # all_to_all_single: input is length 4, output is length 4.
    # Each rank r sends input[r->k] = (r*10 + k) to rank k.
    inp = torch.tensor([rank * 10 + k for k in range(world)], dtype=torch.int64, device=device)
    out = torch.empty(world, dtype=torch.int64, device=device)
    dist.all_to_all_single(out, inp)
    # Rank k receives [0*10+k, 1*10+k, 2*10+k, 3*10+k]
    ordered_print(rank, world, f"all_to_all_single inp={inp.tolist()} => out={out.tolist()}")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="demo collectives")
    default_collectives=["broadcast", "allreduce", "allgather", "reducescatter", "reduce", "gather", "scatter", "all-to-all"]
    parser.add_argument(
        "--device", default="xpu", type=str, choices=["xpu", "cuda", "cpu"]
    )
    parser.add_argument(
        "--mode", default="both", type=str, choices=["c10d", "comms", "both"]
    )
    args = parser.parse_args()
    mode = args.mode

    if mode == "c10d":
        demo_c10d(args)
    elif mode == "comms":
        demo_comms(args)
    else:
        demo_c10d(args)
        demo_comms(args)
