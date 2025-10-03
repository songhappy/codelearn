# ep.py
# Run with: torchrun --nproc_per_node 4 ep.py
import os
import math
import torch
import torch.nn as nn
import torch.distributed as dist


# ---------------- Config ----------------
HIDDEN = 64             # token hidden size
FFN = 128               # expert MLP width
TOKENS_PER_RANK = 16    # local tokens each rank starts with
CAPACITY_FACTOR = 1.25  # >1.0 to allow some headroom per expert
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- Modules ----------------
class ExpertMLP(nn.Module):
    """One expert per rank."""
    def __init__(self, hidden=HIDDEN, ffn=FFN):
        super().__init__()
        self.fc1 = nn.Linear(hidden, ffn)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(ffn, hidden)

    def forward(self, x):  # [N, H]
        return self.fc2(self.act(self.fc1(x)))


class Top1Router(nn.Module):
    """Replicated router on every rank: projects tokens -> expert logits; pick argmax."""
    def __init__(self, hidden=HIDDEN, num_experts=1):
        super().__init__()
        self.proj = nn.Linear(hidden, num_experts, bias=False)

    def forward(self, x):  # [T, H]
        logits = self.proj(x)                 # [T, E]
        expert_id = torch.argmax(logits, -1)  # [T]
        return expert_id


# ---------------- Helpers ----------------
def capacity(tokens_total, num_experts, cap_factor):
    """Fixed per-expert capacity (for *each receiver*) used by all_to_all."""
    avg = math.ceil(tokens_total / num_experts)
    return math.ceil(avg * cap_factor)


def pack_tokens_to_experts(x, expert_id, E, cap):
    """
    x: [T, H], expert_id: [T], E experts
    Returns:
      send_buf: [E, cap, H]  (per-destination expert, padded)
      idx_per_exp: list[E] of original token indices assigned (<= cap per expert)
      mask: [E, cap]  True where slot is real (not pad)
    """
    T, H = x.shape
    device = x.device
    idx_per_exp = [[] for _ in range(E)]
    for i in range(T):
        e = int(expert_id[i])
        if len(idx_per_exp[e]) < cap:
            idx_per_exp[e].append(i)
        # else: token dropped due to capacity (kept zero)

    send_buf = torch.zeros(E, cap, H, device=device, dtype=x.dtype)
    mask = torch.zeros(E, cap, dtype=torch.bool, device=device)
    for e in range(E):
        idxs = idx_per_exp[e]
        if idxs:
            n = len(idxs)
            send_buf[e, :n] = x[idxs]
            mask[e, :n] = True
    return send_buf, idx_per_exp, mask


def unpack_tokens_from_experts(out_buf, idx_per_exp, mask, T):
    """
    out_buf: [E, cap, H]  (per-expert, padded)
    idx_per_exp: as returned by pack()
    mask: [E, cap]
    T: number of original local tokens
    Returns y: [T, H] in original token order (zeros where token was dropped)
    """
    device = out_buf.device
    H = out_buf.shape[-1]
    y = torch.zeros(T, H, device=device, dtype=out_buf.dtype)
    for e in range(out_buf.shape[0]):
        valid = mask[e]
        if valid.any():
            slots = valid.nonzero(as_tuple=False).squeeze(-1)
            toks = idx_per_exp[e]
            # slots and toks have same length/order (filled in pack())
            y[toks] = out_buf[e, slots]
    return y


def main():
    # ---------- Distributed init ----------
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)

    NUM_EXPERTS = world_size  # EP only: one expert per rank

    # ---------- Build modules ----------
    expert = ExpertMLP().to(device)                      # 1 expert on *this* rank
    router = Top1Router(num_experts=NUM_EXPERTS).to(device)  # replicated

    # (Optional) deterministic-ish routing so experts differ:
    with torch.no_grad():
        w = torch.zeros_like(router.proj.weight)
        for e in range(NUM_EXPERTS):
            w[e, e % HIDDEN] = 1.0
        router.proj.weight.copy_(w)

    # ---------- Fake local tokens ----------
    torch.manual_seed(1234 + rank)
    x_local = torch.randn(TOKENS_PER_RANK, HIDDEN, device=device)

    # ---------- Route locally (top-1) ----------
    expert_id = router(x_local)  # [T]
    total_tokens_global = TOKENS_PER_RANK * world_size
    cap = capacity(total_tokens_global, NUM_EXPERTS, CAPACITY_FACTOR)

    # Pack per-destination expert (== receiver rank)
    send_buf, idx_per_exp_local, mask_send = pack_tokens_to_experts(
        x_local, expert_id, NUM_EXPERTS, cap
    )  # send_buf: [E, cap, H]

    # ---------- all_to_all: dispatch tokens to experts ----------
    # Layout for a2a: split dim0 into equal chunks for each peer.
    send_flat = send_buf.reshape(NUM_EXPERTS * cap, HIDDEN).contiguous()
    recv_flat = torch.empty_like(send_flat)
    dist.all_to_all_single(recv_flat, send_flat)

    # Receiver’s view: chunks come from each SENDER (world_size chunks), each of size [cap, H].
    # So reshape to [sender, cap, H].
    recv_by_sender = recv_flat.view(world_size, cap, HIDDEN)

    # Exchange masks with the *same layout* (per-dest cap, split by sender)
    mask_send_flat = mask_send.view(NUM_EXPERTS * cap).to(torch.int)
    mask_recv_flat = torch.empty_like(mask_send_flat)
    dist.all_to_all_single(mask_recv_flat, mask_send_flat)
    mask_by_sender = mask_recv_flat.view(world_size, cap).to(torch.bool)  # [sender, cap]

    # Gather inputs for *our* local expert (expert id == my rank) from all senders
    expert_chunks = []
    slice_lens = []
    for s in range(world_size):
        valid = mask_by_sender[s]          # slots from sender s destined to *this* expert
        n = int(valid.sum().item())
        if n > 0:
            expert_chunks.append(recv_by_sender[s][valid])  # [n, H]
            slice_lens.append(n)
        else:
            slice_lens.append(0)
    if expert_chunks:
        expert_in = torch.cat(expert_chunks, dim=0)
    else:
        expert_in = torch.zeros(0, HIDDEN, device=device, dtype=x_local.dtype)

    # ---------- Run local expert ----------
    with torch.no_grad():
        expert_out = expert(expert_in)  # [sum_s n_s, H]

    # ---------- Send outputs back to original senders (reverse a2a) ----------
    # We must place outputs back into the *same slots* they arrived in for each sender.
    send_back_by_sender = torch.zeros_like(recv_by_sender)  # [sender, cap, H]
    offset = 0
    for s in range(world_size):
        n = slice_lens[s]
        if n > 0:
            valid = mask_by_sender[s]
            send_back_by_sender[s][valid] = expert_out[offset:offset + n]
            offset += n

    send_back_flat = send_back_by_sender.reshape(world_size * cap, HIDDEN).contiguous()
    recv_back_flat = torch.empty_like(send_back_flat)
    dist.all_to_all_single(recv_back_flat, send_back_flat)

    # Now each origin rank receives from *all experts*; reshape to [expert, cap, H]
    recv_back_buf = recv_back_flat.view(NUM_EXPERTS, cap, HIDDEN)

    # ---------- Restore original token order ----------
    y_local = unpack_tokens_from_experts(
        out_buf=recv_back_buf,
        idx_per_exp=idx_per_exp_local,
        mask=mask_send,
        T=TOKENS_PER_RANK
    )

    # ---------- Debug prints ----------
    processed_local = int(sum(slice_lens))
    print(f"[rank {rank}] expert processed {processed_local} tokens; y_local.shape={tuple(y_local.shape)}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
