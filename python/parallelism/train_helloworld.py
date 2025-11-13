# train_helloworld.py
from accelerate import Accelerator
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def main():
    accelerator = Accelerator()  # handles CPU/GPU/TPU + DDP setup
    device = accelerator.device

    # Hello from each rank
    accelerator.print(
        f"Hello from rank {accelerator.process_index}/{accelerator.num_processes} on device {device}"
    )

    # --- toy data ---
    N, in_features, out_features = 1024, 10, 1
    x = torch.randn(N, in_features)
    y = torch.randn(N, out_features)
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    # --- toy model/optim ---
    model = nn.Linear(in_features, out_features)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    # Prepare objects for distributed training (wraps DDP, samplers, etc.)
    model, opt, dl = accelerator.prepare(model, opt, dl)

    # --- tiny training loop ---
    for epoch in range(3):
        for xb, yb in dl:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
        accelerator.print(f"[rank {accelerator.process_index}] epoch {epoch} loss={loss.item():.4f}")

    # Save a small checkpoint once (main process only)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        to_save = accelerator.unwrap_model(model).state_dict()
        torch.save(to_save, "checkpoint.pt")
        print("Saved checkpoint.pt")

if __name__ == "__main__":
    main()
