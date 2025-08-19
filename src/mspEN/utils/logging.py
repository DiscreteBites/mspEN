import torch
from tqdm import tqdm

from mspEN.modules.types import LossDict

def make_status_bar() -> tqdm:
    status_bar = tqdm(
        total=0,
        position=1,
        bar_format="{desc}",
        dynamic_ncols=True,
        leave=False,
    )
    return status_bar

def fmt_train_status(total_loss: torch.Tensor, scaled_losses: LossDict, batch_size: int, keys: list[str], max_len=120):
    parts = [f"Total={total_loss.item()/batch_size:.4f}"]
    parts += [f"{k}={scaled_losses[k].item()/batch_size:.4f}" for k in keys]
    s = " | ".join(parts)
    return s if len(s) <= max_len else s[:max_len - 1] + "…"