"""Vẽ biểu đồ từng thí nghiệm và biểu đồ so sánh.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from pathlib import Path


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và nên có val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    Các bước: fig, axes = plt.subplots(1, 3, figsize=...); plot; set_title/xlabel/legend;
              fig.savefig(path, dpi=..., bbox_inches="tight"); plt.close(fig)
    Gợi ý: đánh dấu best_epoch bằng đường thẳng đứng.
    """
    Path(path).parent.mkdir(parents=True,exist_ok=True); h=result["history"]; c=result["cfg"]; s=result["summary"]
    fig,ax=plt.subplots(1,3,figsize=(16,4.5)); ep=h["epoch"]
    ax[0].plot(ep,h["train_loss"],label="train"); ax[0].plot(ep,h["val_loss"],label="val")
    ax[1].plot(ep,h["val_acc"],label="accuracy"); ax[1].plot(ep,h["val_macro_f1"],label="macro-F1")
    ax[2].plot(ep,h["grad_norm"],label="pre-clip")
    for a,t in zip(ax,("Loss","Validation","Gradient norm")):
        a.set(title=t,xlabel="epoch"); a.grid(alpha=.25); a.legend()
        if s["best_epoch"]: a.axvline(s["best_epoch"],ls="--",color="gray")
    fig.suptitle(f"{c['exp_id']} | {c['optimizer']} lr={c['lr']}"); fig.tight_layout()
    fig.savefig(path,dpi=160,bbox_inches="tight"); plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    Path(path).parent.mkdir(parents=True,exist_ok=True); fig,ax=plt.subplots(figsize=(8,5))
    for r in results: ax.plot(r["history"]["epoch"],r["history"][metric],label=r["cfg"]["exp_id"])
    ax.set(title=title or metric,xlabel="epoch",ylabel=metric); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(path,dpi=160,bbox_inches="tight"); plt.close(fig)
