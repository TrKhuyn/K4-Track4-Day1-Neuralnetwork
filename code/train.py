"""Pipeline huấn luyện, đánh giá và xuất dự đoán.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import time
import copy, random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # TODO: chọn bằng val, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    cm=np.asarray(cm,dtype=float); tp=np.diag(cm)
    p=np.divide(tp,cm.sum(0),out=np.zeros_like(tp),where=cm.sum(0)>0)
    r=np.divide(tp,cm.sum(1),out=np.zeros_like(tp),where=cm.sum(1)>0)
    return float(np.divide(2*p*r,p+r,out=np.zeros_like(tp),where=p+r>0).mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    model.eval()
    return torch.cat([model(X[i:i+batch_size]).argmax(1) for i in range(0,len(X),batch_size)])


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    Dùng hàm này cho: train loss (trên toàn bộ hoặc một tập con CỐ ĐỊNH của train), val, và eval cuối cùng.
    """
    model.eval(); total=0.; preds=[]
    for i in range(0,len(X),batch_size):
        xb,yb=X[i:i+batch_size],y[i:i+batch_size]; z=model(xb)
        if loss_name=="ce": loss=F.cross_entropy(z,yb,reduction="sum")
        elif loss_name=="mse": loss=F.mse_loss(z,F.one_hot(yb,7).to(z.dtype),reduction="sum")/7
        else: raise ValueError(loss_name)
        total+=loss.item(); preds.append(z.argmax(1))
    pred=torch.cat(preds); cm=torch.bincount(y.long()*7+pred.long(),minlength=49).reshape(7,7)
    return {"loss":total/len(y),"acc":float((pred==y).float().mean()),"macro_f1":macro_f1_from_confusion(cm.cpu().numpy())}


def compute_loss(logits, y, loss_name: str):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (ghi rõ bạn lấy trung bình thế nào).
    """
    if loss_name=="ce": return F.cross_entropy(logits,y)
    if loss_name=="mse": return F.mse_loss(logits,F.one_hot(y,7).to(logits.dtype))
    raise ValueError(loss_name)


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}
    (tên khoá của summary trùng tên cột trong experiments.xlsx)

    Các bước:
      0. set_seed(cfg["seed"]); tạo model = MLP(...), assert count_params(model) == EXPECTED_PARAMS[hidden]
         chuyển model lên device; tạo optimizer = build_optimizer(...)
         nếu precision == "fp16": scaler = torch.amp.GradScaler(...)
      1. step0_loss = evaluate(model, X_val, y_val)["loss"]   # TRƯỚC bước cập nhật đầu tiên; kỳ vọng ≈ ln 7
      2. for epoch in 1..epochs:
           model.train()
           for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator):
               with torch.autocast(...)  nếu precision != "fp32":   # chỉ bọc forward + loss
                   logits = model(xb); loss = compute_loss(logits, yb, cfg["loss"])
               optimizer.zero_grad(set_to_none=True)
               backward (qua scaler nếu fp16)
               nếu fp16 và có clip: scaler.unscale_(optimizer)  TRƯỚC khi clip
               gn = clip_gradients(model.parameters(), cfg["clip_norm"])   # chuẩn TRƯỚC khi cắt; ghi lại
               bước cập nhật (scaler.step(optimizer); scaler.update() nếu fp16, ngược lại optimizer.step())
               nếu loss là NaN/inf: đặt diverged=True và dừng sớm, ĐỪNG để notebook treo
           cuối epoch (dùng evaluate, chế độ eval):
               train_loss trên toàn bộ train (hoặc 1 tập con CỐ ĐỊNH ~50 000 mẫu), val_loss/val_acc/val_macro_f1
               grad_norm trung bình của epoch; thời gian epoch (torch.cuda.synchronize() nếu dùng GPU)
               nếu val_loss tốt nhất từ trước tới giờ: lưu best_state (bản sao state_dict) và best_epoch
      3. tổng hợp summary tại best_epoch (val_acc, val_macro_f1 lấy ở best_epoch); peak_mem_MB nếu có GPU
    TUYỆT ĐỐI không đưa X_eval vào hàm này để chọn epoch/cấu hình. Chỉ dùng val.
    """
    cfg={**DEFAULT_CFG,**cfg}
    if cfg["lr"] is None: raise ValueError("select lr on validation")
    set_seed(cfg["seed"]); device=data["X_tr"].device; hidden=tuple(cfg["hidden"])
    model=MLP(hidden,cfg["dropout"],cfg["init"]).to(device)
    if hidden in EXPECTED_PARAMS: assert count_params(model)==EXPECTED_PARAMS[hidden]
    opt=build_optimizer(cfg["optimizer"],model.parameters(),cfg["lr"],cfg["weight_decay"],cfg["momentum"])
    precision=cfg["precision"]
    if precision=="fp16" and device.type!="cuda": raise ValueError("fp16 requires CUDA")
    dtype=torch.float16 if precision=="fp16" else torch.bfloat16
    scaler=torch.amp.GradScaler("cuda",enabled=precision=="fp16")
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats()
    step0=evaluate(model,data["X_val"],data["y_val"],cfg["loss"])["loss"]
    hist={k:[] for k in ("epoch","train_loss","val_loss","val_acc","val_macro_f1","grad_norm","epoch_time_s")}
    best,best_ep,state,diverged=float("inf"),0,None,False; gen=torch.Generator().manual_seed(cfg["seed"])
    for ep in range(1,cfg["epochs"]+1):
        if device.type=="cuda": torch.cuda.synchronize()
        start=time.perf_counter(); model.train(); norms=[]
        for xb,yb in iterate_batches(data["X_tr"],data["y_tr"],cfg["batch"],gen):
            opt.zero_grad(set_to_none=True)
            ctx=torch.autocast(device_type=device.type,dtype=dtype) if precision!="fp32" else nullcontext()
            with ctx: loss=compute_loss(model(xb),yb,cfg["loss"])
            if not torch.isfinite(loss): diverged=True; break
            scaler.scale(loss).backward()
            if scaler.is_enabled(): scaler.unscale_(opt)
            norms.append(clip_gradients(model.parameters(),cfg["clip_norm"]))
            scaler.step(opt); scaler.update()
        if device.type=="cuda": torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        if diverged: break
        tr=evaluate(model,data["X_tr"],data["y_tr"],cfg["loss"]); va=evaluate(model,data["X_val"],data["y_val"],cfg["loss"])
        vals=(ep,tr["loss"],va["loss"],va["acc"],va["macro_f1"],float(np.mean(norms)),elapsed)
        for k,v in zip(hist,vals): hist[k].append(v)
        if va["loss"]<best:
            best,best_ep=va["loss"],ep; state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
        print(f"[{cfg['exp_id']}] {ep:02d}/{cfg['epochs']} train={tr['loss']:.4f} val={va['loss']:.4f} f1={va['macro_f1']:.4f}")
    if state is None: state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
    i=best_ep-1 if best_ep else -1; at=lambda k:hist[k][i] if hist[k] else float("nan")
    summary={"step0_loss":step0,"best_val_loss":best,"best_epoch":best_ep,
      "final_train_loss":hist["train_loss"][-1] if hist["train_loss"] else float("nan"),
      "final_val_loss":hist["val_loss"][-1] if hist["val_loss"] else float("nan"),
      "val_acc":at("val_acc"),"val_macro_f1":at("val_macro_f1"),
      "time_per_epoch_s":float(np.mean(hist["epoch_time_s"])) if hist["epoch_time_s"] else float("nan"),
      "peak_mem_MB":float(torch.cuda.max_memory_allocated()/2**20) if device.type=="cuda" else 0.,"diverged":diverged}
    return {"cfg":cfg,"history":hist,"summary":summary,"best_state":state}


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    import pandas as pd
    ids,p=np.asarray(row_id),np.asarray(preds)
    if ids.ndim!=1 or p.ndim!=1 or len(ids)!=len(p) or len(np.unique(ids))!=len(ids): raise ValueError("invalid predictions")
    if np.any((p<0)|(p>6)): raise ValueError("pred must be 0..6")
    out=Path(path); out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({"row_id":ids.astype("int64"),"pred":p.astype("int64")}).to_csv(out,index=False)


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    """
    model=MLP(tuple(cfg["hidden"]),cfg["dropout"],cfg["init"]); model.load_state_dict(result["best_state"])
    model.to(data["X_eval"].device)
    write_predictions(data["eval_row_id"],predict(model,data["X_eval"]).cpu().numpy(),pred_path)
