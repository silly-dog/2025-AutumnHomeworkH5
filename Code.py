# 多模态情感分类（3类）：text(.txt)+image(.jpg) -> {negative, neutral, positive}
# 调优版：优先使用 CLIP（文本+图像预训练编码）提升准确率；若无法下载模型则自动回退到轻量模型。

from __future__ import annotations

import os
import re
import random
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ============== 基础配置 ==============
SEED = 42
def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", device)

# 让 CUDA 上更快一些（不影响结果正确性）
if device.type == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

ROOT = Path(".").resolve()
DATA_DIR = ROOT / "data"
TRAIN_CSV = ROOT / "train.txt"
TEST_CSV = ROOT / "test_without_label.txt"

assert DATA_DIR.exists(), f"找不到数据目录: {DATA_DIR}"
assert TRAIN_CSV.exists(), f"找不到: {TRAIN_CSV}"
assert TEST_CSV.exists(), f"找不到: {TEST_CSV}"

label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v: k for k, v in label2id.items()}

# ============== 读取数据 ==============
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
for df_name, df in [("train", train_df), ("test", test_df)]:
    assert "guid" in df.columns and "tag" in df.columns, f"{df_name} 缺少列 guid/tag"
    df["guid"] = df["guid"].astype(str)

train_df = train_df[train_df["tag"].isin(label2id)].copy()
train_df["label"] = train_df["tag"].map(label2id).astype(int)

def add_paths(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_path"] = df["guid"].apply(lambda g: str(DATA_DIR / f"{g}.txt"))
    df["img_path"]  = df["guid"].apply(lambda g: str(DATA_DIR / f"{g}.jpg"))
    return df

train_df = add_paths(train_df)
test_df = add_paths(test_df)

missing_train = train_df[~train_df["text_path"].map(os.path.exists) | ~train_df["img_path"].map(os.path.exists)]
missing_test  = test_df[~test_df["text_path"].map(os.path.exists)  | ~test_df["img_path"].map(os.path.exists)]
print("train size:", len(train_df), "missing:", len(missing_train))
print("test  size:", len(test_df),  "missing:", len(missing_test))
assert len(missing_train) == 0, "训练集中存在缺失的txt/jpg文件"
assert len(missing_test) == 0, "测试集中存在缺失的txt/jpg文件"

# ============== 划分验证集（可调整） ==============
VAL_RATIO = 0.1
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=VAL_RATIO,
    random_state=SEED,
    shuffle=True,
    stratify=train_df["label"],
)
train_split = train_df.iloc[train_idx].reset_index(drop=True)
val_split = train_df.iloc[val_idx].reset_index(drop=True)
print("train_split:", len(train_split), "val_split:", len(val_split))
print("val label dist:\n", val_split["tag"].value_counts(normalize=True))

# ============== 训练超参数（可调，偏向更高准确率） ==============
MAX_LEN = 96
IMG_SIZE = 224
BATCH_SIZE = 32 if device.type == "cuda" else 16
EPOCHS = 10 if device.type == "cuda" else 6
NUM_WORKERS = 0  # Windows/Notebook 用0最稳
USE_AMP = device.type == "cuda"

# 训练策略（可调）
GRAD_ACCUM_STEPS = 2 if device.type == "cuda" else 1
PATIENCE = 3  # early stopping
WARMUP_RATIO = 0.1

# ============== I/O 工具 ==============
def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

# ============== 尝试使用 CLIP（更强的多模态预训练） ==============
USE_CLIP = True
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
FREEZE_CLIP = True if device.type == "cpu" else False  # CPU下建议冻结以加速
HEAD_LR = 1e-3
FULL_LR = 1e-5
WEIGHT_DECAY = 1e-4

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        # 注意：不要把 labels 传进 model.forward
        logits = model(**{k: v.to(device, non_blocking=True) for k, v in batch.items() if k not in ("guid", "labels")})
        preds = torch.argmax(logits, dim=-1)
        all_preds.append(preds.cpu())
        all_labels.append(batch["labels"].cpu())
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    return acc, mf1, y_true, y_pred

def _compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    # 反比于频次的权重，避免模型偏向多数类；再做归一化让平均权重约为1
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    inv = inv * (num_classes / inv.sum())
    return torch.tensor(inv, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    scaler: torch.amp.GradScaler | None,
    loss_fn: nn.Module,
    grad_accum_steps: int = 1,
):
    model.train()
    running, n = 0.0, 0
    grad_accum_steps = max(int(grad_accum_steps), 1)
    for batch in loader:
        labels = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast(device_type="cuda", enabled=USE_AMP):
                logits = model(**{k: v.to(device, non_blocking=True) for k, v in batch.items() if k not in ("guid", "labels")})
                loss = loss_fn(logits, labels) / grad_accum_steps
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(**{k: v.to(device, non_blocking=True) for k, v in batch.items() if k not in ("guid", "labels")})
            loss = loss_fn(logits, labels) / grad_accum_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        bs = labels.size(0)
        running += (loss.item() * grad_accum_steps) * bs
        n += bs
    return running / max(n, 1)

@torch.no_grad()
def predict_labels(model: nn.Module, loader: DataLoader) -> list[int]:
    model.eval()
    out = []
    for batch in loader:
        logits = model(**{k: v.to(device, non_blocking=True) for k, v in batch.items() if k not in ("guid", "labels")})
        out.append(torch.argmax(logits, dim=-1).cpu())
    return torch.cat(out).tolist()

def run_clip_pipeline():
    from transformers import CLIPProcessor, CLIPModel
    # 关键：强制用 safetensors，避免 torch<2.6 时触发 torch.load 安全限制
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, use_fast=True)
    clip = CLIPModel.from_pretrained(
        CLIP_MODEL_NAME,
        use_safetensors=True,
        torch_dtype=(torch.float16 if device.type == "cuda" else None),
    )

    class MMDS_CLIP(Dataset):
        def __init__(self, df: pd.DataFrame, with_label: bool):
            self.df = df.reset_index(drop=True)
            self.with_label = with_label

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx: int):
            row = self.df.iloc[idx]
            guid = row["guid"]
            text = load_text(row["text_path"])
            img = load_image(row["img_path"])
            item = {"guid": guid, "text": text, "image": img}
            if self.with_label:
                item["labels"] = int(row["label"])
            return item

    def collate_fn(batch: list[dict]):
        texts = [b["text"] for b in batch]
        images = [b["image"] for b in batch]
        guids = [b["guid"] for b in batch]
        enc = processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        out = {
            "guid": guids,
            "input_ids": enc["input_ids"],
            "attention_mask": enc.get("attention_mask", None),
            "pixel_values": enc["pixel_values"],
        }
        if out["attention_mask"] is None:
            out.pop("attention_mask")
        if "labels" in batch[0]:
            out["labels"] = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
        return out

    train_ds = MMDS_CLIP(train_split, with_label=True)
    val_ds = MMDS_CLIP(val_split, with_label=True)
    test_ds = MMDS_CLIP(test_df, with_label=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)

    class CLIPClassifier(nn.Module):
        def __init__(self, clip: CLIPModel, num_classes: int = 3, dropout: float = 0.2):
            super().__init__()
            self.clip = clip
            dim = clip.config.projection_dim  # usually 512
            self.head = nn.Sequential(
                nn.Linear(dim * 2, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, num_classes),
            )

        def forward(self, input_ids: torch.Tensor, pixel_values: torch.Tensor, attention_mask: torch.Tensor | None = None):
            out = self.clip(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                return_dict=True,
            )
            # CLIP returns already projected embeddings
            t = out.text_embeds
            v = out.image_embeds
            fused = torch.cat([t, v], dim=-1)
            return self.head(fused)

    model = CLIPClassifier(clip, num_classes=len(label2id)).to(device)

    # 类别权重（缓解类别不平衡：neutral 很少）
    class_w = _compute_class_weights(train_split["label"].tolist(), num_classes=len(label2id)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)

    if FREEZE_CLIP:
        for p in model.clip.parameters():
            p.requires_grad = False
        params = list(model.head.parameters())
        lr = HEAD_LR
        print("CLIP frozen: train head only")
    else:
        params = model.parameters()
        lr = FULL_LR
        print("CLIP finetune enabled")

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)

    # 学习率调度：余弦 + warmup（更稳）
    try:
        from transformers import get_cosine_schedule_with_warmup

        num_update_steps = max(1, (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS)
        num_warmup_steps = int(num_update_steps * WARMUP_RATIO)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_update_steps)
    except Exception:
        scheduler = None

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_state, best_mf1 = None, -1.0
    bad_epochs = 0
    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler if USE_AMP else None,
            loss_fn=loss_fn,
            grad_accum_steps=GRAD_ACCUM_STEPS,
        )
        if scheduler is not None:
            scheduler.step()
        acc, mf1, y_true, y_pred = evaluate(model, val_loader)
        print(f"Epoch {epoch}/{EPOCHS} | loss={tr_loss:.4f} | val_acc={acc:.4f} | val_mf1={mf1:.4f}")
        if mf1 > best_mf1:
            best_mf1 = mf1
            best_state = deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"Early stopping triggered (patience={PATIENCE}).")
                break

    model.load_state_dict(best_state)
    acc, mf1, y_true, y_pred = evaluate(model, val_loader)
    print("best val macro-F1:", mf1, "val_acc:", acc)
    print(classification_report(y_true, y_pred, target_names=["negative", "neutral", "positive"]))

    test_pred_ids = predict_labels(model, test_loader)
    submission = pd.DataFrame({
        "guid": test_df["guid"].astype(str).values,
        "tag": [id2label[i] for i in test_pred_ids],
    })
    out_path = ROOT / "submission.csv"
    submission.to_csv(out_path, index=False, encoding="utf-8")
    print("saved:", out_path)
    print(submission.head())

def run_fallback_light_model():
    # 轻量回退模型（无torchvision、无外部下载），比之前略微调优：更强图像增强+更长训练
    _token_pat = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]", re.UNICODE)
    def basic_tokenize(s: str) -> list[str]:
        s = (s or "").lower()
        return _token_pat.findall(s)

    def build_vocab(df: pd.DataFrame, max_vocab: int = 30000, min_freq: int = 2) -> dict[str, int]:
        freq: dict[str, int] = {}
        for p in df["text_path"].tolist():
            toks = basic_tokenize(load_text(p))
            for t in toks:
                freq[t] = freq.get(t, 0) + 1
        items = [(t, c) for t, c in freq.items() if c >= min_freq]
        items.sort(key=lambda x: x[1], reverse=True)
        items = items[:max_vocab]
        vocab = {"[PAD]": 0, "[UNK]": 1}
        for t, _ in items:
            if t not in vocab:
                vocab[t] = len(vocab)
        return vocab

    vocab = build_vocab(train_split, max_vocab=30000, min_freq=2)
    pad_id = vocab["[PAD]"]
    unk_id = vocab["[UNK]"]
    print("[fallback] vocab size:", len(vocab))

    def encode_text(text: str, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        toks = basic_tokenize(text)
        ids = [vocab.get(t, unk_id) for t in toks][:max_len]
        attn = [1] * len(ids)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long)

    img_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    img_std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def pil_to_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    def image_transform(img: Image.Image, train: bool) -> torch.Tensor:
        img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        if train:
            if random.random() < 0.5:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                img = ImageEnhance.Brightness(img).enhance(1.0 + random.uniform(-0.15, 0.15))
            if random.random() < 0.5:
                img = ImageEnhance.Contrast(img).enhance(1.0 + random.uniform(-0.15, 0.15))
            if random.random() < 0.35:
                img = ImageEnhance.Color(img).enhance(1.0 + random.uniform(-0.15, 0.15))
        x = pil_to_tensor(img)
        x = (x - img_mean) / img_std
        return x

    class MMDS(Dataset):
        def __init__(self, df: pd.DataFrame, train_img: bool, with_label: bool):
            self.df = df.reset_index(drop=True)
            self.train_img = train_img
            self.with_label = with_label

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx: int):
            row = self.df.iloc[idx]
            guid = row["guid"]
            text = load_text(row["text_path"])
            input_ids, attention_mask = encode_text(text, MAX_LEN)
            img = load_image(row["img_path"])
            img_t = image_transform(img, train=self.train_img)
            item = {"guid": guid, "input_ids": input_ids, "attention_mask": attention_mask, "image": img_t}
            if self.with_label:
                item["labels"] = torch.tensor(int(row["label"]), dtype=torch.long)
            return item

    def collate(batch: list[dict]):
        guids = [b["guid"] for b in batch]
        input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_id)
        attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
        images = torch.stack([b["image"] for b in batch], dim=0)
        out = {"guid": guids, "input_ids": input_ids, "attention_mask": attention_mask, "image": images}
        if "labels" in batch[0]:
            out["labels"] = torch.stack([b["labels"] for b in batch], dim=0)
        return out

    train_ds = MMDS(train_split, train_img=True, with_label=True)
    val_ds   = MMDS(val_split,   train_img=False, with_label=True)
    test_ds  = MMDS(test_df,     train_img=False, with_label=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, collate_fn=collate, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate, pin_memory=True)

    class TextEncoder(nn.Module):
        def __init__(self, vocab_size: int, emb_dim: int = 192, hidden: int = 192, dropout: float = 0.2):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
            self.gru = nn.GRU(emb_dim, hidden, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(dropout)
            self.out_dim = hidden * 2

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            x = self.emb(input_ids)
            x, _ = self.gru(x)
            x = self.drop(x)
            mask = attention_mask.unsqueeze(-1).float()
            x = x * mask
            denom = mask.sum(dim=1).clamp_min(1.0)
            return x.sum(dim=1) / denom

    class ImageEncoder(nn.Module):
        def __init__(self, out_dim: int = 256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.proj = nn.Linear(256, out_dim)

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            x = self.net(image).flatten(1)
            return self.proj(x)

    class Fusion(nn.Module):
        def __init__(self, vocab_size: int, num_classes: int = 3, fusion_dim: int = 256, dropout: float = 0.25):
            super().__init__()
            self.text = TextEncoder(vocab_size=vocab_size)
            self.img = ImageEncoder(out_dim=fusion_dim)
            self.text_proj = nn.Sequential(nn.Linear(self.text.out_dim, fusion_dim), nn.ReLU(), nn.Dropout(dropout))
            self.cls = nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, num_classes),
            )

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, image: torch.Tensor):
            t = self.text_proj(self.text(input_ids, attention_mask))
            v = self.img(image)
            return self.cls(torch.cat([t, v], dim=-1))

    model = Fusion(vocab_size=len(vocab), num_classes=len(label2id)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=WEIGHT_DECAY)
    class_w = _compute_class_weights(train_split["label"].tolist(), num_classes=len(label2id)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_state, best_mf1 = None, -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, n = 0.0, 0
        for batch in train_loader:
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            if USE_AMP:
                with torch.amp.autocast(device_type="cuda", enabled=USE_AMP):
                    logits = model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        image=batch["image"].to(device),
                    )
                    loss = loss_fn(logits, labels)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    image=batch["image"].to(device),
                )
                loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            bs = labels.size(0)
            running += loss.item() * bs
            n += bs
        tr_loss = running / max(n, 1)

        # eval
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    image=batch["image"].to(device),
                )
                preds = torch.argmax(logits, dim=-1).cpu()
                all_preds.append(preds)
                all_labels.append(batch["labels"])
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()
        acc = accuracy_score(y_true, y_pred)
        mf1 = f1_score(y_true, y_pred, average="macro")
        print(f"[fallback] Epoch {epoch}/{EPOCHS} | loss={tr_loss:.4f} | val_acc={acc:.4f} | val_mf1={mf1:.4f}")
        if mf1 > best_mf1:
            best_mf1 = mf1
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    print("[fallback] best val macro-F1:", best_mf1)
    print(classification_report(y_true, y_pred, target_names=["negative", "neutral", "positive"]))

    # predict
    model.eval()
    preds_all = []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                image=batch["image"].to(device),
            )
            preds_all.append(torch.argmax(logits, dim=-1).cpu())
    test_pred_ids = torch.cat(preds_all).tolist()
    submission = pd.DataFrame({
        "guid": test_df["guid"].astype(str).values,
        "tag": [id2label[i] for i in test_pred_ids],
    })
    out_path = ROOT / "submission.csv"
    submission.to_csv(out_path, index=False, encoding="utf-8")
    print("saved:", out_path)
    print(submission.head())

# ============== 运行（优先CLIP；失败则回退） ==============
if USE_CLIP:
    try:
        run_clip_pipeline()
    except Exception as e:
        print("[WARN] CLIP pipeline failed, fallback to light model. Error:", repr(e))
        run_fallback_light_model()
else:
    run_fallback_light_model()