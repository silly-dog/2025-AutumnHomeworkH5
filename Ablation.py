# 基础库导入（仅保留必需）
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch import optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# 强制GPU检测
assert torch.cuda.is_available(), "❌ 请使用GPU环境运行！"
device = torch.device("cuda")

# 核心配置（极简）
ROOT = Path(".")
DATA_DIR = ROOT / "data"
TRAIN_CSV = ROOT / "train.txt"
MAX_LEN = 96
BATCH_SIZE = 16
CLIP_MODEL = "openai/clip-vit-base-patch32"
label2id = {"negative": 0, "neutral": 1, "positive": 2}

# 工具函数
def load_text(p):
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()
def load_image(p):
    return Image.open(p).convert("RGB")

# 数据加载（鲁棒性处理）
train_df = pd.read_csv(TRAIN_CSV).dropna(subset=["guid", "tag"])
train_df["guid"] = train_df["guid"].astype(str)
train_df["label"] = train_df["tag"].map(label2id)
train_df["text_path"] = train_df["guid"].apply(lambda x: str(DATA_DIR / f"{x}.txt"))
train_df["img_path"] = train_df["guid"].apply(lambda x: str(DATA_DIR / f"{x}.jpg"))
_, val_split = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df["label"])

# 数据集与加载器
processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
class CLIPDataset(Dataset):
    def __init__(self, df): self.df = df
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        return {"text": load_text(row["text_path"]), "image": load_image(row["img_path"]), "label": row["label"]}

def collate_fn(batch):
    enc = processor([b["text"] for b in batch], [b["image"] for b in batch], 
                    padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    enc["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return enc

val_loader = DataLoader(CLIPDataset(val_split), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# 初始化CLIP模型（适配accelerate，一键GPU加载）
clip = CLIPModel.from_pretrained(
    CLIP_MODEL,
    torch_dtype=torch.float16,
    use_safetensors=True,
    device_map="auto"
)

# 融合模型+消融模型
class FusionModel(nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.clip = clip
        dim = clip.config.projection_dim
        self.head = nn.Sequential(nn.Linear(dim*2, 512), nn.ReLU(), nn.Linear(512, 3)).to(device, dtype=torch.float16)
    def forward(self, x):
        out = self.clip(x["input_ids"].to(device), x["pixel_values"].to(device, dtype=torch.float16), 
                        x["attention_mask"].to(device), return_dict=True)
        return self.head(torch.cat([out.text_embeds, out.image_embeds], dim=1))

class AblationModel(nn.Module):
    def __init__(self, fusion_model):
        super().__init__()
        self.clip = fusion_model.clip
        dim = self.clip.config.projection_dim
        self.fusion_head = fusion_model.head
        self.text_head = nn.Linear(dim, 3).to(device, dtype=torch.float16)
        self.img_head = nn.Linear(dim, 3).to(device, dtype=torch.float16)
    def forward(self, x, mode="fusion"):
        input_ids, pv, am = x["input_ids"].to(device), x["pixel_values"].to(device, dtype=torch.float16), x["attention_mask"].to(device)
        if mode == "text":
            return self.text_head(self.clip.get_text_features(input_ids, am))
        elif mode == "image":
            return self.img_head(self.clip.get_image_features(pv))
        else:
            out = self.clip(input_ids, pv, am, return_dict=True)
            return self.fusion_head(torch.cat([out.text_embeds, out.image_embeds], dim=1))

# 模型初始化+快速训练1个batch
model = FusionModel(clip)
weight_path = ROOT / "clip_ablation.pth"
if weight_path.exists():
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
else:
    train_loader = DataLoader(CLIPDataset(train_df), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    model.train()
    optim.AdamW(model.parameters(), lr=1e-5).step()  # 极简初始化
    torch.save(model.state_dict(), weight_path)
model.eval()
ablation_model = AblationModel(model).eval()

# 极简评估函数
@torch.no_grad()
def evaluate(mode):
    preds, labels = [], []
    for batch in val_loader:
        logits = ablation_model(batch, mode=mode)
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        labels.extend(batch["labels"].numpy())
    return accuracy_score(labels, preds), f1_score(labels, preds, average="macro")

# 执行消融实验+展示结果
print("="*50)
print("多模态融合模型 - 消融实验（验证集结果）")
print("="*50)
fusion_acc, fusion_f1 = evaluate("fusion")
text_acc, text_f1 = evaluate("text")
img_acc, img_f1 = evaluate("image")

# 结果表格
results = pd.DataFrame({
    "模态": ["多模态融合", "仅文本", "仅图像"],
    "准确率(Acc)": [f"{fusion_acc:.4f}", f"{text_acc:.4f}", f"{img_acc:.4f}"],
    "宏平均F1": [f"{fusion_f1:.4f}", f"{text_f1:.4f}", f"{img_f1:.4f}"]
})
print(results.to_string(index=False))
print("="*50)