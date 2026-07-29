"""Evaluate current model on validation set."""
import json, torch, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from train import BETOMultiLabel, TagDataset, compute_metrics, find_optimal_thresholds
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

with open("taxonomy.json", "r", encoding="utf-8") as f:
    taxonomy = json.load(f)
tag2idx = {t.lower(): i for i, t in enumerate(taxonomy)}

with open("dataset/val.json", "r", encoding="utf-8") as f:
    val_data = json.load(f)

model = BETOMultiLabel("dccuchile/bert-base-spanish-wwm-cased", len(taxonomy)).to(device)
model.load_state_dict(torch.load("model/pytorch_model.bin", map_location=device, weights_only=True))
model.eval()

tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
val_dataset = TagDataset(val_data, tokenizer, tag2idx)
val_loader = DataLoader(val_dataset, batch_size=16)

# Evaluate with threshold 0.5
all_logits = []
all_labels = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids, attention_mask)
        all_logits.append(logits.cpu())
        all_labels.append(batch["labels"].cpu())

y_logits = torch.cat(all_logits).numpy()
y_true = torch.cat(all_labels).numpy()
y_pred = (y_logits >= 0.5).astype(int)

metrics = compute_metrics(y_true, y_pred, taxonomy)
print(f"\n=== With threshold 0.5 ===")
print(f"F1 Macro: {metrics['f1_macro']:.4f}")
print(f"F1 Micro: {metrics['f1_micro']:.4f}")
print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
print(f"Precision Macro: {metrics['precision_macro']:.4f}")
print(f"Recall Macro: {metrics['recall_macro']:.4f}")

# Find optimal thresholds
print(f"\nFinding optimal thresholds...")
thresholds = find_optimal_thresholds(model, val_loader, device, taxonomy)
np.save("model/thresholds.npy", thresholds)
print(f"Thresholds saved")

# Evaluate with optimal thresholds
y_pred_opt = (y_logits >= thresholds.reshape(1, -1)).astype(int)
metrics_opt = compute_metrics(y_true, y_pred_opt, taxonomy)
print(f"\n=== With optimal thresholds ===")
print(f"F1 Macro: {metrics_opt['f1_macro']:.4f}")
print(f"F1 Micro: {metrics_opt['f1_micro']:.4f}")
print(f"Hamming Loss: {metrics_opt['hamming_loss']:.4f}")

# Save metrics
metrics_clean = {k: v for k, v in metrics_opt.items() if k != "per_tag_f1"}
metrics_clean["per_tag_f1"] = {t: float(v) for t, v in metrics_opt["per_tag_f1"].items()}
with open("model/metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics_clean, f, ensure_ascii=False, indent=2)

print(f"\nPer-tag F1 (worst to best):")
sorted_tags = sorted(metrics_opt["per_tag_f1"].items(), key=lambda x: x[1])
for tag, f1 in sorted_tags:
    print(f"  {tag}: {f1:.4f}")

print(f"\nDone! Metrics saved to model/metrics.json")
