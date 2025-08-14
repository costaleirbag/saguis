# 🐒 Saguis Classifier

A training pipeline for fine-tuning image classification models (e.g., ConvNeXt) on marmoset (sagui) datasets — supporting **hybrid vs. non-hybrid** classification, multiple sources, and MLflow experiment tracking.

---

## 📂 Project Structure

```

data/
│
├── raw\_aurita/           # Original aurita images by class
├── raw\_jacchus/          # Original jacchus images by class
├── raw\_penicillata/      # Penicillata images (unlabeled, for later testing)
│
├── train/                # Train split after `saguis-split`
├── val/                  # Validation split after `saguis-split`
│
└── csv/                  # CSVs with "url,class" format for downloading images

outputs/
│
├── best\_model.pth        # Best EMA model weights (highest val acc)
├── last.pth              # Last checkpoint (for resume)
├── class\_to\_idx.json     # Mapping from class name to index
└── mlruns/               # MLflow tracking logs

````

---

## ⚡ Commands

### 🆕 Fresh Training Run
```bash
poetry run saguis-train \
  --data_dir data \
  --model convnext_tiny \
  --epochs 15 \
  --warmup_epochs 1 \
  --batch_size 16 \
  --img_size 224 \
  --mixup 0.1 \
  --lr 5e-4 --weight_decay 0.05 \
  --freeze_epochs 2 \
  --out_dir outputs/aurita_test
````

---

### 🔄 Resume from Last Checkpoint Automatically

Searches for `last.pth` inside `--out_dir`:

```bash
poetry run saguis-train \
  --data_dir data \
  --auto_resume \
  --out_dir outputs/aurita_test
```

---

### 📂 Resume from a Specific Checkpoint

```bash
poetry run saguis-train \
  --data_dir data \
  --resume outputs/aurita_test/last.pth \
  --out_dir outputs/aurita_test
```

---

### 🎯 Fine-Tuning on New Dataset

Freezes backbone for first epochs:

```bash
poetry run saguis-train \
  --data_dir data/new_dataset \
  --freeze_epochs 3 \
  --epochs 15
```

---

### 🔀 Create Train/Validation Split

```bash
poetry run saguis-split \
  --data_root data \
  --train_ratio 0.9 \
  --clear_existing
```

---

### 📊 MLflow UI for Metrics

```bash
mlflow ui
```

Open: [http://localhost:5000](http://localhost:5000)

---

## 🧭 Resume Logic Diagram

```
                ┌────────────────────────────┐
                │   Start saguis-train run    │
                └─────────────┬──────────────┘
                              │
              ┌───────────────▼───────────────┐
              │ --auto_resume flag provided?  │
              └───────┬───────────────────────┘
                      │Yes
                      ▼
       ┌─────────────────────────────────────┐
       │ Search for last.pth in --out_dir     │
       │ If found → load checkpoint           │
       └─────────────────────────────────────┘
                      │
                 ┌────▼────┐
                 │ Resume  │
                 └─────────┘
                      │
               No (--resume flag?)
                      │
        Yes           ▼            No
         ┌────────────┼──────────────┐
         │ Load from  │              │
         │ given path │              │
         └────────────┘              │
                                      ▼
                                Start Fresh
```

---

## 🧪 Training Tips

* **Imbalanced data**: If one class is much bigger than another, consider:

  * Using `WeightedRandomSampler`
  * Data augmentation (`mixup`, `RandomHorizontalFlip`, `ColorJitter`)
  * Lower learning rate for fine-tuning to avoid overfitting minority class

* **Freeze strategy**: Start with frozen backbone for `freeze_epochs` to let the head adapt, then unfreeze.

* **Monitoring**: Use MLflow to track `train_loss`, `val_loss`, `val_acc`, `ema_val_acc`, and learning rate over time.

---

## 📌 Arguments Explained

| Argument          | Description                                     |
| ----------------- | ----------------------------------------------- |
| `--data_dir`      | Root folder with `train/` and `val/` subfolders |
| `--model`         | Backbone architecture (e.g., `convnext_tiny`)   |
| `--epochs`        | Total training epochs                           |
| `--warmup_epochs` | Gradual LR warmup at start                      |
| `--batch_size`    | Number of samples per batch                     |
| `--img_size`      | Image resize size                               |
| `--mixup`         | Mixup augmentation alpha                        |
| `--lr`            | Learning rate                                   |
| `--weight_decay`  | Weight decay for regularization                 |
| `--freeze_epochs` | Freeze backbone for given epochs                |
| `--out_dir`       | Output folder for models and logs               |
| `--auto_resume`   | Resume from last checkpoint in `out_dir`        |
| `--resume`        | Resume from specific checkpoint path            |

---

## 📸 Example Output

### Console Log

```
Epoch 12/15
Train loss 0.5613 | Train acc 82.69% | Val loss 0.5989 | Val acc 81.58% | EMA Val acc 31.58% | LR 0.000462

Epoch 13/15
Train loss 0.5443 | Train acc 82.21% | Val loss 0.4722 | Val acc 81.58% | EMA Val acc 31.58% | LR 0.000500
```

### MLflow Metrics UI

![MLflow Training Curves](docs/mlflow_example.png)

---

## 🐾 Example Workflow

```bash
# 1. Download datasets
poetry run saguis-download \
  --csv data/csv/raw_aurita.csv \
  --source aurita \
  --out_sources data/raw_sources \
  --consolidate_dir data/raw \
  --split_after_download 0.85 \
  --workers 8 --link_mode hard

# 2. Merge aurita + jacchus and split
poetry run saguis-split --data_root data --train_ratio 0.9 --clear_existing

# 3. Train
poetry run saguis-train \
  --data_dir data \
  --model convnext_tiny \
  --epochs 15 \
  --warmup_epochs 1 \
  --out_dir outputs/combined

# 4. View training in MLflow
mlflow ui
```

---

## 📜 License

MIT License
