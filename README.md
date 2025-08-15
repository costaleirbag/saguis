# 🐒 Saguis Classifier

A robust pipeline for fine-tuning image classification models (e.g., ConvNeXt) on marmoset (sagui) datasets — supporting **hybrid vs. non-hybrid** classification, multiple sources, advanced data augmentation, weighted sampling, MLflow experiment tracking, and reproducible splits.

---

## 📂 Project Structure

```
src/app/
    datasets.py         # Data loading, augmentation (Albumentations/Torchvision), weighted sampling
    train.py            # Training loop, MLflow logging, resume logic, metrics (F1, AUROC, threshold sweep)
    models.py           # Model creation, optimizer, scheduler
    infer.py            # Inference utilities
    split_dataset.py    # Deterministic train/val split with reproducibility
    download_images.py  # Download images from CSVs
    __init__.py         # Module init
```

```
data/
    raw/                # Consolidated images by class (after download)
    train/              # Train split (after saguis-split)
    val/                # Validation split (after saguis-split)
    csv/                # CSVs with "url,class" format for downloading images
outputs/
    best_model.pth      # Best EMA model weights (highest val acc)
    last.pth            # Last checkpoint (for resume)
    class_to_idx.json   # Mapping from class name to index
    mlruns/             # MLflow tracking logs
```

---

## 🚀 Workflow Example

### 1. Download and Consolidate Datasets

```bash
poetry run saguis-download \
  --csv data/csv/raw_aurita.csv \
  --source aurita \
  --out_sources data/raw_sources \
  --consolidate_dir data/raw \
  --workers 8 --link_mode hard

poetry run saguis-download \
  --csv data/csv/raw_jacchus.csv \
  --source jacchus \
  --out_sources data/raw_sources \
  --consolidate_dir data/raw \
  --workers 8 --link_mode hard
```

### 2. Create Train/Validation Split (Deterministic, Stratified)

```bash
poetry run saguis-split --data_root data --train_ratio 0.9 --clear_existing
# Output:
hibrido: total=123  train=111  val=12
nao_hibrido: total=403  train=363  val=40
✅ Split criado em train/ e val/
```

### 3. Training (with Resume, Weighted Sampling, Augmentation)

```bash
poetry run saguis-train \
  --data_dir data \
  --model convnext_tiny \
  --epochs 15 --warmup_epochs 1 \
  --batch_size 16 --img_size 224 \
  --mixup 0.1 --lr 5e-4 --weight_decay 0.05 \
  --freeze_epochs 2 \
  --out_dir outputs/aurita_jacchus \
  --auto_resume \
  --ema_decay 0.99 \
  --pos_class hibrido \
  --aug_backend alb \
  --aug_preset strong
```

#### Key Features in Training:
- **Resume logic**: `--auto_resume` finds `last.pth` in `out_dir` and resumes automatically. Or use `--resume path/to/checkpoint.pth`.
- **WeightedRandomSampler**: Handles class imbalance by sampling inversely to class frequency.
- **Augmentation**: Choose backend (`alb` for Albumentations, `torchvision` for Torchvision) and preset (`none`, `light`, `strong`).
- **MLflow logging**: All metrics, parameters, and artifacts are logged for experiment tracking.
- **Metrics**: F1-macro, AUROC, threshold sweep, orientation correction (flipping), EMA parameter gap.
- **Freeze strategy**: Backbone frozen for `freeze_epochs`, then unfrozen for full fine-tuning.

#### Console Output Example:
```
Epoch 12/15
Train loss 0.5613 | Train acc 82.69% | Val loss 0.5989 | Val acc 81.58% | EMA Val acc 83.33% | [BASE] F1 0.812 AUROC 0.900 th 0.50 flip=False | [EMA]  F1 0.825 AUROC 0.915 th 0.55 flip=False | EMA_gap 1.2e-4 | LR 0.000462
```

---

## 🧬 Data Augmentation (datasets.py)

- **Albumentations**: Advanced transforms (Affine, Perspective, CLAHE, Blur, Noise, GridDropout, etc.)
- **Torchvision**: Standard transforms (RandomResizedCrop, ColorJitter, GaussianBlur, RandomErasing, etc.)
- **Presets**: `none`, `light`, `strong` for easy switching
- **Backend selection**: `--aug_backend alb` or `--aug_backend torchvision`

---

## 🏋️ Weighted Sampling (datasets.py)

- Handles class imbalance automatically
- Uses `WeightedRandomSampler` (inverse frequency)
- Can be disabled for random sampling

---

## 🧪 Metrics & Thresholds (train.py)

- **F1-macro** and **AUROC** for both base and EMA models
- **Threshold sweep**: Finds best threshold for F1
- **Orientation correction**: Flips positive/negative if AUROC is higher
- **MLflow logging**: All metrics, parameters, and artifacts

---

## 🔀 Deterministic Split (split_dataset.py)

- Stratified by class
- Deterministic (uses hash + seed)
- Optionally clears previous splits
- Example:
```bash
poetry run saguis-split --data_root data --train_ratio 0.9 --clear_existing
```

---

## 🔁 Treinamento com K-Fold Cross Validation

O pipeline do Saguis suporta treinamento com validação cruzada k-fold para avaliação mais robusta do modelo. Isso é útil para medir a variabilidade do desempenho e evitar overfitting em splits específicos.

### Como usar k-fold:

1. **Prepare os dados**: Certifique-se de que todos os dados estejam em `data/raw/` organizados por classe.
2. **Escolha o número de folds**: Recomenda-se 5 ou 10 folds para datasets médios/grandes.
3. **Execute o loop de treinamento**: Para cada fold, crie splits determinísticos e treine separadamente.

#### Exemplo de script para 5-fold cross validation:

```bash
for fold in {0..4}; do
  poetry run saguis-split \
    --data_root data \
    --train_ratio 0.8 \
    --seed $fold \
    --clear_existing

  poetry run saguis-train \
    --data_dir data \
    --model convnext_tiny \
    --epochs 15 --warmup_epochs 1 \
    --batch_size 16 --img_size 224 \
    --mixup 0.1 --lr 5e-4 --weight_decay 0.05 \
    --freeze_epochs 2 \
    --out_dir outputs/fold_$fold \
    --auto_resume \
    --ema_decay 0.99 \
    --pos_class hibrido \
    --aug_backend alb \
    --aug_preset strong

done
```

- O parâmetro `--seed $fold` garante splits diferentes e reprodutíveis para cada fold.
- Os resultados de cada fold ficam em `outputs/fold_0`, `outputs/fold_1`, etc.
- Recomenda-se agregar métricas (média, desvio padrão) ao final para análise global.

#### Dicas:
- Use MLflow para comparar todos os experimentos dos folds.
- Para datasets pequenos, aumente o número de folds para melhor estimativa.
- O split é estratificado por classe, mantendo proporções em cada fold.

---

## 📊 MLflow UI for Metrics

```bash
mlflow ui
```
Open: [http://localhost:5000](http://localhost:5000)

---

## 🐾 Full Example Workflow

```bash
# 1. Download datasets
poetry run saguis-download --csv data/csv/raw_aurita.csv --source aurita --out_sources data/raw_sources --consolidate_dir data/raw --split_after_download 0.85 --workers 8 --link_mode hard

# 2. Merge aurita + jacchus and split
poetry run saguis-split --data_root data --train_ratio 0.9 --clear_existing

# 3. Train
poetry run saguis-train --data_dir data --model convnext_tiny --epochs 15 --warmup_epochs 1 --out_dir outputs/combined --auto_resume --aug_backend alb --aug_preset strong

# 4. View training in MLflow
mlflow ui
```

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
| `--pos_class`     | Name of positive class for metrics              |
| `--ema_decay`     | EMA decay rate                                 |
| `--aug_backend`   | Augmentation backend (`alb` or `torchvision`)   |
| `--aug_preset`    | Augmentation preset (`none`, `light`, `strong`) |

---

## 🐒 Inference Example (infer.py)

```python
from app.infer import load_model, infer_image

model, class_to_idx = load_model('outputs/aurita_test/best_model.pth', 'convnext_tiny', num_classes=2, device='cpu')
pred = infer_image(model, 'path/to/image.jpg', img_size=224, device='cpu')
print('Predicted class:', [k for k,v in class_to_idx.items() if v==pred][0])
```

---

## 📜 License
MIT License
