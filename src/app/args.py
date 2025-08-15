import argparse

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, type=str)
    p.add_argument("--model", default="convnext_tiny", type=str)
    p.add_argument("--img_size", default=224, type=int)
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--epochs", default=30, type=int)
    p.add_argument("--lr", default=5e-4, type=float)
    p.add_argument("--weight_decay", default=0.05, type=float)
    p.add_argument("--warmup_epochs", default=3, type=int)
    p.add_argument("--freeze_epochs", default=2, type=int)
    p.add_argument("--mixup", default=0.2, type=float)
    p.add_argument("--cutmix", default=0.0, type=float)
    p.add_argument("--smoothing", default=0.1, type=float)
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--out_dir", default="outputs", type=str)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--auto_resume", action="store_true")
    p.add_argument("--pos_class", type=str, default="nao_hibrido")
    p.add_argument("--ema_decay", type=float, default=0.99)
    p.add_argument("--aug_backend", type=str, default="alb", choices=["alb", "torchvision"])
    p.add_argument("--aug_preset", type=str, default="strong", choices=["none", "light", "strong"])
    p.add_argument("--cv_splits", type=int, default=0)
    p.add_argument("--cv_fold",   type=int, default=0)

    # fusão imagem + tabular
    p.add_argument("--fusion_tab_csv_glob", type=str, default="",
                   help='Ex.: "data/csv/*_location_fixed.csv" ativa a fusão')

    return p
