import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models


# =========================================================
# REPRODUCIBILITY
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =========================================================
# MODEL
# =========================================================
def build_model(num_classes: int, pretrained: bool = True):
    if pretrained:
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            print("Pretrained weights yuklanmadi. weights=None bilan davom etiladi.")
            model = models.resnet18(weights=None)
    else:
        model = models.resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# =========================================================
# HELPERS
# =========================================================
def check_folder(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"{name} papka topilmadi: {path}")
    classes = sorted([d.name for d in path.iterdir() if d.is_dir()])
    if not classes:
        raise FileNotFoundError(f"{name} ichida class papkalar yo‘q: {path}")
    return classes


def make_class_weights_and_sampler(targets, num_classes):
    targets_np = np.array(targets, dtype=np.int64)
    counts = np.bincount(targets_np, minlength=num_classes).astype(np.float32)

    # class imbalance uchun og‘irlik
    class_weights = 1.0 / (counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32)

    # sample-level weights
    sample_weights = class_weights_t[torch.tensor(targets_np, dtype=torch.long)]

    sampler = WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    return class_weights_t, counts, sampler


def confusion_matrix_torch(y_true, y_pred, num_classes):
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_report_from_confusion(cm, class_names):
    rows = []
    for i, cls in enumerate(class_names):
        tp = cm[i, i].item()
        total_true = cm[i].sum().item()
        total_pred = cm[:, i].sum().item()

        recall = tp / total_true if total_true > 0 else 0.0
        precision = tp / total_pred if total_pred > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        rows.append({
            "class": cls,
            "support": total_true,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })
    return pd.DataFrame(rows)


def save_confusion_matrix_csv(cm, class_names, output_path: Path):
    df = pd.DataFrame(cm.numpy(), index=class_names, columns=class_names)
    df.to_csv(output_path, encoding="utf-8-sig")


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def find_pair_indices(class_to_idx):
    """
    Ikkala naming scheme ni ham qo‘llab-quvvatlaydi:
    - NoImpairment / VeryMildImpairment
    - NonDemented / VeryMildDemented
    """
    no_candidates = ["NoImpairment", "NonDemented"]
    vm_candidates = ["VeryMildImpairment", "VeryMildDemented"]

    no_name = next((x for x in no_candidates if x in class_to_idx), None)
    vm_name = next((x for x in vm_candidates if x in class_to_idx), None)

    if no_name is None or vm_name is None:
        return None

    return {
        "no_name": no_name,
        "vm_name": vm_name,
        "no_idx": class_to_idx[no_name],
        "vm_idx": class_to_idx[vm_name],
    }


# =========================================================
# TRAIN / EVAL
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total = 0
    correct = 0

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

        if batch_idx == 1 or batch_idx % 20 == 0:
            print(f"[Train] batch {batch_idx}/{len(loader)} loss={loss.item():.4f}")

    avg_loss = total_loss / max(total, 1)
    avg_acc = correct / max(total, 1)
    return avg_loss, avg_acc


def evaluate(model, loader, criterion, device, num_classes):
    model.eval()

    total_loss = 0.0
    total = 0
    correct = 0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_true.extend(labels.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())

    avg_loss = total_loss / max(total, 1)
    avg_acc = correct / max(total, 1)
    cm = confusion_matrix_torch(all_true, all_pred, num_classes)

    return avg_loss, avg_acc, cm


# =========================================================
# MAIN
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True, help="Train dataset papkasi")
    ap.add_argument("--test_dir", required=True, help="Test/validation dataset papkasi")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--pretrained", action="store_true", help="ImageNet pretrained backbone ishlatish")
    args = ap.parse_args()

    set_seed(args.seed)

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    train_classes = check_folder(train_dir, "train")
    test_classes = check_folder(test_dir, "test")

    print("Train classlar:", train_classes)
    print("Test classlar :", test_classes)

    if train_classes != test_classes:
        raise ValueError("Train va test klasslari bir xil emas.")

    image_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.08, contrast=0.08),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    test_ds = datasets.ImageFolder(str(test_dir), transform=test_tf)

    print("Train total:", len(train_ds))
    print("Test total :", len(test_ds))
    print("Class mapping:", train_ds.class_to_idx)

    if train_ds.classes != test_ds.classes:
        raise ValueError("ImageFolder train/test class tartibi bir xil emas.")

    train_targets = [label for _, label in train_ds.samples]
    class_weights, class_counts, sampler = make_class_weights_and_sampler(
        train_targets,
        num_classes=len(train_ds.classes),
    )

    print("Train class counts:", dict(zip(train_ds.classes, class_counts.tolist())))
    print("Class weights:", class_weights.tolist())

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = build_model(
        num_classes=len(train_ds.classes),
        pretrained=args.pretrained,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=0.05,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_test_acc = -1.0
    best_epoch = 0
    bad_epochs = 0
    history = []

    pair_info = find_pair_indices(train_ds.class_to_idx)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        test_loss, test_acc, cm = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            num_classes=len(train_ds.classes),
        )

        scheduler.step(test_acc)

        per_class_df = per_class_report_from_confusion(cm, train_ds.classes)

        epoch_info = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "test_loss": round(test_loss, 6),
            "test_acc": round(test_acc, 6),
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": round(time.time() - epoch_start, 2),
        }
        history.append(epoch_info)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
            f"test_loss={test_loss:.4f} test_acc={test_acc*100:.2f}% | "
            f"lr={optimizer.param_groups[0]['lr']:.6f} | "
            f"time={time.time() - epoch_start:.1f}s"
        )
        print(per_class_df.to_string(index=False))

        if pair_info is not None:
            no_idx = pair_info["no_idx"]
            vm_idx = pair_info["vm_idx"]
            print(
                f"[{pair_info['no_name']} <-> {pair_info['vm_name']}] "
                f"true {pair_info['no_name']} -> pred {pair_info['vm_name']}: {cm[no_idx, vm_idx].item()} | "
                f"true {pair_info['vm_name']} -> pred {pair_info['no_name']}: {cm[vm_idx, no_idx].item()}"
            )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            bad_epochs = 0

            checkpoint = {
                "arch": "resnet18",
                "model_state": model.state_dict(),
                "class_to_idx": train_ds.class_to_idx,
                "image_size": image_size,
                "mean": mean,
                "std": std,
                "best_test_acc": float(best_test_acc),
                "best_epoch": int(best_epoch),
                "history": history,
            }

            torch.save(checkpoint, "model_resnet18_4class.pth")
            save_confusion_matrix_csv(cm, train_ds.classes, output_dir / "best_confusion_matrix.csv")
            per_class_df.to_csv(output_dir / "best_per_class_report.csv", index=False, encoding="utf-8-sig")
            save_json(history, output_dir / "train_history.json")

            print("✅ Yangi best model saqlandi: model_resnet18_4class.pth")
        else:
            bad_epochs += 1
            print(f"Best model yangilanmadi. bad_epochs={bad_epochs}/{args.patience}")

        if bad_epochs >= args.patience:
            print("⏹ Early stopping ishga tushdi.")
            break

    print(f"✅ Trening tugadi. Best epoch={best_epoch}, best test acc={best_test_acc*100:.2f}%")


if __name__ == "__main__":
    main()