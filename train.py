import argparse
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

class AlzheimerCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def check_folder(p: Path, name: str):
    if not p.exists():
        raise FileNotFoundError(f"{name} papka topilmadi: {p}")
    classes = [d.name for d in p.iterdir() if d.is_dir()]
    if not classes:
        raise FileNotFoundError(f"{name} ichida class papkalar yo‘q: {p}")
    return sorted(classes)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)         # kichik batch
    ap.add_argument("--max_train", type=int, default=2000)  # tez test uchun limit
    ap.add_argument("--max_test", type=int, default=800)
    args = ap.parse_args()

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)

    print("Train classlar:", check_folder(train_dir, "train"))
    print("Test classlar :", check_folder(test_dir, "test"))

    tfm = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_full = datasets.ImageFolder(str(train_dir), transform=tfm)
    test_full  = datasets.ImageFolder(str(test_dir), transform=tfm)

    print("Class mapping:", train_full.class_to_idx)
    print("Train total:", len(train_full), "| Test total:", len(test_full))

    # limit (qotmasin + tez yuradi)
    train_idx = list(range(min(args.max_train, len(train_full))))
    test_idx  = list(range(min(args.max_test, len(test_full))))
    train_data = Subset(train_full, train_idx)
    test_data  = Subset(test_full, test_idx)

    # Windows uchun xavfsiz: num_workers=0
    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_data, batch_size=args.batch, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = AlzheimerCNN(num_classes=len(train_full.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        correct = total = 0

        for i, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

            # har 50 batchda progress ko‘rsatadi
            if i % 10 == 0 or i == 1:
                print(f"  [Train] epoch {epoch+1} batch {i}/{len(train_loader)} loss={loss.item():.4f}")

        train_acc = 100 * correct / total

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        test_acc = 100 * correct / total
        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | time: {dt:.1f}s")

    torch.save(model.state_dict(), "model_4class.pth")
    print("✅ Tayyor: model_4class.pth saqlandi")

if __name__ == "__main__":
    main()

