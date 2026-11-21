"""Tiny training run: teach a small network to compute (a + b) mod P.

The data is generated here, so nothing has to be downloaded.
Takes under a minute, even on a laptop CPU.

Outputs (in --out, default results/):
    metrics.jsonl   one line per epoch: loss, train accuracy, test accuracy
    model.pt        final weights
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn


def make_data(p, train_frac, seed):
    """Every pair (a, b) with 0 <= a, b < p, shuffled and split into train/test."""
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    x = torch.stack([a.flatten(), b.flatten()], dim=1)
    y = (x[:, 0] + x[:, 1]) % p
    order = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
    n_train = int(train_frac * len(x))
    train, test = order[:n_train], order[n_train:]
    return x[train], y[train], x[test], y[test]


class Net(nn.Module):
    def __init__(self, p, dim):
        super().__init__()
        self.embed = nn.Embedding(p, dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, 4 * dim), nn.ReLU(), nn.Linear(4 * dim, p)
        )

    def forward(self, x):
        return self.mlp(self.embed(x).flatten(1))


@torch.no_grad()
def accuracy(model, x, y):
    return (model(x).argmax(dim=1) == y).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1.0)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    if device == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)

    x_train, y_train, x_test, y_test = (
        t.to(device) for t in make_data(args.p, args.train_frac, args.seed)
    )
    model = Net(args.p, args.dim).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(args.out, exist_ok=True)
    metrics_path = os.path.join(args.out, "metrics.jsonl")
    start = time.time()

    with open(metrics_path, "w") as f:
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = torch.randperm(len(x_train), device=device)
            for i in range(0, len(order), args.batch_size):
                idx = order[i : i + args.batch_size]
                loss = loss_fn(model(x_train[idx]), y_train[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()

            model.eval()
            row = {
                "epoch": epoch,
                "loss": round(loss.item(), 4),
                "train_acc": round(accuracy(model, x_train, y_train), 4),
                "test_acc": round(accuracy(model, x_test, y_test), 4),
                "seconds": round(time.time() - start, 1),
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            if epoch % 10 == 0 or epoch == 1:
                print(row, flush=True)

    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    print(f"done. wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
