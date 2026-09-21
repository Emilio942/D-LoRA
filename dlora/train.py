import torch
import torch.nn as nn

from .conflict import conflict_loss


@torch.no_grad()
def accuracy(model, x, y):
    model.eval()
    return (model(x).argmax(dim=1) == y).float().mean().item()


@torch.no_grad()
def per_class_accuracy(model, x, y, n_classes=4):
    model.eval()
    pred = model(x).argmax(dim=1)
    return [(pred[y == c] == c).float().mean().item() for c in range(n_classes)]


def train(model, x, y, params, epochs=8000, lr=1e-2, lambda_conflict=0.0,
          seed=0, log_every=2000):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(params, lr=lr)
    ce = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        loss = ce(model(x), y)
        if lambda_conflict > 0:
            loss = loss + lambda_conflict * conflict_loss(model.hidden(x))
        loss.backward()
        opt.step()
        if epoch % log_every == 0:
            print(
                f"  Epoche {epoch:5d} | Loss {loss.item():.4f} "
                f"| Acc {accuracy(model, x, y):.3f}"
            )
