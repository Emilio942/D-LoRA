import torch
import torch.nn as nn


class Substrate(nn.Module):
    def __init__(self, c=1.5, b=0.2):
        super().__init__()
        w = torch.tensor([float(c), -float(c)])
        bias = torch.tensor([float(b), -float(b)])
        self.fc = nn.Linear(2, 2)
        with torch.no_grad():
            self.fc.weight.copy_(torch.stack([w, -w]))
            self.fc.bias.copy_(bias)
        self.unfreeze()

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad_(True)

    def forward(self, x):
        return torch.tanh(self.fc(x))


class BaselineNet(nn.Module):
    def __init__(self, substrate, n_classes=4):
        super().__init__()
        self.substrate = substrate
        self.head = nn.Linear(2, n_classes)

    def hidden(self, x):
        return self.substrate(x)

    def forward(self, x):
        return self.head(self.hidden(x))


class DLoRANet(nn.Module):
    def __init__(self, substrate, adapter, n_classes=4):
        super().__init__()
        self.substrate = substrate
        self.adapter = adapter
        self.head = nn.Linear(2, n_classes)

    def hidden(self, x):
        return self.adapter(self.substrate(x))

    def forward(self, x):
        return self.head(self.hidden(x))
