import torch


def create_optimizer(name, nn):
    """
    lr = 1e-3
    """
    if name == "adam_lr_1e3_wd_1e2":
        return torch.optim.Adam(nn.parameters(), lr=1e-3, weight_decay=0.01)

    if name == "adam_lr_1e3_wd_5e2":
        return torch.optim.Adam(nn.parameters(), lr=1e-3, weight_decay=0.05)

    if name == "adam_lr_1e3_wd_1e1":
        return torch.optim.Adam(nn.parameters(), lr=1e-3, weight_decay=0.1)

    if name == "adam_lr_1e3_wd_5e1":
        return torch.optim.Adam(nn.parameters(), lr=1e-3, weight_decay=0.5)

    """
    lr = 1e-4 
    """
    if name == "adam_lr_1e4_wd_1e2":
        return torch.optim.Adam(nn.parameters(), lr=1e-4, weight_decay=0.01)

    if name == "adam_lr_1e4_wd_5e2":
        return torch.optim.Adam(nn.parameters(), lr=1e-4, weight_decay=0.05)

    if name == "adam_lr_1e4_wd_1e1":
        return torch.optim.Adam(nn.parameters(), lr=1e-4, weight_decay=0.1)

    if name == "adam_lr_1e4_wd_5e1":
        return torch.optim.Adam(nn.parameters(), lr=1e-4, weight_decay=0.5)

    """
    lr = 1e-5
    """
    if name == "adam_lr_1e5_wd_1e2":
        return torch.optim.Adam(nn.parameters(), lr=1e-5, weight_decay=0.01)

    if name == "adam_lr_1e5_wd_5e2":
        return torch.optim.Adam(nn.parameters(), lr=1e-5, weight_decay=0.05)

    if name == "adam_lr_1e5_wd_1e1":
        return torch.optim.Adam(nn.parameters(), lr=1e-5, weight_decay=0.1)

    if name == "adam_lr_1e5_wd_5e1":
        return torch.optim.Adam(nn.parameters(), lr=1e-5, weight_decay=0.5)
