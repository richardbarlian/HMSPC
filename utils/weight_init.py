import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        # Xavier uniform is correct for tanh (Kaiming is for ReLU)
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.GRUCell, nn.RNNCell, nn.LSTMCell)):
        for name, param in m.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
