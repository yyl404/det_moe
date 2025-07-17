from typing import Tuple, List, Union, Optional

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as NF
import torchvision.transforms.functional as TF

from det_moe.registry import MODELS


@MODELS.register_module()
class GRUMultiLevel(nn.Module):
    def __init__(self,
                 in_channels: Union[List[int], Tuple[int]],
                 out_channels: Union[List[int], Tuple[int]],
                 kernel_sizes: Union[List[int], Tuple[int]],
                 paddings: Union[List[int], Tuple[int]]):
        super(GRUMultiLevel, self).__init__()

        self.grus = nn.ModuleList()
        self.output_channels = out_channels

        for _in_channel, _out_channel, _kernel_size, _p in zip(in_channels, out_channels, kernel_sizes, paddings):
            self.grus.append(GRU(_in_channel, _out_channel, _kernel_size, _p))

    def forward(self,
                x:Union[List[Tensor], Tuple[Tensor]], 
                h:Optional[Union[List[Tensor], Tuple[Tensor]]]=None) \
                    -> Union[List[Tensor], Tuple[Tensor]]:
        if h is None:
            h = []
            for _x, HC in zip(x, self.output_channels):
                N, _, H, W = _x.shape
                _h = torch.zeros((N, HC, H, W), dtype=torch.float, device=_x.device)
                h.append(_h)
        assert len(x)==len(h), f"Inconsistent feature level num between x:{len(x)} and h:{len(h)}"
        y = []
        for _x, _h, _gru in zip(x, h, self.grus):
            if _x.shape[2] != _h.shape[2] or _x.shape[3] != _h.shape[3]:
                _x = TF.resize(_x, _h.shape[2:4], TF.InterpolationMode.NEAREST)
            y.append(_gru(_x, _h))
        return tuple(y)
    

class GRU(nn.Module):
    def __init__(self, input_channel:int, output_channel:int, kernel_size:int, padding:int):
        super(GRU, self).__init__()

        # filters used for gates
        gru_input_channel = input_channel + output_channel
        self.output_channel = output_channel

        self.gate_conv = nn.Conv2d(gru_input_channel, output_channel * 2, kernel_size, padding=padding)
        self.reset_gate_norm = nn.GroupNorm(1, output_channel, 1e-5, True)
        self.update_gate_norm = nn.GroupNorm(1, output_channel, 1e-5, True)

        # filters used for outputs
        self.output_conv = nn.Conv2d(gru_input_channel, output_channel, kernel_size, padding=padding)
        self.output_norm = nn.GroupNorm(1, output_channel, 1e-5, True)

        self.activation = nn.Tanh()

    def gates(self, x, h):
        # x = N x C x H x W
        # h = N x C x H x W

        # c = N x C*2 x H x W
        c = torch.cat((x, h), dim=1)
        f = self.gate_conv(c)

        # r = reset gate, u = update gate
        # both are N x O x H x W
        C = f.shape[1]
        r, u = torch.split(f, C // 2, 1)

        rn = self.reset_gate_norm(r)
        un = self.update_gate_norm(u)
        rns = NF.sigmoid(rn)
        uns = NF.sigmoid(un)
        return rns, uns

    def output(self, x, h, r):
        f = torch.cat((x, r * h), dim=1)
        o = self.output_conv(f)
        on = self.output_norm(o)
        return on

    def forward(self, x, h=None):
        N, _, H, W = x.shape
        HC = self.output_channel
        if(h is None):
            h = torch.zeros((N, HC, H, W), dtype=torch.float, device=x.device)
        r, u = self.gates(x, h)
        o = self.output(x, h, r)
        y = self.activation(o)
        return u * y + (1 - u) * h