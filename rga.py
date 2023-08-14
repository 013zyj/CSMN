# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch import nn
from torch.nn import functional as F

import pdb


# ===================
#     RGA Module
# ===================

class Non_local(nn.Module):
    def __init__(self, in_channel, in_spatial,use_channel=True,cha_ratio=4, spa_ratio=4,down_ratio=4):
        super(Non_local, self).__init__()
        self.use_channel = use_channel
        self.in_channel = in_channel
        self.inter_channel = in_channel // cha_ratio
        self.in_spatial = in_spatial
        self.inter_spatial = in_spatial // spa_ratio
        
        # input: 256*64*32
        # Embedding functions for original features
        self.gx_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
         )
        self.gg_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel*2, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
        if self.use_channel:
            num_channel_c = 1 + self.inter_channel
            self.W_channel = nn.Sequential(
                    nn.Conv2d(in_channels=num_channel_c, out_channels=num_channel_c // down_ratio,
                              kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(num_channel_c // down_ratio),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=num_channel_c // down_ratio, out_channels=1,
                              kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(1)
                )
        self.theta_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )
        self.phi_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )

    def forward(self, x):
         if self.use_channel:
            b, c, h, w = x.size()# 
            xc = x.view(b, c, -1).permute(0, 2, 1).unsqueeze(-1)#b*2048*128*1
            theta_xc = self.theta_channel(xc).squeeze(-1).permute(0, 2, 1) #b*128*256
            phi_xc = self.phi_channel(xc).squeeze(-1)#b*256*128
            Gc = torch.matmul(theta_xc, phi_xc)#b*128*128
            Gc_in = Gc.permute(0, 2, 1).unsqueeze(-1)#b*128*128*1
            Gc_out = Gc.unsqueeze(-1)#b*128*128*1
            Gc_joint = torch.cat((Gc_in, Gc_out), 1)#256*128*1
            Gc_joint = self.gg_channel(Gc_joint)#64*128*1

            g_xc = self.gx_channel(xc)#256*128*1
            g_xc = torch.mean(g_xc, dim=1, keepdim=True)#1*128*1
            yc = torch.cat((g_xc, Gc_joint), 1)#65*128*1

            W_yc = self.W_channel(yc).transpose(1, 2)#128*1*1
            out = F.sigmoid(W_yc) * x#(b*256*64*32)
#             out = out*x
            return out

