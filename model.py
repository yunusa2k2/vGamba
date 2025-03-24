#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 14:38:19 2025
"""

import torch
import torch.nn as nn
from thop import profile
from mamba_ssm import Mamba
from torchinfo import summary
import torch.nn.functional as F
from einops import pack, repeat, unpack
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GambaCell(nn.Module):
    def __init__(self, n_dims, height, width):
        super().__init__()
        self.rel_h = nn.Parameter(torch.randn([1, n_dims, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, n_dims, width, 1]), requires_grad=True)
        self.mamba_block = Mamba(d_model=n_dims, d_state=64, d_conv=4, expand=2)
        
        self.gate = nn.Sequential(
            nn.Conv1d(n_dims, n_dims, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, H, W) -> Reshape to (B, seq_len, C)
        B, C, height, width = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # (B, seq_len, C)
        
        position = (self.rel_h + self.rel_w).view(1, C, -1).permute(0, 2, 1)  # (B, seq_len, C)
        x = x + position
        
        # Fix: Permute before Conv1d
        x_permuted = x.permute(0, 2, 1)  # (B, C, seq_len)
        context = self.gate(x_permuted)  # Apply Conv1d
        context = context.permute(0, 2, 1)  # Back to (B, seq_len, C)
        
        x_mamba = self.mamba_block(x)
        
        # Combine outputs
        x = x_mamba * context
        x = x.view(B, C, height, width)
        return x

class MHSA(nn.Module):    #with register tokens (iiANET)
    def __init__(self, n_dims, width, height, head, num_register_tokens=1):
        super(MHSA, self).__init__()
        self.head = head

        self.Q = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.K = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.V = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        
        self.rel_h = nn.Parameter(torch.randn([1, head, n_dims // head, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, head, n_dims // head, width, 1]), requires_grad=True)
        
        self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, width*height, width*height))
        self.register_tokens_v = nn.Parameter(torch.randn(num_register_tokens, n_dims // head, width*height))
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        
        q = self.Q(x).view(n_batch, self.head, C // self.head, -1)
        k = self.K(x).view(n_batch, self.head, C // self.head, -1)
        v = self.V(x).view(n_batch, self.head, C // self.head, -1)
        
        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.head, C // self.head, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)
        energy = content_content + content_position
        
        r_qk = repeat(
            self.register_tokens, 
            'n w h -> b n w h', 
            b=n_batch
        )
        
        r_v = repeat(
            self.register_tokens_v, 
            'n w h -> b n w h', 
            b=n_batch
        )

        energy, _ = pack([energy, r_qk], 'b * w h')
        v, ps = pack([v, r_v], 'b * d h')
        
        attention = self.softmax(energy)
        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out, _ = unpack(out, ps, 'b * d h')
        out = out.view(n_batch, C, width, height)
        return out

class Gating(nn.Module):
    def __init__(self, input_dim):
        super(Gating, self).__init__()
        self.proj_a = nn.Conv2d(input_dim // 2, input_dim, kernel_size=1)
        self.proj_b = nn.Conv2d(input_dim // 2, input_dim, kernel_size=1)
        self.gate = nn.Conv2d(input_dim, 1, kernel_size=1)
    
    def forward(self, a, b):
        a_proj = self.proj_a(a)
        b_proj = self.proj_b(b)
        gate_a = torch.sigmoid(self.gate(a_proj))
        gate_b = torch.sigmoid(self.gate(b_proj))
        gated_output = gate_a * a_proj + gate_b * b_proj
        return gated_output

class GambaBlock(nn.Module):
    def __init__(self, planes, height, width, heads=4):
        super().__init__()
        self.ggnn = GambaCell(planes // 2, height, width)
        self.mhsa = MHSA(planes // 2, width, height, head=heads)
        self.gating = Gating(input_dim=planes)

    def forward(self, x):
        
        n_batch, n_dims, width, height = x.size()
        half_dims = n_dims // 2
        x1 = x[:, :half_dims, :, :]
        x2 = x[:, half_dims:, :, :]
        x_ggnn = self.ggnn(x1)
        x_mhsa = self.mhsa(x2)
        x_out = self.gating(x_ggnn, x_mhsa)
        return x_out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, heads=4, mhsa=False, resolution=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if not mhsa:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        else:
            self.conv2 = nn.ModuleList()
            self.conv2.append(GambaBlock(planes, width=int(resolution[0]), height=int(resolution[1])))
            if stride == 2:
                self.conv2.append(nn.AvgPool2d(2, 2))
            self.conv2 = nn.Sequential(*self.conv2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, resolution=(224, 224), heads=4):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.resolution = list(resolution)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if self.conv1.stride[0] == 2:
            self.resolution[0] /= 2
        if self.conv1.stride[1] == 2:
            self.resolution[1] /= 2
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # for ImageNet
        if self.maxpool.stride == 2:
            self.resolution[0] /= 2
            self.resolution[1] /= 2

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, heads=heads, gamba=True) # 
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, heads=heads, gamba=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.3), # All architecture deeper than ResNet-200 dropout_rate: 0.2
            nn.Linear(512 * block.expansion, num_classes)
        )

    def _make_layer(self, block, planes, num_blocks, stride=1, heads=4, gamba=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, heads, gamba, self.resolution))
            if stride == 2:
                self.resolution[0] /= 2
                self.resolution[1] /= 2
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out) # for ImageNet

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def ResNet50(num_classes=1000, resolution=(224, 224), heads=4):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, resolution=resolution, heads=heads)

def measure_throughput_and_memory(model, batch_size=32, num_warmup=10, num_iters=100):
    model.eval()
    x = torch.randn([batch_size, 3, 224, 224]).to(device)

    # Warm-up runs
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)

    torch.cuda.synchronize()

    # Reset memory stats before timing
    torch.cuda.reset_peak_memory_stats(device)
    initial_memory = torch.cuda.memory_allocated(device)

    # Timed runs
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model(x)

    torch.cuda.synchronize()

    # Memory measurements
    peak_memory = torch.cuda.max_memory_allocated(device)
    memory_used = peak_memory - initial_memory

    return memory_used / (1024**2)  # Convert to MB

def main():
    x = torch.randn([1, 3, 224, 224]).to(device)
    model = ResNet50(resolution=tuple(x.shape[2:]), heads=4).to(device)

    flops, params = profile(model, inputs=(x,))
    throughput, memory_usage = measure_throughput_and_memory(model, batch_size=32)

    print(f"Total FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Total Parameters: {params / 1e6:.2f}M")
    print(f"Peak Memory Usage: {memory_usage:.2f} MB")

if __name__ == '__main__':
    main()