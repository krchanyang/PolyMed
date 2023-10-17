import torch
import torch.nn as nn


class Disease_classifier(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        h = self.layer1(x)
        return h


class Residual_block(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.residual_layer = torch.nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_size),
            nn.BatchNorm1d(output_size),
        )

    def forward(self, x):
        h = self.residual_layer(x)
        h = x + h

        return h


class Linear_resnet(nn.Module):
    def __init__(self, input_size, output_size, block_num):
        super().__init__()
        self.residual_block = nn.ModuleList()
        self.initial_layer = torch.nn.Sequential(
            nn.Linear(input_size, output_size), nn.BatchNorm1d(output_size), nn.ReLU()
        )
        for block in range(block_num):
            self.residual_block.append(Residual_block(output_size, output_size))

    def forward(self, x):
        h = self.initial_layer(x)
        for block in self.residual_block:
            h = block(h)

        return h
