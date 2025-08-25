import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class WeberFchnerLayer(nn.Module):
    def __init__(self):
        super(WeberFchnerLayer, self).__init__()

        self.alpha = nn.Parameter(torch.empty(1))
        self.beta = nn.Parameter(torch.empty(1))
        self.lambda_param = nn.Parameter(torch.empty(1))

        self._init_parameters()

    def _init_parameters(self):
        # Initialize alpha with a normal distribution (mean=0, std=0.1)
        torch.nn.init.uniform_(self.alpha, a=0.0001, b=0.001)

        # Initialize beta with a uniform distribution between 0 and 1
        torch.nn.init.uniform_(self.beta, a=0.1, b=0.5)

        # Initialize lambda_param with Xavier initialization
        torch.nn.init.uniform_(self.lambda_param, a=-1, b=1)

    def forward(self, x):
        x_shifted = x - self.lambda_param
        x_shifted = F.softplus(x_shifted)
        x_shifted = torch.tanh(self.alpha * x_shifted * self.beta)
        x_shifted = x * (1 - x_shifted)
        return x_shifted


class BlockEmbedding(nn.Module):
    def __init__(self, in_channel, block_size, embed_dim, num_blockies, dropout):
        super(BlockEmbedding, self).__init__()
        # 逐步分块至块大小为16
        self.Blocker = nn.Sequential(
            nn.Conv3d(in_channel, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.Flatten(2)
        )

        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embed_dim), requires_grad=True)).cuda()
        self.pos_embedding = nn.Parameter(torch.randn(size=(1, num_blockies + 1, embed_dim), requires_grad=True)).cuda()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.Blocker(x)
        x=x.permute(0,2,1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)
        return x


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channel, pool_sizes):
        super(PyramidPoolingModule, self).__init__()
        self.pool_sizes = pool_sizes
        self.stage = nn.ModuleList([self._make_layer(in_channel, pool_size) for pool_size in self.pool_sizes])

    def _make_layer(self, in_channel, pool_size):

        return nn.Sequential(
            nn.AdaptiveAvgPool3d(pool_size),
            nn.Conv3d(in_channel, max(1, in_channel // len(self.pool_sizes)), kernel_size=1, bias=False),
            nn.LayerNorm([max(1, in_channel // len(self.pool_sizes)), pool_size, pool_size, pool_size]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w, d = x.size(2), x.size(3), x.size(4)

        pyramids = [F.interpolate(stage(x), size=(h, w, d), mode='trilinear', align_corners=True) for stage in self.stage]
        output = torch.cat([x] + pyramids, dim=1)
        return output



class WFE_Seg(nn.Module):
    def __init__(self, in_channel, img_size, block_size, embed_dim, num_blockies, dropout,
                 num_heads, activation, num_encoders, num_classes, target_depth, training=True):
        super(WFE_Seg, self).__init__()
        self.feature = embed_dim
        self.dropout = dropout
        self.training = training

        self.weber_fechner = WeberFchnerLayer()

        self.block_embedding = BlockEmbedding(in_channel, block_size, embed_dim, num_blockies, self.dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=self.dropout,
                                                   activation=activation, batch_first=True, norm_first=True)
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)

        # self.encoder_blocks = WindowAttention(embed_dim, num_heads, self.dropout, activation, num_encoders)
        self.ppm = PyramidPoolingModule(embed_dim, pool_sizes=[1, 2, 4, 8])

        self.decoder2 = nn.ConvTranspose3d(512+ 256, 256,  3, stride=2, padding=1, output_padding=1)  # b, 8, 15, 1
        self.decoder3 = nn.ConvTranspose3d(256 + 128, 64, 5, stride=2, padding=2, output_padding=1)
        self.decoder4 = nn.ConvTranspose3d(64 + 32,  16,  3, stride=2, padding=1, output_padding=1)

        self.map3 = nn.Sequential(
            nn.Conv3d(16, 2, 1, 1),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True),
            nn.Softmax(dim=1)
        )

        self.map2 = nn.Sequential(
            nn.Conv3d(64, 2, 1, 1),
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear', align_corners=True),
            nn.Softmax(dim=1)
        )

        self.map1 = nn.Sequential(
            nn.Conv3d(256, 2, 1, 1),
            nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear', align_corners=True),
            nn.Softmax(dim=1)
        )


        self.block_size = block_size
        self.target_depth = target_depth
        self.img_size = img_size

    def forward(self, x):

        x=x.cuda()

        # 跳跃连接层
        skip_connection1 = self.block_embedding.Blocker[0](x)
        skip_connection2 = self.block_embedding.Blocker[2](F.relu(skip_connection1, inplace=True))
        skip_connection3 = self.block_embedding.Blocker[4](F.relu(skip_connection2, inplace=True))
        skip_connection4 = self.block_embedding.Blocker[6](F.relu(skip_connection3, inplace=True))
        x = self.weber_fechner(x)

        x = self.block_embedding(x)

        # x = self.encoder_blocks(x)

        cls_token = x[:, 0, :].unsqueeze(1)  # 提取cls_token
        x = x[:, 1:, :]

        batch_size = x.shape[0]
        num_blockies = self.target_depth // self.block_size
        h_w_size = self.img_size // self.block_size
        x = x.permute(0, 2, 1).contiguous().view(batch_size, self.feature, num_blockies, h_w_size, h_w_size)


        cls_token = cls_token.expand(-1, num_blockies * h_w_size * h_w_size, -1)
        cls_token = cls_token.permute(0, 2, 1).contiguous().view(batch_size, self.feature,
                                                                                   num_blockies, h_w_size, h_w_size)
        x = x + cls_token

        x = self.ppm(x)

        x = F.relu(self.decoder2(torch.cat([x, skip_connection4], dim=1)), inplace=True)  # 跳跃连接 skip2

        output1 = self.map1(x)

        x = F.relu(self.decoder3(torch.cat([x, skip_connection3], dim=1)), inplace=True)  # 跳跃连接 skip1

        output2 = self.map2(x)

        x = F.relu(self.decoder4(torch.cat([x, skip_connection2], dim=1)), inplace=True)
        output3 = self.map3(x)


        if self.training is True:
            return output1, output2, output3
        else:
            return output3


if __name__ == "__main__":
    # 设置参数
    in_channel = 1           # 输入通道数
    img_size = 512          # 输入图像大小（假设为立方体）
    block_size = 16         # 块大小
    embed_dim = 256         # 嵌入维度
    num_blockies = 16384       # 块数量
    dropout = 0.1           # dropout 概率
    num_heads = 8           # 自注意力头数
    activation = 'relu'     # 激活函数
    num_encoders = 6        # 编码器层数
    num_classes = 2         # 类别数
    target_depth = 256      # 目标深度
    training = True          # 是否在训练模式

    # 创建模型实例
    model = WFE_Seg(in_channel, img_size, block_size, embed_dim, num_blockies,
                     dropout, num_heads, activation, num_encoders,
                     num_classes, target_depth, training).cuda()

    # 创建随机输入数据，形状为 (batch_size, in_channel, depth, height, width)
    batch_size = 2           # 批次大小



    x = torch.randn(batch_size, in_channel, target_depth, img_size, img_size).cuda()


    # 前向传播
    output = model(x)
    print(output)
