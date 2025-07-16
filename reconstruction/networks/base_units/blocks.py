import torch
import torch.nn as nn
from networks.base_units.conv_layers import down_conv, up_conv, conv3x3
from networks.base_units.memory_module import MemModule

import csv


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, num_layers, downsample=False, upsample=False, last_layer=False):
        super(BasicBlock, self).__init__()
        assert not (downsample and upsample)
        layers = []
        if downsample:
            layers.append(down_conv(inplanes, planes))
        elif upsample:
            layers.append(up_conv(inplanes, planes))
        else:
            layers.append(conv3x3(inplanes, planes))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))

        # Deeper block
        if upsample:
            for _ in range(1, num_layers):
                add_layer = [conv3x3(inplanes, inplanes),
                             nn.BatchNorm2d(inplanes),
                             nn.ReLU(inplace=True)]

                layers = add_layer + layers
        else:
            for _ in range(1, num_layers):
                add_layer = [conv3x3(planes, planes),
                             nn.BatchNorm2d(planes),
                             nn.ReLU(inplace=True)]

                layers = layers + add_layer

        if last_layer:
            layers = layers[:-2]  # remove the BN and ReLU for the output layer.

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, num_layers, downsample=False, upsample=False, last_layer=False):
        super(ResBlock, self).__init__()
        assert not (downsample and upsample)
        self.last_layer = last_layer
        self.relu = nn.ReLU(inplace=True)

        layers = []
        if downsample:
            layers.append(down_conv(inplanes, planes))
            self.skip = nn.Sequential(
                down_conv(inplanes, planes),
                nn.BatchNorm2d(planes)
            )
        elif upsample:
            layers.append(up_conv(inplanes, planes))
            self.skip = nn.Sequential(
                up_conv(inplanes, planes),
                nn.BatchNorm2d(planes)
            )
        else:
            layers.append(conv3x3(inplanes, planes))
            self.skip = nn.Identity()
        layers.append(nn.BatchNorm2d(planes))

        # Deeper block
        if upsample:
            for _ in range(1, num_layers):
                add_layer = [conv3x3(inplanes, inplanes),
                             nn.BatchNorm2d(inplanes),
                             nn.ReLU(inplace=True)]

                layers = add_layer + layers
        else:
            for _ in range(1, num_layers):
                add_layer = [nn.ReLU(inplace=True),
                             conv3x3(planes, planes),
                             nn.BatchNorm2d(planes)]

                layers = layers + add_layer

        if last_layer:  # remove the BN for the output layer.
            layers = layers[:-1]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        identity = x

        out = self.model(x)

        identity = self.skip(identity)
        out += identity

        if not self.last_layer:  # remove the relu for the output layer.
            out = self.relu(out)

        return out


class BottleNeck(nn.Module):
    def __init__(self, in_planes, feature_size, mid_num=2048, latent_size=16, is_laplace=False
    ,epsilon=1.0, sensitivity_path = None):
        super(BottleNeck, self).__init__()
        self.in_planes = in_planes
        self.feature_size = feature_size
        self.linear_enc = nn.Sequential(
            nn.Linear(in_planes * feature_size * feature_size, mid_num),
            nn.BatchNorm1d(mid_num),
            nn.ReLU(True),
            nn.Linear(mid_num, latent_size))

        self.linear_dec = nn.Sequential(
            nn.Linear(latent_size, mid_num),
            nn.BatchNorm1d(mid_num),
            nn.ReLU(True),
            nn.Linear(mid_num, in_planes * feature_size * feature_size))
        
        self.is_laplace = is_laplace
        self.epsilon = epsilon
        self.sensitivity_path = sensitivity_path

    def laplace_noise(self, z, epsilon, max_list, min_list, sensitivity_list):
        B, latent_size = z.shape
        device = z.device  # zのデバイスを取得
        
        # 一様分布 [-0.5, 0.5] の乱数（バッチサイズ×次元数）
        U = torch.rand((B, latent_size), device=device) - 0.5
        noise = torch.zeros_like(U)
        
        # 各次元ごとに異なるスケールでノイズを生成
        for i in range(latent_size):
            scale = sensitivity_list[i] / epsilon
            u_i = U[:, i]
            noise[:, i] = -scale * torch.sign(u_i) * torch.log1p(-2 * torch.abs(u_i))
        
        # ノイズを加える
        z_noised = z + noise
        
        # 各次元ごとにクリップ
        for i in range(latent_size):
            z_noised[:, i] = torch.clamp(z_noised[:, i], min_list[i], max_list[i])
        
        return z_noised

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.linear_enc(x)
        
        if self.is_laplace:
            # laplace mechanism
            max_list = []
            min_list = []
            abs_list = []
            with open(self.sensitivity_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    max_val = float(row[0])
                    min_val = float(row[1])
                    abs_val = float(row[2])

                    max_list.append(max_val)
                    min_list.append(min_val)
                    abs_list.append(abs_val)

            z = self.laplace_noise(z,self.epsilon, max_list, min_list, abs_list)

        out = self.linear_dec(z)

        out = out.view(x.size(0), self.in_planes, self.feature_size, self.feature_size)

        return {'out': out, 'z': z}


class SpatialBottleNeck(nn.Module):
    def __init__(self, in_planes, feature_size, mid_num=2048, latent_size=16):
        super(SpatialBottleNeck, self).__init__()
        self.in_planes = in_planes
        self.feature_size = feature_size
        self.linear_enc = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=mid_num, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_num),
            nn.ReLU(True),
            nn.Conv2d(in_channels=mid_num, out_channels=latent_size, kernel_size=1, stride=1, padding=0, bias=False))

        self.linear_dec = nn.Sequential(
            nn.Conv2d(in_channels=latent_size, out_channels=mid_num, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_num),
            nn.ReLU(True),
            nn.Conv2d(in_channels=mid_num, out_channels=in_planes, kernel_size=1, stride=1, padding=0, bias=False),)

    def forward(self, x):
        z = self.linear_enc(x)
        out = self.linear_dec(z)

        return {'out': out, 'z': z}


class MemBottleNeck(BottleNeck):
    def __init__(self, in_planes, feature_size, mid_num=2048, latent_size=16, mem_size=25, shrink_thres=0.0025):
        super(MemBottleNeck, self).__init__(in_planes, feature_size, mid_num, latent_size)
        self.memory_module = MemModule(mem_dim=mem_size, fea_dim=latent_size, shrink_thres=shrink_thres)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.linear_enc(x)

        mem_out = self.memory_module(z)
        z_hat, att = mem_out['output'], mem_out['att']

        out = self.linear_dec(z_hat)
        out = out.view(x.size(0), self.in_planes, self.feature_size, self.feature_size)

        return {'out': out, 'att': att, 'z': z, 'z_hat': z_hat}


class VaeBottleNeck(BottleNeck):
    def __init__(self, in_planes, feature_size, mid_num=2048, latent_size=16):
        super(VaeBottleNeck, self).__init__(in_planes, feature_size, mid_num, latent_size)
        self.linear_enc = nn.Sequential(
            nn.Linear(in_planes * feature_size * feature_size, mid_num),
            nn.BatchNorm1d(mid_num),
            nn.ReLU(True),
            nn.Linear(mid_num, 2 * latent_size))

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.linear_enc(x)

        mu, log_var = z.chunk(2, dim=1)
        z_hat = self.reparameterize(mu, log_var)

        out = self.linear_dec(z_hat)
        out = out.view(x.size(0), self.in_planes, self.feature_size, self.feature_size)
        return {'out': out, 'mu': mu, 'log_var': log_var}