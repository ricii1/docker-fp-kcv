from torch import nn, optim
import torch
from torch.nn import functional as F
from typing import Any, Callable, Optional
import math

class WSLinear(nn.Module):
    '''
    Weighted scale linear for equalized learning rate.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
    '''

    def __init__(self, in_features: int, out_features: int) -> None:
        super(WSLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.linear = nn.Linear(self.in_features, self.out_features)
        self.scale = (2 / self.in_features) ** 0.5
        self.bias = self.linear.bias
        self.linear.bias = None

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x * self.scale) + self.bias
    
class WSConv2d(nn.Module):
    """
    Weight-scaled Conv2d layer for equalized learning rate.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolving kernel. Default: 3.
        stride (int, optional): Stride of the convolution. Default: 1.
        padding (int, optional): Padding added to all sides of the input. Default: 1.
        gain (float, optional): Gain factor for weight initialization. Default: 2.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * kernel_size ** 2)) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None  # Remove bias to apply it after scaling

        # Initialize weights
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)

class Mapping(nn.Module):
    '''
    Mapping network.

    Args:
        features (int): Number of features in the input and output.
        num_layers (int): Number of layers in the feed forward network.
        num_styles (int): Number of styles to generate.
    '''

    def __init__(
        self,
        features: int,
        num_styles: int,
        num_layers: int = 8,
    ) -> None:
        super(Mapping, self).__init__()
        self.features = features
        self.num_layers = num_layers
        self.num_styles = num_styles

        layers = []
        for _ in range(self.num_layers):
            layers.append(WSLinear(self.features, self.features))
            layers.append(nn.LeakyReLU(0.2))

        self.fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x (torch.Tensor): Input tensor of shape (b, l).

        Returns:
            torch.Tensor: Output tensor with the same shape as input.
        '''

        x = self.fc(x) # (b, l)
        return x

class AdaIN(nn.Module):
    '''
    Adaptive Instance Normalization (AdaIN)
    AdaIN(x_i, y) = y_s,i * (x_i - mean(x_i)) / std(x_i) + y_b,i

    Args:
        eps (float, optional): Small value to avoid division by zero. Default value is 0.00001.
    '''

    def __init__(self, eps: float= 1e-5) -> None:
        super(AdaIN, self).__init__()
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor
    ) -> torch.Tensor:
        '''
        Args:
            x (torch.Tensor): Input tensor of shape (b, c, h, w).
            scale (torch.Tensor): Scale tensor of shape (b, c).
            shift (torch.Tensor): Shift tensor of shape (b, c).

        Returns:
            torch.Tensor: Output tensor of shape (b, c, h, w).
        '''

        b, c, *_ = x.shape

        mean = x.mean(dim=(2, 3), keepdim=True) # (b, c, 1, 1)
        std = x.std(dim=(2, 3), keepdim=True) # (b, c, 1, 1)
        x_norm = (x - mean) / (std ** 2 + self.eps) ** .5

        scale = scale.view(b, c, 1, 1) # (b, c, 1, 1)
        shift = scale.view(b, c, 1, 1) # (b, c, 1, 1)
        outputs = scale * x_norm + shift # (b, c, h, w)

        return outputs

class SynthesisLayer(nn.Module):
    '''
    Synthesis network layer which consist of:
    - Conv2d.
    - AdaIN.
    - Affine transformation.
    - Noise injection.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        latent_features (int): The number of latent features.
        use_conv (bool, optional): Whether to use convolution or not. Default value is True.
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_features: int,
        use_conv: bool = True
    ) -> None:
        super(SynthesisLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_features = latent_features
        self.use_conv = use_conv

        self.conv = nn.Sequential(
            WSConv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        ) if self.use_conv else nn.Identity()
        self.norm = AdaIN()
        self.scale_transform = WSLinear(self.latent_features, self.out_channels)
        self.shift_transform = WSLinear(self.latent_features, self.out_channels)
        self.noise_factor = nn.Parameter(torch.zeros(1, self.out_channels, 1, 1))

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.ones_(self.scale_transform.bias)

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        '''
        Args:
            x (torch.Tensor): Input tensor of shape (b, c, h, w).
            w (torch.Tensor): Latent space vector of shape (b, l).
            noise (torch.Tensor, optional): Noise tensor of shape (b, 1, h, w). Default value is None.

        Returns:
            torch.Tensor: Output tensor of shape (b, c, h, w).
        '''

        b, _, h, w_ = x.shape
        x = self.conv(x) # (b, o_c, h, w)
        if noise is None:
            noise = torch.randn(b, 1, h, w_, device=x.device) # (b, 1, h, w)
        x += self.noise_factor * noise # (b, o_c, h, w)
        y_s = self.scale_transform(w) # (b, o_c)
        y_b = self.shift_transform(w) # (b, o_c)
        x = self.norm(x, y_s, y_b) # (b, i_c, h, w)

        return x
    

class SynthesisBlock(nn.Module):
    '''
    Synthesis network block which consist of:
    - Optional upsampling.
    - 2 Synthesis Layers.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        latent_features (int): The number of latent features.
        use_conv (bool, optional): Whether to use convolution or not. Default value is True.
        upsample (bool, optional): Whether to use upsampling or not. Default value is True.
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_features: int,
        *,
        use_conv: bool = True,
        upsample: bool = True
     ) -> None:
        super(SynthesisBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_features = latent_features
        self.use_conv = use_conv
        self.upsample = upsample

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear') if self.upsample else nn.Identity()
        self.layers = nn.ModuleList([
            SynthesisLayer(self.in_channels, self.in_channels, self.latent_features, use_conv=self.use_conv),
            SynthesisLayer(self.in_channels, self.out_channels, self.latent_features)
        ])

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x (torch.Tensor): Input tensor of shape (b, c, h, w).
            w (torch.Tensor): Latent vector of shape (b, l).

        Returns:
            torch.Tensor: Output tensor of shape (b, c, h, w) if not upsample else (b, c, 2h, 2w).
        '''

        x = self.upsample(x) # (b, c, h, w) if not upsample else (b, c, 2h, 2w)

        for layer in self.layers:
            x = layer(x, w) # (b, c, h, w) if not upsample else (b, c, 2h, 2w)

        return x

class Synthesis(nn.Module):
    '''
    Synthesis network which consist of:
    - Constant tensor.
    - Synthesis blocks.
    - ToRGB convolutions.

    Args:
        resolution (int): The resolution of the image.
        const_channels (int): The number of channels in the constant tensor. Default value is 512.
    '''

    def __init__(self, resolution: int, const_channels: int = 512) -> None:
        super(Synthesis, self).__init__()
        self.const_channels = const_channels
        self.resolution = resolution

        self.resolution_levels = int(math.log2(resolution) - 1)

        self.constant = nn.Parameter(torch.ones(1, self.const_channels, 4, 4)) # (c, 4, 4)

        in_channels = self.const_channels
        blocks = [ SynthesisBlock(in_channels, in_channels, self.const_channels, use_conv=False, upsample=False) ]
        to_rgb = [ WSConv2d(in_channels, 3, kernel_size=1, padding=0) ]

        for _ in range(self.resolution_levels - 1):
            blocks.append(SynthesisBlock(in_channels, in_channels // 2, self.const_channels))
            to_rgb.append(WSConv2d(in_channels // 2, 3, kernel_size=1, padding=0))
            in_channels //= 2

        self.blocks = nn.ModuleList(blocks)
        self.to_rgb = nn.ModuleList(to_rgb)

    def forward(self, w: torch.Tensor, alpha: float, steps: int) -> torch.Tensor:
        '''
        Args:
            w (torch.Tensor): Latent space vector of shape (b, l).
            alpha (float): Fade in alpha value.
            steps (int): The number of steps starting from 0.

        Returns:
            torch.Tensor: Output tensor of shape (b, 3, h, w).
        '''

        b = w.size(0)
        x = self.constant.expand(b, -1, -1, -1).clone() # (b, c, h, w)

        if steps == 0:
            x = self.blocks[0](x, w) # (b, c, h, w)
            x = self.to_rgb[0](x) # (b, c, h, w)
            return x

        for i in range(steps):
            x = self.blocks[i](x, w) # (b, c, h/2, w/2)

        old_rgb = self.to_rgb[steps - 1](x) # (b, 3, h/2, w/2)

        x = self.blocks[steps](x, w) # (b, 3, h, w)
        new_rgb = self.to_rgb[steps](x) # (b, 3, h, w)
        old_rgb = F.interpolate(old_rgb, scale_factor=2, mode='bilinear', align_corners=False) # (b, 3, h, w)

        x = (1 - alpha) * old_rgb + alpha * new_rgb # (b, 3, h, w)

        return x

class StyleGAN(nn.Module):
    '''
    StyleGAN implementation.

    Args:
        num_features (int): The number of features in the latent space vector.
        resolution (int): The resolution of the image.
        num_blocks (int, optional): The number of blocks in the synthesis network. Default value is 10.
    '''

    def __init__(self, num_features: int, resolution: int, num_blocks: int = 10):
        super(StyleGAN, self).__init__()
        self.num_features = num_features
        self.resolution = resolution
        self.num_blocks = num_blocks

        self.mapping = Mapping(self.num_features, self.num_blocks)
        self.synthesis = Synthesis(self.resolution, self.num_features)

    def forward(self, x: torch.Tensor, alpha: float, steps: int) -> torch.Tensor:
        '''
        Args:
            x (torch.Tensor): Random input tensor of shape (b, l).
            alpha (float): Fade in alpha value.
            steps (int): The number of steps starting from 0.

        Returns:
            torch.Tensor: Output tensor of shape (b, c, h, w).
        '''

        w = self.mapping(x) # (b, l)
        outputs = self.synthesis(w, alpha, steps) # (b, c, h, w)

        return outputs