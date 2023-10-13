import torch
import torch.nn as nn

class UNetWithSpaceToDepth(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetWithSpaceToDepth, self).__init__()

        # Encoder blocks
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64 * 4, 128)
        self.enc3 = self._block(128 * 4, 256)
        self.enc4 = self._block(256 * 4, 512)

        # Decoder blocks
        self.dec1 = self._block(384, 256)
        self.dec2 = self._block(192, 128)
        self.dec3 = self._block(96, 64)

        # Final output layer
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

        # Space-to-depth and depth-to-space modules
        self.space_to_depth = nn.PixelUnshuffle(downscale_factor=2)
        self.depth_to_space = nn.PixelShuffle(upscale_factor=2)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.space_to_depth(e1))
        e3 = self.enc3(self.space_to_depth(e2))
        e4 = self.enc4(self.space_to_depth(e3))

        # Decoder with skip connections
        d1 = self.dec1(torch.cat([self.depth_to_space(e4), e3], 1))
        d2 = self.dec2(torch.cat([self.depth_to_space(d1), e2], 1))
        d3 = self.dec3(torch.cat([self.depth_to_space(d2), e1], 1))
        
        return self.out(d3)


sample = torch.randn((4,1,512,512))
# Testing the UNetWithSpaceToDepth module
unet_std = UNetWithSpaceToDepth(in_channels=1, out_channels=1)
print(unet_std(sample).shape)
