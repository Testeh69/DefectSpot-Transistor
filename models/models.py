from torch import nn


### GEN 5

class Encoder(nn.Module):
  def __init__(self, in_ch = 1, latent_space = 50):
    super().__init__()
    self.f1_conv = nn.Conv2d(in_ch,32,kernel_size=(4,4), stride = 2, padding = 1)
    self.f2_conv = nn.Conv2d(32,32,kernel_size=(4,4), stride = 2, padding = 1)
    self.f3_conv = nn.Conv2d(32,32,kernel_size=(4,4), stride = 2, padding = 1)
    self.f4_conv = nn.Conv2d(32,32,kernel_size=(3,3), stride = 1, padding = 1)
    self.f5_conv = nn.Conv2d(32,64,kernel_size=(4,4), stride = 2, padding = 1)
    self.f6_conv = nn.Conv2d(64,64,kernel_size=(3,3), stride = 1, padding = 1)
    self.f7_conv = nn.Conv2d(64,128,kernel_size=(4,4), stride = 2, padding = 1)
    self.f8_conv = nn.Conv2d(128,64,kernel_size=(3,3), stride = 1, padding = 1)
    self.f9_conv = nn.Conv2d(64,32,kernel_size=(3,3), stride = 1, padding = 1)
    self.f10_conv = nn.Conv2d(32,latent_space,kernel_size=(8,8), stride = 1, padding = 0)
    self.leaky_relu = nn.LeakyReLU(0.2)

  def forward(self, x):
    x = self.leaky_relu(self.f1_conv(x))
    x = self.leaky_relu(self.f2_conv(x))
    x = self.leaky_relu(self.f3_conv(x))
    x = self.leaky_relu(self.f4_conv(x))
    x = self.leaky_relu(self.f5_conv(x))
    x = self.leaky_relu(self.f6_conv(x))
    x = self.leaky_relu(self.f7_conv(x))
    x = self.leaky_relu(self.f8_conv(x))
    x = self.leaky_relu(self.f9_conv(x))
    x = self.f10_conv(x)
    return x
  


class Decoder(nn.Module):
    def __init__(self, in_ch=1, latent_space=50):
        super().__init__()

        self.f1_upconv = nn.ConvTranspose2d(
            latent_space, 32, kernel_size=8, stride=1, padding=0
        )  # 1x1 → 8x8

        self.f2_upconv = nn.ConvTranspose2d(
            32, 64, kernel_size=3, stride=1, padding=1
        )  # 8x8

        self.f3_upconv = nn.ConvTranspose2d(
            64, 128, kernel_size=3, stride=1, padding=1
        )  # 8x8

        self.f4_upconv = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )  # 8x8 → 16x16

        self.f5_upconv = nn.ConvTranspose2d(
            64, 64, kernel_size=3, stride=1, padding=1
        )  # 16x16

        self.f6_upconv = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1
        )  # 16x16 → 32x32

        self.f7_upconv = nn.ConvTranspose2d(
            32, 32, kernel_size=3, stride=1, padding=1
        )  # 32x32

        self.f8_upconv = nn.ConvTranspose2d(
            32, 32, kernel_size=4, stride=2, padding=1
        )  # 32x32 → 64x64

        self.f9_upconv = nn.ConvTranspose2d(
            32, in_ch, kernel_size=4, stride=2, padding=1
        )  # 64x64 → 128x128

        self.f10_upconv = nn.ConvTranspose2d(
            in_ch, in_ch, kernel_size=4, stride=2, padding=1
        )  # 64x64 → 128x128
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky_relu(self.f1_upconv(x))
        x = self.leaky_relu(self.f2_upconv(x))
        x = self.leaky_relu(self.f3_upconv(x))
        x = self.leaky_relu(self.f4_upconv(x))
        x = self.leaky_relu(self.f5_upconv(x))
        x = self.leaky_relu(self.f6_upconv(x))
        x = self.leaky_relu(self.f7_upconv(x))
        x = self.leaky_relu(self.f8_upconv(x))
        x = self.leaky_relu(self.f9_upconv(x))
        x = self.f10_upconv(x)
        return x


class AutoEncodeur(nn.Module):
  def __init__(self, in_ch = 1, latent_space = 50):
    super().__init__()
    self.encoder = Encoder(in_ch, latent_space)
    self.decoder = Decoder(in_ch, latent_space)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x