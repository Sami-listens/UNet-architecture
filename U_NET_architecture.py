import torch
import torch.nn as nn

class U_NET(nn.Module):
    def __init__(self):
        super(U_NET, self).__init__()

        self.DC1 = self.repeated_conv(in_c=1, out_c=64)
        self.DC2 = self.repeated_conv(in_c=64, out_c=128)
        self.DC3 = self.repeated_conv(in_c=128, out_c=256)
        self.DC4 = self.repeated_conv(in_c=256, out_c=512)
        self.DC5 = self.repeated_conv(in_c=512, out_c=1024)
        
        self.MP_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.UTC1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.UC1 = self.repeated_conv(in_c=1024, out_c=512)
        self.UTC2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.UC2 = self.repeated_conv(in_c=512, out_c=256)
        self.UTC3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.UC3 = self.repeated_conv(in_c=256, out_c=128)
        self.UTC4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.UC4 = self.repeated_conv(in_c=128, out_c=64)

        self.out = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)

    def repeated_conv(self, in_c, out_c):
        double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3),
            nn.ReLU(inplace=True)
        )
        return double_conv

    def cropping(self, input, target):
        size_in = input.shape[2]
        size_target = target.shape[2]
        margin = (size_in - size_target) // 2
        return input[:, :, margin:size_in-margin, margin:size_in-margin] # center cropping
        

    def forward(self, x):
        # Encoder / Contraction(Left-Path)
        x1 = self.DC1(x)
        x2 = self.MP_2x2(x1)
        x3 = self.DC2(x2)
        x4 = self.MP_2x2(x3)
        x5 = self.DC3(x4)
        x6 = self.MP_2x2(x5)
        x7 = self.DC4(x6)
        x8 = self.MP_2x2(x7)
        x9 = self.DC5(x8)
        print(x9.shape)


        # Decoder / Expension(Right-Path)
        x = self.UTC1(x9)
        x_cropped = self.cropping(x7, x)
        x = self.UC1(torch.cat([x_cropped, x],1))

        x = self.UTC2(x)
        x_cropped = self.cropping(x5, x)
        x = self.UC2(torch.cat([x_cropped, x],1))
        
        x = self.UTC3(x)
        x_cropped = self.cropping(x3, x)
        x = self.UC3(torch.cat([x_cropped, x],1))

        x = self.UTC4(x)
        x_cropped = self.cropping(x1, x)
        x = self.UC4(torch.cat([x_cropped, x],1))
        
        x = self.out(x)
        print(x.shape)
        
        return x


input = torch.rand(1, 1, 572, 572)
model = U_NET()
output = model(input)
print(output)