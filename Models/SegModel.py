from Models.ModelUtils import *

"""
Total params: 1,234,285
Total memory: 234.16MB
Total MAdd: 4.68GMAdd
Total Flops: 2.37GFlops
Total MemR+W: 464.5MB
"""
class FasterSegmentation(nn.Module):
    def __init__(self):
        super().__init__()

        # input tensor size: (B, 3, 600, 800)
        self.stem = ConvNormAct(3, 32, 5, 4, 2)    # (B, 32, 150, 200)

        self.s1 = nn.Sequential(
            FasterNetBlock(32, 2),          # (B, 32, 150, 200)
            FasterNetBlock(32, 2),          # (B, 32, 150, 200)
            nn.MaxPool2d(2, 2),             # (B, 32, 75, 100)
            ConvNormAct(32, 64, 1),         # (B, 64, 75, 100)
        )

        self.s2 = nn.Sequential(
            FasterNetBlock(64, 2),          # (B, 64, 75, 100)
            FasterNetBlock(64, 2),          # (B, 64, 75, 100)
            nn.MaxPool2d(2, 2),             # (B, 64, 37, 50)
            ConvNormAct(64, 128, 1),        # (B, 128, 37, 50)
        )

        self.s3 = nn.Sequential(
            FasterNetBlock(128, 2),         # (B, 128, 37, 50)
            FasterNetBlock(128, 2),         # (B, 128, 37, 50)
            nn.MaxPool2d(2, 2),             # (B, 128, 18, 25)
            ConvNormAct(128, 256, 1),       # (B, 256, 18, 25)
        )

        self.neck = nn.Sequential(
            FasterNetBlock(256, 2),         # (B, 256, 18, 25)
            FasterNetBlock(256, 2),         # (B, 256, 18, 25)
            FasterNetBlock(256, 2),         # (B, 256, 18, 25)
        )

        self.upsample_1 = nn.UpsamplingNearest2d(size=(37, 50))  # (B, 256, 18, 25) -> (B, 256, 37, 50)
        self.squeeze_1 = ConvNormAct(256+128, 128, 1)             # (B, 128, 37, 50)

        self.upsample_2 = nn.UpsamplingNearest2d(size=(75, 100))  # (B, 128, 37, 50) -> (B, 256, 75, 100)
        self.squeeze_2 = ConvNormAct(128+64, 64, 1)               # (B, 64, 75, 100)

        self.upsample_3 = nn.UpsamplingNearest2d(size=(150, 200)) # (B, 64, 75, 100) -> (B, 64, 150, 200)
        self.squeeze_3 = ConvNormAct(64+32, 64, 1)                # (B, 64, 150, 200)

        self.head = nn.Sequential(
            FasterNetBlock(64, 2),          # (B, 32, 150, 200)
            nn.Conv2d(64, 13, 1),           # (B, 1, 150, 200)
            nn.UpsamplingBilinear2d(size=(600, 800)),  # (B, 1, 150, 200) -> (B, 1, 600, 800)
            # Multi class segmentation, so we use softmax
            nn.Softmax(dim=1)               # (B, 13, 150, 200)
        )

    def forward(self, x):
        # Encoder
        x = self.stem(x)
        s1 = self.s1(x)
        s2 = self.s2(s1)
        s3 = self.s3(s2)

        # Neck
        s3 = self.neck(s3)

        # Decoder
        up1 = self.upsample_1(s3)
        sq1 = self.squeeze_1(torch.cat([up1, s2], dim=1))

        up2 = self.upsample_2(sq1)
        sq2 = self.squeeze_2(torch.cat([up2, s1], dim=1))

        up3 = self.upsample_3(sq2)
        sq3 = self.squeeze_3(torch.cat([up3, x], dim=1))

        return self.head(sq3)


if __name__ == '__main__':
    from torchstat import stat
    model = FasterSegmentation()
    # stat(model, (3, 600, 800))
    model.cuda()
    inferSpeedTest(model, (3, 600, 800), "cuda")





