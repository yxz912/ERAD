import torch
import torch.nn as nn
from wavelet import DWT_2D,IDWT_2D

dwt = DWT_2D("haar")
iwt = IDWT_2D("haar")

class SegmentationSubNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, use_fp16=False, base_channels=64, out_features=False):
        super(SegmentationSubNetwork, self).__init__()
        base_width = base_channels
        self.encoder_segment = EncoderSegmentation(in_channels, base_width)
        self.decoder_segment = DecoderSegmentation(base_width, out_channels=out_channels)
        self.out_features = out_features

    def forward(self, x):
        b1, b2, b3, b4, b5, b6 = self.encoder_segment(x)
        output_segment = self.decoder_segment(b1, b2, b3, b4, b5, b6)
        if self.out_features:
            return output_segment, b2, b3, b4, b5, b6
        else:
            return output_segment


class EncoderSegmentation(nn.Module):
    def __init__(self, in_channels, base_width):
        super(EncoderSegmentation, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True))
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True))
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        self.block5 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True))

        self.mp5 = nn.Sequential(nn.MaxPool2d(2))
        self.block6 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True))

    def forward(self, x):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp3(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)
        b5 = self.block5(mp4)
        mp5 = self.mp5(b5)
        b6 = self.block6(mp5)
        return b1, b2, b3, b4, b5, b6


class DecoderSegmentation(nn.Module):
    def __init__(self, base_width, out_channels=1):
        super(DecoderSegmentation, self).__init__()

        self.up_b = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                  nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(base_width * 8),
                                  nn.ReLU(inplace=True))

        # cat 96
        self.db_b = nn.Sequential(
            nn.Conv2d(base_width * (8 + 8), base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 4),
                                 nn.ReLU(inplace=True))
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width * (4 + 8), base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 2),
                                 nn.ReLU(inplace=True))
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * (2 + 4), base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width * (2 + 1), base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )

        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)
                                     , nn.Sigmoid()
                                     )

    def forward(self, b1, b2, b3, b4, b5, b6):
        up_b = self.up_b(b6)
        LL6, LH6, HL6, HH6 = dwt(up_b)
        LL5, LH5, HL5, HH5 = dwt(b5)
        LL65 = torch.cat((LL6, LL5),dim=1)
        HH65 = torch.cat((HH6, HH5),dim=1)
        LH65 = torch.cat((LH6, LH5),dim=1)
        HL65 = torch.cat((HL6, HL5),dim=1)
        cat_b = iwt(LL65,LH65,HL65,HH65)
        db_b = self.db_b(cat_b)

        up1 = self.up1(db_b)
        LL1, LH1, HL1, HH1 = dwt(up1)
        LL4, LH4, HL4, HH4 = dwt(b4)
        LL41 = torch.cat((LL4 , LL1),dim=1)
        HH41 = torch.cat((HH4, HH1),dim=1)
        LH41 = torch.cat((LH4, LH1),dim=1)
        HL41 = torch.cat((HL4, HL1),dim=1)
        cat1 = iwt(LL41, LH41, HL41, HH41)
        db1 = self.db1(cat1)

        up2 = self.up2(db1)
        LL2, LH2, HL2, HH2 = dwt(up2)
        LL3, LH3, HL3, HH3 = dwt(b3)
        LL32 = torch.cat((LL3, LL2),dim=1)
        HH32 = torch.cat((HH3, HH2),dim=1)
        LH32 = torch.cat((LH3, LH2),dim=1)
        HL32 = torch.cat((HL3, HL2),dim=1)
        cat2 = iwt(LL32, LH32, HL32, HH32)
        db2 = self.db2(cat2)

        up3 = self.up3(db2)
        LL3, LH3, HL3, HH3 = dwt(up3)
        LL2, LH2, HL2, HH2 = dwt(b2)
        LL23 = torch.cat((LL3, LL2),dim=1)
        HH23 = torch.cat((HH3, HH2),dim=1)
        LH23 = torch.cat((LH3, LH2),dim=1)
        HL23 = torch.cat((HL3, HL2),dim=1)
        cat3 = iwt(LL23, LH23, HL23, HH23)
        db3 = self.db3(cat3)

        up4 = self.up4(db3)
        LL4, LH4, HL4, HH4 = dwt(up4)
        LL1, LH1, HL1, HH1 = dwt(b1)
        LL14 = torch.cat((LL4, LL1),dim=1)
        HH14 = torch.cat((HH4, HH1),dim=1)
        LH14 = torch.cat((LH4, LH1),dim=1)
        HL14 = torch.cat((HL4, HL1),dim=1)
        cat4 = iwt(LL14, LH14, HL14, HH14)
        db4 = self.db4(cat4)

        out = self.fin_out(db4)
        return out

