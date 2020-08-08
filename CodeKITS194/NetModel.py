import torch
import torch.nn as nn

# [conv3d+IN+Leaky Relu+conv3d+IN],
def Conv_IN_LeRU_2s(in_dim, out_dim, kernel_size, stride, padding, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size, stride, padding),
        nn.InstanceNorm3d(out_dim),
        activation,
        nn.Conv3d(out_dim, out_dim, kernel_size, stride, padding),
        nn.InstanceNorm3d(out_dim)
    )

# 跨步卷积
def stride_conv(in_dim, out_dim, kernel_size, stride, padding):
    return nn.Sequential(nn.Conv3d(in_dim, out_dim, kernel_size, stride, padding))

# 残差网络
def ResNet(raw, processed):
    temp = torch.add(raw, processed)
    return temp

# 反卷积
def conv_trans(in_dim, out_dim, kernel_size, stride, padding):
    return nn.ConvTranspose3d(in_dim, out_dim, kernel_size, stride, padding)

def de_conv_in_relu_2s(in_dim, out_dim, kernel_size, stride, padding, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size, stride, padding),
        nn.InstanceNorm3d(out_dim),
        activation,
        nn.Conv3d(out_dim, out_dim, kernel_size=(1, 1, 1), stride=1)
    )

class UNetStage1(nn.Module):
    def __init__(self):
        super(UNetStage1, self).__init__()

        # 按照网络结构，进入网络后马上进行一次卷积
        self.init = nn.Conv3d(1, 30, 3, 1, 1)

        # 第一层
        self.encoder1 = Conv_IN_LeRU_2s(30, 30, 3, 1, 1, nn.LeakyReLU())
        # 加入残差网络1
        self.encoder1_1 = nn.LeakyReLU()

        # 第二层
        self.stride_conv1 = stride_conv(30, 60, (1, 2, 2), 2, 1)
        self.encoder2 = Conv_IN_LeRU_2s(60, 60, 3, 1, 1, nn.LeakyReLU())
        # 加入残差网络2
        self.encoder2_1 = nn.LeakyReLU()

        # 第三层
        self.stride_conv2 = stride_conv(60, 120, (2, 2, 2), 2, 1)
        self.encoder3 = Conv_IN_LeRU_2s(120, 120, 3, 1, 1, nn.LeakyReLU())
        # 加入残差网络3
        self.encoder3_1 = nn.LeakyReLU()

        # 第四层
        self.stride_conv3 = stride_conv(120, 240, (2, 2, 2), 2, 1)
        self.encoder4 = Conv_IN_LeRU_2s(240, 240, 3, 1, 1, nn.LeakyReLU())
        # 加入残差网络4
        self.encoder4_1 = nn.LeakyReLU()

        # 第五层
        self.stride_conv4 = stride_conv(240, 480, (2, 2, 2), 2, 1)
        self.encoder5 = Conv_IN_LeRU_2s(480, 480, 3, 1, 1, nn.LeakyReLU())
        # 加入残差网络5
        self.encoder5_1 = nn.LeakyReLU()

        # 第六层
        self.stride_conv5 = stride_conv(480, 960, (2, 2, 2), 2, 1)
        self.encoder6 = Conv_IN_LeRU_2s(960, 960, 3, 1, 1, nn.LeakyReLU())
        # 加入残差网络6
        self.encoder6_1 = nn.LeakyReLU()

        # 第六层的ResNet结果

        # decode部分
        # Out = (in - 1) * stride - 2 * padding + kernel_size,
        # Link: https://pytorch.org/docs/master/generated/torch.nn.ConvTranspose3d.html
        self.decoder1 = conv_trans(960, 480, kernel_size=2, stride=2, padding=1)
        # 进行cat操作,skip connection
        self.decoder1_1 = nn.Conv3d(960, 480, 1, 1, 1)  # 将通道数减少
        self.decoder1_2 = de_conv_in_relu_2s(480, 480, 3, 1, 1, nn.LeakyReLU())
        # ResNet
        self.decoder1_3 = nn.LeakyReLU()

        self.decoder2 = conv_trans(480, 240, kernel_size=2, stride=2, padding=1)
        # 进行cat操作,skip connection
        self.decoder2_1 = nn.Conv3d(480, 240, 1, 1, 1)  # 将通道数减少
        self.decoder2_2 = de_conv_in_relu_2s(240, 240, 3, 1, 1, nn.LeakyReLU())
        # ResNet
        self.decoder2_3 = nn.LeakyReLU()

        self.decoder3 = conv_trans(240, 120, kernel_size=2, stride=2, padding=1)
        # 进行cat操作,skip connection
        self.decoder3_1 = nn.Conv3d(240, 120, 1, 1, 1)  # 将通道数减少
        self.decoder3_2 = de_conv_in_relu_2s(120, 120, 3, 1, 1, nn.LeakyReLU())
        # ResNet
        self.decoder3_3 = nn.LeakyReLU()

        self.decoder4 = conv_trans(120, 60, kernel_size=2, stride=2, padding=1)
        # 进行cat操作,skip connection
        self.decoder4_1 = nn.Conv3d(120, 60, 1, 1, 1)  # 将通道数减少
        self.decoder4_2 = de_conv_in_relu_2s(60, 60, 3, 1, 1, nn.LeakyReLU())
        # ResNet
        self.decoder4_3 = nn.LeakyReLU()

        self.decoder5 = conv_trans(60, 60, kernel_size=2, stride=2, padding=1)
        # 进行cat操作,skip connection
        self.decoder5_1 = nn.Conv3d(60, 30, 1, 1, 1)  # 将通道数减少
        self.decoder5_2 = de_conv_in_relu_2s(30, 30, 3, 1, 1, nn.LeakyReLU())
        # ResNet
        self.decoder5_3 = nn.LeakyReLU()

        self.end = nn.Conv3d(30, 30, 3, 1, 1)

    def get_features(self, x):
        enc0 = self.init(x)

        enc1 = self.encoder1(enc0)
        res1 = ResNet(enc0, enc1)
        sync1 = self.encoder1_1(res1)

        enc2 = self.stride_conv1(sync1)
        enc2_1 = self.encoder2(enc2)
        res2 = ResNet(enc2, enc2_1)
        sync2 = self.encoder2_1(res2)

        enc3 = self.stride_conv2(sync2)
        enc3_1 = self.encoder3(enc3)
        res3 = ResNet(enc3, enc3_1)
        sync3 = self.encoder3_1(res3)

        enc4 = self.stride_conv3(sync3)
        enc4_1 = self.encoder4(enc4)
        res4 = ResNet(enc4, enc4_1)
        sync4 = self.encoder4_1(res4)

        enc5 = self.stride_conv4(sync4)
        enc5_1 = self.encoder5(enc5)
        res5 = ResNet(enc5, enc5_1)
        sync5 = self.encoder5_1(res5)

        enc6 = self.stride_conv5(sync5)
        enc6_1 = self.encoder6(enc6)
        res6 = ResNet(enc6, enc6_1)
        sync6 = self.encoder6_1(res6)
        return sync6, sync5, sync4, sync3, sync2, sync1

    def upSample(self, enc):
        dec1 = self.decoder1(enc[0])
        skip_con1 = torch.cat((enc[0], dec1), dim=1)
        dec1_2 = self.decoder1_1(skip_con1)
        dec1_3 = self.decoder1_2(dec1_2)
        resnet1 = ResNet(dec1_2, dec1_3)

        dec2 = self.decoder2(resnet1)
        skip_con2 = torch.cat((enc[1], dec2), dim=1)
        dec2_2 = self.decoder2_1(skip_con2)
        dec2_3 = self.decoder2_2(dec2_2)
        resnet2 = ResNet(dec2_2, dec2_3)

        dec3 = self.decoder3(resnet2)
        skip_con3 = torch.cat((enc[2], dec3), dim=1)
        dec3_2 = self.decoder3_1(skip_con3)
        dec3_3 = self.decoder3_2(dec3_2)
        resnet3 = ResNet(dec3_2, dec3_3)

        dec4 = self.decoder4(resnet3)
        skip_con4 = torch.cat((enc[3], dec4), dim=1)
        dec4_2 = self.decoder4_1(skip_con4)
        dec4_3 = self.decoder4_2(dec4_2)
        resnet4 = ResNet(dec4_2, dec4_3)

        dec5 = self.decoder2(resnet4)
        skip_con5 = torch.cat((enc[4], dec5), dim=1)
        dec5_2 = self.decoder5_1(skip_con5)
        dec5_3 = self.decoder5_2(dec5_2)
        resnet5 = ResNet(dec5_2, dec5_3)

        result = self.end(resnet5)
        return result

    def forward(self, x):
        enc = self.get_features(x)
        res = self.upSample(enc)
        return res
