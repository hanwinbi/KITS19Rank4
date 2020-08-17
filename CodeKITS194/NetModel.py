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

# 三线性插值
def tri_inter(input, size, mode):
    return nn.functional.interpolate(input=input, size=size, mode=mode)

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
        # padding的计算：https://pytorch.org/docs/master/generated/torch.nn.Conv3d.html
        self.stride_conv1 = stride_conv(30, 60, (3, 3, 3), (1, 2, 2), 1)
        self.encoder2 = Conv_IN_LeRU_2s(60, 60, 3, 1, 1, nn.LeakyReLU())
        # 加入残差网络2
        self.encoder2_1 = nn.LeakyReLU()

        # 第三层
        self.stride_conv2 = stride_conv(60, 120, 3, 2, 1)
        self.encoder3 = Conv_IN_LeRU_2s(120, 120, 3, 1, 1, nn.LeakyReLU())
        # 加入残差网络3
        self.encoder3_1 = nn.LeakyReLU()

        # 第四层
        self.stride_conv3 = stride_conv(120, 240, 3, 2, 1)
        self.encoder4 = Conv_IN_LeRU_2s(240, 240, 3, 1, 1, nn.LeakyReLU())
        # 加入残差网络4
        self.encoder4_1 = nn.LeakyReLU()

        # 第五层
        self.stride_conv4 = stride_conv(240, 480, 3, 2, 1)
        self.encoder5 = Conv_IN_LeRU_2s(480, 480, 3, 1, 1, nn.LeakyReLU())
        # 加入残差网络5
        self.encoder5_1 = nn.LeakyReLU()

        # 第六层
        self.stride_conv5 = stride_conv(480, 960, 3, 2, 1)
        self.encoder6 = Conv_IN_LeRU_2s(960, 960, 3, 1, 1, nn.LeakyReLU())
        # 加入残差网络6
        self.encoder6_1 = nn.LeakyReLU()

        # 第六层的ResNet结果

        # decode部分
        # Out = (in - 1) * stride - 2 * padding + kernel_size,
        # Link: https://pytorch.org/docs/master/generated/torch.nn.ConvTranspose3d.html
        self.decoder1 = conv_trans(960, 480, kernel_size=2, stride=2, padding=0)
        # 进行cat操作,skip connection
        self.decoder1_1 = nn.Conv3d(960, 480, 1, 1, 0)  # 将通道数减少
        self.decoder1_2 = de_conv_in_relu_2s(480, 480, 3, 1, 1, nn.LeakyReLU())
        # ResNet
        self.decoder1_3 = nn.LeakyReLU()

        self.decoder2 = conv_trans(480, 240, kernel_size=2, stride=2, padding=0)
        # 进行cat操作,skip connection
        self.decoder2_1 = nn.Conv3d(480, 240, 1, 1, 0)  # 将通道数减少
        self.decoder2_2 = de_conv_in_relu_2s(240, 240, 3, 1, 1, nn.LeakyReLU())
        # ResNet
        self.decoder2_3 = nn.LeakyReLU()

        self.decoder3 = conv_trans(240, 120, kernel_size=2, stride=2, padding=0)
        # 进行cat操作,skip connection
        self.decoder3_1 = nn.Conv3d(240, 120, 1, 1, 0)  # 将通道数减少
        self.decoder3_2 = de_conv_in_relu_2s(120, 120, 3, 1, 1, nn.LeakyReLU())
        # ResNet
        self.decoder3_3 = nn.LeakyReLU()

        self.decoder4 = conv_trans(120, 60, kernel_size=2, stride=2, padding=0)
        self.decoder4_1 = nn.Conv3d(120, 60, 1, 1, 0)  # 将通道数减少
        # 进行cat操作,skip connection
        self.decoder4_2 = de_conv_in_relu_2s(60, 60, 3, 1, 1, nn.LeakyReLU())
        # ResNet
        self.decoder4_3 = nn.LeakyReLU()

        self.decoder5 = conv_trans(60, 30, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)
        # 进行cat操作,skip connection
        self.decoder5_1 = nn.Conv3d(60, 30, 1, 1, 0)  # 将通道数减少
        self.decoder5_2 = de_conv_in_relu_2s(30, 30, 3, 1, 1, nn.LeakyReLU())
        # ResNet
        self.decoder5_3 = nn.LeakyReLU()

        self.end = nn.Conv3d(30, 3, 3, 1, 1)

    # 下采样
    '''
    方法名：内部处理
    encoderNum：conv->ins_norm->relu  conv->ins_norm
    encoderNum_1: relu
    stride_convNum: 跨步卷积 kernal：3*3*3 第一层stride：1*2*2 其他层：2*2*2 padding：1
    '''
    def get_features(self, x):
        enc0 = self.init(x)  # [N C 32 160 160] -> [N 30 32 160 160] 放入网络之前进行一次3d卷积，通道数变为30

        enc1 = self.encoder1(enc0)  # [N 30 32 160 160]
        res1 = ResNet(enc0, enc1)  # 残差块
        sync1 = self.encoder1_1(res1)
        print('sync1', sync1.shape)

        enc2 = self.stride_conv1(sync1)  # [N 30 32 160 160] -> [N 60 32 80 80]
        enc2_1 = self.encoder2(enc2)
        res2 = ResNet(enc2, enc2_1)
        sync2 = self.encoder2_1(res2)
        print('sync2', sync2.shape)

        enc3 = self.stride_conv2(sync2)  # [N 60 32 80 80] -> [N 120 16 40 40]
        enc3_1 = self.encoder3(enc3)
        res3 = ResNet(enc3, enc3_1)
        sync3 = self.encoder3_1(res3)
        print('sync3', sync3.shape)

        enc4 = self.stride_conv3(sync3)  # [N 120 16 40 40] -> [N 240 8 20 20]
        enc4_1 = self.encoder4(enc4)
        res4 = ResNet(enc4, enc4_1)
        sync4 = self.encoder4_1(res4)
        print('sync4', sync4.shape)

        enc5 = self.stride_conv4(sync4)  # [N 240 8 20 20] -> [N 480 4 10 10]
        enc5_1 = self.encoder5(enc5)
        res5 = ResNet(enc5, enc5_1)
        sync5 = self.encoder5_1(res5)
        print('sync5', sync5.shape)

        enc6 = self.stride_conv5(sync5)  # [N 480 4 10 10] -> [N 960 2 5 5]
        enc6_1 = self.encoder6(enc6)
        res6 = ResNet(enc6, enc6_1)
        sync6 = self.encoder6_1(res6)
        print('sync6', sync6.shape)
        return sync6, sync5, sync4, sync3, sync2, sync1

    # 上采样
    '''
    方法名：解释
    decoderNum: 反卷积
    skip_conNum: 跨层连接
    decoderNum_1: 将拼接后结果通道数减半
    decoderNum_2: conv->ins_norm->relu  conv->ins_norm
    decoderNum_3: relu
    tri_inter: 三线性插值
    '''
    def upSample(self, enc):
        dec1 = self.decoder1(enc[0])  # [N 960 2 5 5] -> [N 480 4 10 10]最后一层的结果直接进行上采样
        print("enc[0]:{0},dec1:{1}".format(enc[1].shape, dec1.shape))
        skip_con1 = torch.cat((enc[1], dec1), dim=1)  # [N 480 4 10 10] -> [N 960 4 10 10]
        dec1_1 = self.decoder1_1(skip_con1)  # [N 960 4 10 10] -> [N 480 4 10 10]
        dec1_2 = self.decoder1_2(dec1_1)  # [N 480 4 10 10]
        resnet1 = ResNet(dec1_1, dec1_2)
        resnet1 = self.decoder1_3(resnet1)
        result1 = tri_inter(resnet1, (80, 160, 160), 'trilinear')

        dec2 = self.decoder2(resnet1)  # [N 480 4 10 10] -> [N 240 8 20 20]
        print("enc[1]:{0},dec2:{1}".format(enc[2].shape, dec2.shape))
        skip_con2 = torch.cat((enc[2], dec2), dim=1)  # [N 240 8 20 20] -> [N 480 8 20 20]
        dec2_1 = self.decoder2_1(skip_con2)
        dec2_2 = self.decoder2_2(dec2_1)
        resnet2 = ResNet(dec2_1, dec2_2)
        resnet2 = self.decoder2_3(resnet2)
        result2 = tri_inter(resnet2, (80, 160, 160), 'trilinear')

        dec3 = self.decoder3(resnet2)  # [N 480 8 20 20] -> [N 240 16 40 40]
        print("enc3:{0},dec3:{1}".format(enc[3].shape, dec3.shape))
        skip_con3 = torch.cat((enc[3], dec3), dim=1)  # [N 480 16 40 40] -> [N 240 16 40 40]
        dec3_1 = self.decoder3_1(skip_con3)
        dec3_2 = self.decoder3_2(dec3_1)
        resnet3 = ResNet(dec3_2, dec3_2)
        resnet3 = self.decoder3_3(resnet3)
        result3 = tri_inter(resnet3, (80, 160, 160), 'trilinear')

        dec4 = self.decoder4(resnet3)  # [N 240 16 40 40] -> [N 120 32 80 80]
        print("enc[4]:{0},dec4:{1}".format(enc[4].shape, dec4.shape))
        skip_con4 = torch.cat((enc[4], dec4), dim=1)  # [N 120 32 80 80] -> [N 60 32 80 80]
        dec4_1 = self.decoder4_1(skip_con4)
        dec4_2 = self.decoder4_2(dec4_1)
        resnet4 = ResNet(dec4_2, dec4_2)
        resnet4 = self.decoder4_3(resnet4)
        result4 = tri_inter(resnet4, (80, 160, 160), 'trilinear')

        dec5 = self.decoder5(resnet4)  # [N 60 32 80 80] -> [N 30 32 160 160]
        print("enc[5]:{0},dec5:{1}".format(enc[5].shape, dec5.shape))
        skip_con5 = torch.cat((enc[5], dec5), dim=1)  # [N 60 32 160 160] -> [N 30 32 160 160]
        dec5_1 = self.decoder5_1(skip_con5)
        dec5_2 = self.decoder5_2(dec5_1)
        resnet5 = ResNet(dec5_2, dec5_2)
        resnet5 = self.decoder5_3(resnet5)

        result5 = self.end(resnet5)  # 最后一层的输出
        print("result5:", result5.shape)
        return result5

    def forward(self, x):
        enc = self.get_features(x)
        res = self.upSample(enc)
        return res
