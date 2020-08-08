import json
import os
import time
import torch
import cv2
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Loss import dice_coeff
from GenerateFilePath import trainData, validationData
from PrepareData import load_case_data, DataSets
import NetModel

import warnings
warnings.filterwarnings("ignore")

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

model = NetModel.UNetStage1()
model = model.to(device)

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        loss = - (1 - pt) ** self.gamma * target * torch.log(pt) - \
            pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.alpha:
            loss = loss * self.alpha
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
criterion_focal = BCEFocalLoss().to(device)

criterion = nn.BCELoss().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(Epoches,mpth):
    start_epoch = 0
    pthList = os.listdir(mpth)
    print(pthList)
    if not pthList:
        print('starting train:')
    else:
        print('Continue training:')
        pth = pthList[-1]
        checkPoint = torch.load(mpth + pth)
        model.load_state_dict(checkPoint['model'])
        optimizer.load_state_dict(checkPoint['optimizer'])
        start_epoch = checkPoint['epoch'] + 1
    numEpoches = Epoches
    for epoch in range(start_epoch, numEpoches):
        t = time.time()
        print('------------------------')
        print('this is {} epoch.'.format(epoch))
        t = time.localtime(t)
        t = time.strftime("%Y-%m-%d %H:%M:%S", t)
        print(t)

        model.train()

        with open(trainData, 'r') as load_f:
            load_dict = json.load(load_f)
        data = []
        for i in range(len(load_dict)):
            data.append(load_dict[i]['path'])
        print('in the train ', data)

        for it,item in enumerate(data):
            print('item', item)
            print('train data', data)
            oneCaseLoss = 0
            oneCase = load_case_data(trainData, it)
            examOneCase = DataSets(oneCase)
            dataLoader = DataLoader(examOneCase, batch_size=2, shuffle=False, num_workers=4)
            for idx,(x,yy) in enumerate(dataLoader):
                with torch.autograd.set_detect_anomaly(True):
                    x = Variable(x).to(device)
                    y = Variable(yy[0]).to(device)

                    x = x.unsqueeze(1)
                    print(x.shape)

                    output = model(x)
                    output = torch.sigmoid(output)

                    loss = criterion(output, y)
                    focal_loss = criterion_focal(output, y)
                    loss = loss + focal_loss

                    iterLoss = loss.item()
                    oneCaseLoss += iterLoss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            print('one case loss:',oneCaseLoss)

        if epoch % 3 == 0:
            ###保存模型###
            savepPth = mpth + 'v10_NotAll_' + str('%.2d' % epoch) + '.pth'
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, savepPth)
        if epoch % 3 == 0:
            ## 验证集 给出验证效果
            model.eval()
            dice_back = 0
            dice_kidney = 0
            dice_tumor = 0
            for it, item in enumerate(validationData):
                oneCaseValLoss = 0
                backgroundDice = 0
                kidneyDice = 0
                tumorDice = 0

                oneCase = load_case_data(item)
                examOneCase = DataSets(oneCase)
                dataLoader = DataLoader(examOneCase, batch_size=1, shuffle=False, num_workers=1)

                for idx, (x, yy) in enumerate(dataLoader):
                    with torch.autograd.set_detect_anomaly(True):
                        x = Variable(x).to(device)
                        y = Variable(yy[0]).to(device)
                        who = yy[1]
                        arr = who[0].split('/')
                        dir = str(arr[4])
                        who = str(arr[6])

                        output = model(x)
                        output = torch.sigmoid(output)

                        loss = criterion(output, y)
                        focal_loss = criterion_focal(output, y)
                        loss = loss + focal_loss

                        iterLoss = loss.item()
                        oneCaseValLoss += iterLoss

                        y_p = output.detach().cpu().numpy().copy()
                        y_p = y_p.squeeze(0)
                        y_p1 = y_p[0]
                        y_p2 = y_p[1]
                        y_p3 = y_p[2]
                        y_t = y.detach().cpu().numpy().copy()
                        y_t = y_t.squeeze(0)
                        y_t1 = y_t[0]
                        y_t2 = y_t[1]
                        y_t3 = y_t[2]

                        y_p22 = y_p2 * 127
                        y_p33 = y_p3 * 255
                        y_out = y_p22 + y_p33

                        caseDir = str('./res/' + dir)
                        isExists = os.path.exists(caseDir)
                        if not isExists:
                            os.mkdir(caseDir)
                        cv2.imwrite('./res/' + dir + '/' + 'y_p2_' + who, y_p22)
                        cv2.imwrite('./res/' + dir + '/' + 'y_p3_' + who, y_p33)
                        cv2.imwrite('./res/' + dir + '/' + str(epoch) + '-' + who, y_out)

                        dice1 = dice_coeff(y_p1, y_t1)
                        backgroundDice += dice1
                        dice2 = dice_coeff(y_p2, y_t2)
                        kidneyDice += dice2
                        dice3 = dice_coeff(y_p3, y_t3)
                        tumorDice += dice3

                bDice = backgroundDice / len(oneCase)
                kDice = kidneyDice / len(oneCase)
                tDice = tumorDice / len(oneCase)

                dice_back += bDice
                dice_kidney += kDice
                dice_tumor += tDice

                print('********************************************************************')
                print('**** backgroundDice:{:.8f} ****'.format(backgroundDice / len(oneCase)))
                print('**** kidneyDice:    {:.8f} ****'.format(kidneyDice / len(oneCase)))
                print('**** tumorDice:     {:.8f} ****'.format(tumorDice / len(oneCase)))

            print('mean value:\n')
            print('background:', dice_back/len(validationData))
            print('kidney:', dice_kidney/len(validationData))
            print('tumor:', dice_tumor/len(validationData))


if __name__ == "__main__":
    modelPath = './pth/'
    t = time.time()
    t = time.localtime(t)
    t = time.strftime("%Y--%m--%d %H:%M:%S", t)
    print(t)
    train(300, modelPath)
