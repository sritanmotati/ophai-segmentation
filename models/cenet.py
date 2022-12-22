import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from functools import partial
from tqdm import tqdm
from time import time
from torch.autograd import Variable as V
import cv2
import numpy as np
from utils.data_utils import *

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N, H, W = target.size(0), target.size(2), target.size(3)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss

nonlinearity = partial(F.relu, inplace=True)


class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(2, 3, 6, 14)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class CENetModel(nn.Module):
    def __init__(self, num_classes):
        super(CENetModel, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DACblock(512)
        self.spp = SPPblock(512)

        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)
        e4 = self.spp(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)
        # return out

class MyFrame():
    def __init__(self, net, loss, n_classes, lr=2e-4, evalmode=False):
        self.net = net(n_classes).cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        # self.optimizer = torch.optim.SGD(params=self.net.parameters(), lr=lr)
        self.loss = loss()
        self.old_lr = lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()
        
    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id
        
    def test_one_img(self, img):
        pred = self.net.forward(img)
        
        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0

        mask = pred.squeeze().cpu().data.numpy()
        return mask
    
    def test_batch(self, X):
        # self.forward(volatile=True)
        with torch.no_grad():
            pred=self.net(torch.from_numpy(X))
            return pred.cpu().data.numpy()
    
    def eval_batch(self, X, y):
      # self.forward(volatile=True)
      with torch.no_grad():
        pred = self.net(X)
        # print(y.get_device(), pred.get_device())
        return pred, self.loss(y, pred.cpu())
    
    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32)/255.0 * 3.2 - 1.6
        img = V(torch.Tensor(img).cuda())
        
        mask = self.net.forward(img).squeeze().cpu().data.numpy()#.squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        
        return mask
        
    def forward(self, volatile=False):
        self.img = V(self.img.cuda(), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.cuda(), volatile=volatile)
        
    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)
        loss = self.loss(self.mask, pred)
        loss.backward()
        self.optimizer.step()
        return loss.data, pred
        
    def save(self, path):
        torch.save(self.net.state_dict(), path)
        
    def load(self, path):
        self.net.load_state_dict(torch.load(path))
    
    def update_lr(self, new_lr, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print ('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr

class CENet:
    def __init__(self, shape, n_classes):
        self.shape = shape
        self.n_classes = n_classes
        self.model = MyFrame(CENetModel, DiceLoss, n_classes, 2e-4)

    def summary(self):
        print("TBD")
    
    def train(self, train_gen, val_gen, train_steps, val_steps):
        tic=time()
        train_losses = []
        val_losses = []
        train_epoch_best_loss = 1000000
        no_optim = 0
        NUM_EARLY_STOP = 10
        NUM_UPDATE_LR = 10
        num_epochs = 300

        for epoch in range(num_epochs):
            print('********')
            print('epoch:', epoch+1, '    time:', int(time() - tic))
            train_epoch_loss = 0
            index = 0
            for i in tqdm(range(train_steps)):
                img, gt = next(train_gen)
                img, gt = torch.from_numpy(img).float(), torch.from_numpy(gt).float()
                
                self.model.set_input(img, gt)
                train_loss, pred = self.model.optimize()
                train_epoch_loss += train_loss
                index = index + 1

            # show the original images, predication and ground truth on the visdom.

            train_epoch_loss = train_epoch_loss/train_steps
            train_losses.append(train_epoch_loss.item())
            print('train_loss:', train_epoch_loss.item())
            
            if train_epoch_loss >= train_epoch_best_loss:
                no_optim += 1
            else:
                no_optim = 0
                train_epoch_best_loss = train_epoch_loss
                self.model.save('cenet-saved.pth')
            if no_optim > NUM_UPDATE_LR:
                if self.model.old_lr < 5e-7:
                    break
                self.model.load('cenet-saved.pth')
                self.model.update_lr(2.0, factor=True)

            val_loss=0
            for i in range(val_steps):
                img, gt = next(val_gen)
                img, gt = torch.from_numpy(img).float(), torch.from_numpy(gt).float()
                pred_y, val_loss_batch = self.model.eval_batch(img, gt)
                if i == 0:
                    pred0 = np.moveaxis(pred_y.cpu().data.numpy()[0], 0, -1)
                    gt0 = np.moveaxis(gt.cpu().data.numpy()[0], 0, -1)
                val_loss += val_loss_batch
            val_epoch_loss = val_loss/val_steps
            print('val_loss:', val_epoch_loss.item())
            val_losses.append(val_epoch_loss.item())
            eval_pred(np.argmax(gt0, axis=-1), np.argmax(pred0, axis=-1))

            if no_optim > NUM_EARLY_STOP:
                print('early stop at %d epoch' % epoch)
                break
        
        return {'loss': train_losses, 'val_loss': val_losses}

    def predict(self, x):
        return self.model.test_batch(x)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.load(path)

    def get_model(self):
        return self.model