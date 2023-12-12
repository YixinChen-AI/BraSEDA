import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from torch.utils.checkpoint import checkpoint
import albumentations as albu
import torchvision
import os

class decoder_unet(nn.Module):
  def __init__(self, in_channels, middle_channels, out_channels):
    super(decoder_unet, self).__init__()
    self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    self.conv_relu = nn.Sequential(
        nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
        )
  def forward(self, x1, x2):
    x1 = self.up(x1)
    x1 = torch.cat((x1, x2), dim=1)
    x1 = self.conv_relu(x1)
    return x1

class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        
        self.base_model = torchvision.models.resnet18(True)
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2])
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = self.base_layers[5]
        self.layer4 = self.base_layers[6]
        self.layer5 = self.base_layers[7]
        self.decode4 = decoder_unet(512, 256+256, 256)
        self.decode3 = decoder_unet(256, 256+128, 256)
        self.decode2 = decoder_unet(256, 128+64, 128)
        self.decode1 = decoder_unet(128, 64+64, 64)
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
            )
        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        e1 = self.layer1(input) # 64,128,128
        e2 = self.layer2(e1) # 64,64,64
        e3 = self.layer3(e2) # 128,32,32
        e4 = self.layer4(e3) # 256,16,16
        f = self.layer5(e4) # 512,8,8
        d4 = self.decode4(f, e4) # 256,16,16
        d3 = self.decode3(d4, e3) # 256,32,32
        d2 = self.decode2(d3, e2) # 128,64,64
        d1 = self.decode1(d2, e1) # 64,128,128
        d0 = self.decode0(d1) # 64,256,256
        out = self.conv_last(d0) # 1,256,256
        return out


def dice_loss(predict,target):
    target = target.float()
    smooth = 1e-4
    intersect = torch.sum(predict*target)
    dice = (2 * intersect + smooth)/(torch.sum(target)+torch.sum(predict*predict)+smooth)
    loss = 1.0 - dice
    return loss


class DiceLoss(nn.Module):
    def __init__(self,n_classes):
        super().__init__()
        self.n_classes = n_classes
        
    def one_hot_encode(self,input_tensor):
        print(input_tensor.shape,'inpt')
        tensor_list = []
        for i in range(self.n_classes):
            tmp = (input_tensor==i) * torch.ones_like(input_tensor)
            tensor_list.append(tmp)
        output_tensor = torch.cat(tensor_list,dim=1)
        return output_tensor.float()
    
    def forward(self,input,target,weight=None,softmax=True):
        if softmax:
            inputs = F.softmax(input,dim=1)
        target = F.one_hot(target,5).permute(0,3,1,2)
        if weight is None:
            weight = [1] * self.n_classes
#         print(inputs.shape, target.shape,"inputs.shape, target.shape, after")
        assert inputs.shape == target.shape,'size must match'
        class_wise_dice = []
        loss = 0.0
        for i in range(self.n_classes):
            diceloss = dice_loss(inputs[:,i], target[:,i])
            class_wise_dice.append(diceloss)
            loss += diceloss * weight[i]
        return loss/self.n_classes

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.eps = 1e-4
        self.num_classes = num_classes

    def forward(self, predict, target):
        weight = []
        for c in range(self.num_classes):
            weight_c = torch.sum(target == c).float()
            #print("weightc for c",c,weight_c)
            weight.append(weight_c)
        weight = torch.tensor(weight).to(target.device)
        weight = 1 - weight / (torch.sum(weight))
        if len(target.shape) == len(predict.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        wce_loss = F.cross_entropy(predict, target.long(), weight)
        return wce_loss

class DiceCeLoss(nn.Module):
     #predict : output of model (i.e. no softmax)[N,C,*]
     #target : gt of img [N,1,*]
    def __init__(self,num_classes,alpha=1.0,ce=False):
        '''
        calculate loss:
            celoss + alpha*celoss
            alpha : default is 1
        '''
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.diceloss = DiceLoss(self.num_classes)
        self.celoss = WeightedCrossEntropyLoss(self.num_classes)
        self.ce = ce
        
    def forward(self,predict,label,mask=None):
        if self.ce:
            diceloss = self.diceloss(predict,label)
            celoss = self.celoss(predict,label)
            loss = celoss + self.alpha * diceloss
        else:
            loss = self.diceloss(predict,label)
        return loss

class CB(nn.Module):
    def __init__(self,in_,out_):
        super().__init__()
        self.conv = nn.Conv2d(in_,out_,3,stride=1,padding=1)
        self.norm = nn.InstanceNorm2d(out_)
        self.act = nn.LeakyReLU()
    def forward(self,x):
        return self.act(self.norm(self.conv(x)))

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.CB1_1 = CB(1,64)
        self.CB1_2 = CB(64,64)
        self.CB2_1 = CB(64,128)
        self.CB2_2 = CB(128,128)
        self.CB3_1 = CB(128,128)
        self.CB3_2 = CB(128,128)
        self.CB4_1 = CB(128,128)
        self.CB4_2 = CB(128,128)
        self.CB5_1 = CB(128,128)
        self.CB5_2 = CB(128,128)
        self.downsample = nn.MaxPool2d(kernel_size=(2,2))
    def forward(self,x):
        x = self.CB1_2(self.CB1_1(x))
        x1 = self.CB2_2(self.CB2_1(self.downsample(x)))
        x2 = self.CB3_2(self.CB3_1(x1))
        x3 = self.CB4_2(self.CB4_1(x2))
        x4 = self.CB5_2(self.CB5_1(x3))
        return [x1,x2,x3,x4]

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.CB1_1 = nn.Conv2d(64,1,kernel_size=1,bias=False)
        self.CB1_2 = CB(64,64)
        self.CB2_1 = CB(128,64)
        self.CB2_2 = CB(128,128)
        self.CB3_1 = CB(128,128)
        self.CB3_2 = CB(128,128)
        self.CB4_1 = CB(128,128)
        self.CB4_2 = CB(128,128)
        self.CB5_1 = CB(128,128)
        self.CB5_2 = CB(128,128)
        self.upsample = nn.Upsample(scale_factor=2,mode="bilinear")
    def forward(self,x):
        x = self.CB5_1(self.CB5_2(x))
        x = self.CB4_1(self.CB4_2(x))
        x = self.CB3_1(self.CB3_2(x))
        x = self.CB2_1(self.CB2_2(x))
        x = self.CB1_1(self.CB1_2(self.upsample(x)))
        return torch.sigmoid(x)
        
class Dis(nn.Module):
    def __init__(self, input_nc=1):
        super().__init__()
        model = [nn.Conv2d(input_nc, 64, 3, stride=2),
                 nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(64, 64, 3, stride=2),
                  nn.InstanceNorm2d(64),
                  nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(64,64, 3, stride=2),
                  nn.InstanceNorm2d(64),
                  nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(64,64, 3, padding=1),
                  nn.InstanceNorm2d(64),
                  nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(64,1, kernel_size=1,bias=False)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return torch.sigmoid(x)

class CMIN(nn.Module):
    def __init__(self,shape):
        super().__init__()
        self.ESM = nn.Parameter(torch.randn(shape).float())
    def forward(self,zct,zmr):
        sct,mct = self.step_one(zct)
        smr,mmr = self.step_one(zmr)
        recon_zct = self.step_two(sct,mct)
        recon_zmr = self.step_two(smr,mmr)
        pseudo_zmr = self.step_two(sct+self.ESM,mmr)
        pseudo_zct = self.step_two(smr-self.ESM,mct)
        return pseudo_zmr,pseudo_zct,recon_zmr,recon_zct,{"sct":sct,"mct":mct,"smr":smr,"mmr":mmr}
    
    def step_one(self,x):
        mu = torch.mean(x,dim=[2,3])[:,:,None,None]
        std = torch.std(x,dim=[2,3])[:,:,None,None]
        x = (x - mu)/std
        return x,(mu,std)

    def step_two(self,s,m):
        return s*m[1]+m[0]
class BraSEDA_CMIN(nn.Module):
    def __init__(self,device='cuda:0'):
        super().__init__()
        lr_seg = 0.00005
        lr = 0.00003
        self.skip = False
        self.device = device

        self.encoder = Encoder()
        # self.encoder_ct = Encoder()
        # self.encoder_mr = Encoder()
        self.decoder_ct = Decoder()
        self.decoder_mr = Decoder()
        self.cmin4 = CMIN(shape=(128,128,128))
        self.cmin3 = CMIN(shape=(128,128,128))
        self.cmin2 = CMIN(shape=(128,128,128))
        self.cmin1 = CMIN(shape=(128,128,128))
        self.cmins = [self.cmin1,self.cmin2,self.cmin3,self.cmin4]
        
        
        self.m_ct_pattern = [torch.zeros(128).to(device),torch.zeros(128).to(device)]
        self.m_mr_pattern = [torch.zeros(128).to(device),torch.zeros(128).to(device)]
        #network
        self.dis_ct = Dis()
        self.dis_mr = Dis()
        self.dis_seg = Dis(input_nc=97)
        self.seg = UNet(97)
        #optimizer
        self.en_de_opt = torch.optim.Adam([p for p in self.encoder.parameters()]+
                                          # [p for p in self.encoder_mr.parameters()]+
                                          [p for p in self.decoder_ct.parameters()]+
                                          [p for p in self.decoder_mr.parameters()]+
                                          [p for p in self.cmin1.parameters()]+
                                          [p for p in self.cmin2.parameters()]+
                                          [p for p in self.cmin3.parameters()]+
                                          [p for p in self.cmin4.parameters()],
                                          lr=lr,betas=(0.5,0.999),weight_decay=3e-5)
        
        self.disct_opt = torch.optim.Adam(self.dis_ct.parameters(),lr=lr,betas=(0.5,0.999),weight_decay=3e-5)
        self.dismr_opt = torch.optim.Adam(self.dis_mr.parameters(),lr=lr,betas=(0.5,0.999),weight_decay=3e-5)
        self.seg_opt = torch.optim.Adam(self.seg.parameters(),lr=lr)
        self.disseg_opt = torch.optim.Adam(self.dis_seg.parameters(),lr=lr,weight_decay=3e-5)
        
        #loss
        self.L2 = nn.MSELoss()
        self.L1 = nn.L1Loss()
        self.ce = nn.CrossEntropyLoss()
        self.ce_none = nn.CrossEntropyLoss(reduction="none")
        self.iteration = 0
        
        self.gan_train_count = 1

    def forward(self,ct,mr,mr_y):
        self.origin_ct = ct
        self.origin_mr = mr
        self.origin_y = mr_y

        # reconstruction process & adaptation process
        ct_feats = self.encoder(self.origin_ct)
        mr_feats = self.encoder(self.origin_mr)

        pseudo_zmrs,pseudo_zcts,recon_zmrs,recon_zcts,origin_smdicts = [],[],[],[],[]
        for ct_feat,mr_feat,cmin in zip(ct_feats,mr_feats,self.cmins):
            pseudo_zmr,pseudo_zct,recon_zmr,recon_zct,smdict = cmin(ct_feat,mr_feat)
            pseudo_zmrs.append(pseudo_zmr)
            pseudo_zcts.append(pseudo_zct)
            recon_zmrs.append(recon_zmr)
            recon_zcts.append(recon_zct)
            origin_smdicts.append(smdict)

        self.pseudo_mr = self.decoder_mr(pseudo_zmrs[3])
        self.pseudo_ct = self.decoder_ct(pseudo_zcts[3])
        self.recon_mr = self.decoder_mr(recon_zmrs[3])
        self.recon_ct = self.decoder_ct(recon_zcts[3])
        self.origin_smdict = origin_smdicts[3]  
        self.m_ct_pattern[0] = 0.99 * self.m_ct_pattern[0] + 0.01 * torch.mean(self.origin_smdict["mct"][0],dim=0)
        self.m_ct_pattern[1] = 0.99 * self.m_ct_pattern[1] + 0.01 * torch.mean( self.origin_smdict["mct"][1],dim=0)
        self.m_mr_pattern[0] = 0.99 * self.m_mr_pattern[0] + 0.01 * torch.mean( self.origin_smdict["mmr"][0],dim=0)
        self.m_mr_pattern[1] = 0.99 * self.m_mr_pattern[1] + 0.01 * torch.mean(self.origin_smdict["mmr"][1],dim=0)
        
        

        # cycle process
        ct_feats = self.encoder(self.pseudo_ct)
        mr_feats = self.encoder(self.pseudo_mr)
        cycle_zmrs,cycle_zcts,pseudo_smdicts  = [],[],[]
        for ct_feat,mr_feat,cmin in zip(ct_feats,mr_feats,self.cmins):
            cycle_zmr,cycle_zct,_,_,sm_dict = cmin(ct_feat,mr_feat)
            cycle_zmrs.append(cycle_zmr)
            cycle_zcts.append(cycle_zct)
            pseudo_smdicts.append(sm_dict)
            
        self.cycle_mr = self.decoder_mr(cycle_zmrs[3])
        self.cycle_ct = self.decoder_ct(cycle_zcts[3])
        self.pseudo_smdict = pseudo_smdicts[3]
    def update_en_de_seg(self):
        # seg
        loss_seg = self.ce(self.seg(self.origin_mr),self.origin_y)
        loss_seg += self.ce(self.seg(self.pseudo_ct),self.origin_y)
        self.seg_opt.zero_grad();loss_seg.backward(retain_graph=True);self.seg_opt.step()

        # encoder,decoder,ESM
        loss_rec = self.L1(self.recon_ct,self.origin_ct) + self.L1(self.recon_mr,self.origin_mr)
        loss_cyc = self.L1(self.cycle_ct,self.origin_ct) + self.L1(self.cycle_mr,self.origin_mr)
        loss_adv = (torch.mean(self.dis_ct(self.pseudo_ct))-1)**2 +(torch.mean(self.dis_mr(self.pseudo_mr))-1)**2 +  (torch.mean(self.dis_seg(self.seg(self.pseudo_mr)))-1)**2
        # loss_smc = self.L1(self.origin_smdict["sct"],self.pseudo_smdict["smr"]) + self.L1(self.origin_smdict["smr"],self.pseudo_smdict["sct"])
        loss_smc = self.L1(self.origin_smdict["mct"][0],self.pseudo_smdict["mct"][0]) + self.L1(self.origin_smdict["mct"][1],self.pseudo_smdict["mct"][1])
        loss_smc += self.L1(self.origin_smdict["mmr"][0],self.pseudo_smdict["mmr"][0]) + self.L1(self.origin_smdict["mmr"][1],self.pseudo_smdict["mmr"][1])
        loss_esm = torch.tensor(0.).to(self.device)
        for cmin in [self.cmin4]:
            loss_esm += torch.mean(torch.mean(cmin.ESM,dim=[1,2])**2) + torch.mean((torch.std(cmin.ESM,dim=[1,2])-1)**2)
        loss = 5*loss_rec + 5*loss_cyc + loss_adv +1*loss_smc + loss_esm
        self.en_de_opt.zero_grad();loss.backward();self.en_de_opt.step()
        self.loss = loss.item()
        self.loss_rec = loss_rec.item()
        self.loss_cyc = loss_cyc.item()
        self.loss_adv = loss_adv.item()
        self.loss_smc = loss_smc.item()
        self.loss_esm = loss_esm.item()
    def update_dis(self):
        # dis_ct
        loss_disct = (torch.mean(self.dis_ct(self.origin_ct))-1)**2 + (torch.mean(self.dis_ct(self.recon_ct))-1)**2+(torch.mean(self.dis_ct(self.pseudo_ct)))**2
        self.disct_opt.zero_grad();loss_disct.backward(retain_graph=True);self.disct_opt.step()
        loss_dismr = (torch.mean(self.dis_mr(self.origin_mr))-1)**2 + (torch.mean(self.dis_mr(self.recon_mr))-1)**2+(torch.mean(self.dis_mr(self.pseudo_mr)))**2
        self.dismr_opt.zero_grad();loss_dismr.backward(retain_graph=True);self.dismr_opt.step()
        loss_disseg = (torch.mean(self.dis_seg(self.seg(self.origin_mr)))-1)**2+(torch.mean(self.dis_seg(self.seg(self.pseudo_mr))))**2
        self.disseg_opt.zero_grad();loss_disseg.backward(retain_graph=True);self.disseg_opt.step()
        self.loss_disct = loss_disct.item()
        self.loss_dismr = loss_dismr.item()
        self.loss_disseg = loss_disseg.item()
# 