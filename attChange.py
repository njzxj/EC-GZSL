from __future__ import print_function
import argparse
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
import math

import attganmodel
import util2
import classifier
import classifier2
import sys


import numpy as np
import myutils
from myutils import changeAtt,classifierSSL,getLocal, Logger
from attganmodel import MLP_D,MLP_G,FEA2ATT, ATT2FEA

from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='AWA2', help='FLO')
parser.add_argument('--dataroot', default='data', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
parser.add_argument('--gzsl', action='store_true', default=True, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=True, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=1024, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=85, help='size of semantic features')
parser.add_argument('--nz', type=int, default=85, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=1024, help='size of the hidden units in discriminator')
parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=1, help='weight of the classification loss')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.01, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netG_name', default='')
parser.add_argument('--netD_name', default='')
parser.add_argument('--outf', default='./checkpoint/', help='folder to output data and model checkpoints')
parser.add_argument('--outname', help='folder to output data and model checkpoints')
parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--val_every', type=int, default=10)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--manualSeed', type=int,default=1077, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=50, help='number of all classes')
parser.add_argument('--beta2', type=int, default=0.5, help='number of all classes')
parser.add_argument('--beta3', type=int, default=0.3, help='number of all classes')
parser.add_argument('--nepoch_fea2att', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--fea2att_lr', type=float, default=0.001, help='learning rate to train softmax classifier')

parser.add_argument('--tem', type=int, default=1.5, help='number of all classes')
parser.add_argument('--n_clusters', type=int, default=3, help='number of epochs to train for')


sys.stdout = Logger('log/'+time.strftime("%Y_%m_%d %H_%M_%S", time.localtime())+'.log')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass





if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



# 设置随机数种子
setup_seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
data = util2.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)



netG = attganmodel.MLP_G(opt)
print(netG)
netD = attganmodel.MLP_CRITIC(opt)
print(netD)





sig = nn.Sigmoid()
crossentropyloss=nn.CrossEntropyLoss()
cls_criterion = nn.NLLLoss()
mseloss = nn.MSELoss()

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
input_att_gan = torch.FloatTensor(opt.batch_size, opt.attSize)

input_mean_std = torch.zeros([opt.batch_size, opt.attSize,2],dtype=torch.float)

train_X_f2a = data.train_feature.clone().detach()
train_Y_f2a = data.train_label.clone().detach()

input_label = torch.LongTensor(opt.batch_size)
input_label_map = torch.LongTensor(opt.batch_size)
test_label_map = torch.LongTensor(train_Y_f2a.size(0))
all_att = torch.FloatTensor(data.attribute.size(0),data.attribute.size(1))
all_att.copy_(data.attribute)
noise = torch.FloatTensor(opt.batch_size, opt.nz)

cls_att = nn.NLLLoss()

fea2att = FEA2ATT(opt)
if opt.cuda:
    fea2att.cuda()

    input_res = input_res.cuda()
    input_att =  input_att.cuda()
    input_label = input_label.cuda()
    input_label_map =input_label_map.cuda()
    test_label_map=test_label_map.cuda()
    all_att = all_att .cuda()
    input_att_gan =input_att_gan.cuda()
    noise = noise.cuda()
    input_mean_std=input_mean_std.cuda()

    netD.cuda()
    netG.cuda()

    train_X_f2a = train_X_f2a.cuda()
    train_Y_f2a = train_Y_f2a.cuda()

    cls_att.cuda()

    sig .cuda()
    crossentropyloss .cuda()
    cls_criterion.cuda()
    mseloss.cuda()

test_label_map.copy_(util2.map_label(train_Y_f2a, data.seenclasses))

def sample1():
    batch_feature, batch_label, batch_att = data.next_batch1(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(batch_label)
    input_label_map.copy_(util2.map_label(batch_label, data.seenclasses))

def sample2():
    batch_feature, batch_label, batch_att ,batch_att_gan,batch_mean_std= data.next_batch2(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_att_gan.copy_(batch_att_gan)
    input_mean_std.copy_(batch_mean_std)
    input_label.copy_(batch_label)
    input_label_map.copy_(util2.map_label(batch_label, data.seenclasses))






def pretrain_sim(pre_att,all_att,labels,crossentropyloss):
    pre_smi = torch.mm(pre_att, all_att.T)
    a = torch.norm(pre_att, p=2, dim=1).unsqueeze(1)
    b = torch.norm(all_att, p=2, dim=1).unsqueeze(1)
    ab = torch.mm(a, b.T)
    pre_smi = pre_smi / ab
    pre = sig(pre_smi)

    loss = crossentropyloss(pre,labels)

    pre_clone = pre.clone().detach().cpu().numpy()
    pre_labels = np.argmax(pre_clone, axis=1)
    y_true = labels.clone().detach().cpu().numpy().flatten().tolist()
    y_pred = pre_labels.flatten().tolist()
    acc = accuracy_score(y_true, y_pred)
    print("acc is :{0}".format(acc))
    return acc,loss

def cls_attributes(att_e,labels,att_o,cls_att,opt):
    att_o = att_o.unsqueeze(0)
    att_e = att_e.unsqueeze(1)
    cha = att_e - att_o
    cha = torch.norm(cha, dim=2) * opt.tem
    """
    make = np.zeros((att_e.size(0), 40))
    for i in range(att_e.size(0)):
        make[i, labels[i]] = 1
    make = torch.from_numpy(make).cuda()
    cha_loss = cha * make
    """
    cha =1/cha

    cha = cha / opt.tem
    cha = torch.exp(cha)

    sum = torch.sum(cha,dim=1).unsqueeze(1)
    cha = cha / sum

    log_cha = torch.log(cha)
    loss = cls_att(log_cha,Variable(labels))




    #loss = torch.sum(cha_loss) / att_e.size(0)


    numpy_cha = cha.detach().cpu().numpy()
    #print(cha)
    pre = np.argmax(numpy_cha, axis=1)
    numpy_labels = labels.detach().cpu().numpy().flatten()
    acc = accuracy_score(numpy_labels, pre)
    min = np.max(numpy_cha, axis=1)

    #print(min)
    print(acc)
    return loss ,acc


att_o = util2.map_att(data.seenclasses,data.attribute)
att_o  = att_o.cuda()

optimizerFeat2att = optim.Adam(fea2att.parameters(), lr=opt.fea2att_lr, betas=(opt.beta1, 0.999))





# freeze the classifier during the optimization



best_acc_fea2att = 0
for i in range(opt.nepoch_fea2att):
    fea2att.train()

    for j in range(0, data.ntrain, opt.batch_size):
        print("epoch:{0}".format(i))
        print("train")
        fea2att.zero_grad()

        sample1()
        pre_att = fea2att(input_res)

        #_,loss = pretrain_sim(pre_att,all_att,input_label,crossentropyloss)

        #loss_e_cls, _ = cls_attributes(pre_att, input_label_map, att_o, opt,40)
        loss_e_cls, _ = cls_attributes(pre_att, input_label_map, att_o, cls_att,opt)
        input_attv = Variable(input_att)
        input_resv = Variable(input_res)

        l2loss = 0
        for param in fea2att.parameters():
            l2loss += torch.norm(param)


        print("#################")


        loss = loss_e_cls + 0.01*l2loss
        print("loss_e_cls:{0} ".format(loss_e_cls))
        loss.backward()
        optimizerFeat2att.step()

    fea2att.eval()

    print("test")
    pre_att_test = fea2att(train_X_f2a)
    #acc,_=pretrain_sim(pre_att_test, all_att,train_Y_f2a,crossentropyloss)
    _,acc = cls_attributes(pre_att_test, test_label_map, att_o, cls_att,opt)
    print(acc)
    if best_acc_fea2att<acc:
        best_acc_fea2att = acc
        print("1111")
        torch.save(fea2att,"fea2att.pkl")



fea2att = torch.load("fea2att.pkl")


attribute = data.attribute.detach().clone()
A = torch.mm(attribute,attribute.T)
B = torch.norm(attribute,dim=1).unsqueeze(0)
B = torch.mm(B.T,B)
A = A/B
for i in range(A.size(0)):
    A[i,i] = 0
_,max = torch.max(A,0)







all_train = data.train_feature.detach().clone().cuda()
all_pre_att = fea2att(all_train)
all_label = data.train_label.detach().clone()
new_label = data.train_label.detach().clone()
label_set = all_label.clone().detach().clone().numpy().flatten().tolist()
label_set = set(label_set)

new_att= torch.FloatTensor(opt.nclass_all*opt.n_clusters, opt.attSize)
S_new_att= torch.zeros((opt.nclass_all*opt.n_clusters, opt.attSize))

attribute = data.attribute

print(label_set)
torch.set_printoptions(threshold=np.inf)
for i in label_set:
    list = np.where(all_label == i)[0]
    cls_pre_att = all_pre_att[list,:].detach().clone().cpu().squeeze().numpy()

    kmeans=KMeans(n_clusters=opt.n_clusters,random_state=opt.manualSeed)
    kmeans.fit(cls_pre_att)
    n=0
    for j in list:
        new_label[j] = new_label[j]*opt.n_clusters+kmeans.labels_[n]
        n=n+1
    centers=kmeans.cluster_centers_
    centers = torch.from_numpy(centers)
    print(centers)
    new_att.narrow(0,i*opt.n_clusters,opt.n_clusters).copy_(centers)
    all_cent = attribute[i,:]
    S = centers-all_cent
    S_new_att.narrow(0, i * opt.n_clusters, opt.n_clusters).copy_(S)

# new_att= attribute.clone();
# new_label=all_label;

unseen_label = data.test_unseen_label.detach().clone().numpy().flatten().tolist()
unseen_label = set(unseen_label)
print("//////////////////////////////////////////")
for i in unseen_label:
    smi = max[i]
    s_smi = S_new_att[smi*opt.n_clusters:smi*opt.n_clusters+opt.n_clusters,:]*A[smi,i]*0.5
    new = attribute[i,:]+s_smi
    print(new)
    new_att.narrow(0, i * opt.n_clusters, opt.n_clusters).copy_(new)



np.save('new_att'+str(opt.n_clusters),new_att.numpy())

np.save('new_label'+str(opt.n_clusters),new_label.numpy())





