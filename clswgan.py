from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import math
import util 
import classifier
import classifier2
import sys
import model
from attganmodel import FEA2ATT
from myutils import changeAtt,classifierSSL,getLocal, Logger
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='AWA2', help='FLO')
parser.add_argument('--dataroot', default='data', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=2000, help='number features to generate per class')
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
parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator')
parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=0.01, help='weight of the classification loss')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
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
parser.add_argument('--manualSeed', type=int,default=2288, help='manual seed')
#parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=50, help='number of all classes')
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
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

# initialize generator and discriminator
netG = model.MLP_G(opt)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = model.MLP_CRITIC(opt)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# classification loss, Equation (4) of the paper
cls_criterion = nn.NLLLoss()

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)

input_label = torch.LongTensor(opt.batch_size)
input_label_org = torch.LongTensor(opt.batch_size)

fea2att = FEA2ATT(opt)

if opt.cuda:
    netD.cuda()
    netG.cuda()
    fea2att.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()

    cls_criterion.cuda()
    input_label = input_label.cuda()
    input_label_org = input_label_org.cuda()

def sample():
    batch_feature, batch_label, batch_att ,batch_label_org = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.new_seenclasses))
    input_label_org.copy_(util.map_label(batch_label_org, data.seenclasses))

def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label_new = torch.LongTensor(nclass*num)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
        
    for i in range(nclass):
        iclass = classes[i].long()
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        output = netG(Variable(syn_noise, volatile=True), Variable(syn_att, volatile=True))
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label_new.narrow(0, i*num, num).fill_(iclass)
        syn_label.narrow(0, i * num, num).fill_(iclass / opt.n_clusters)

    return syn_feature, syn_label_new ,syn_label



def prototype_generate_syn_feature(netG, attribute):
    nclass = attribute.size(0)
    syn_feature = torch.FloatTensor(nclass , opt.resSize)
    syn_noise = torch.zeros((nclass , opt.nz))
    if opt.cuda:
        syn_noise = syn_noise.cuda()
        syn_att = attribute.detach().clone().cuda()
    else:
        syn_att = attribute.detach().clone()

    #syn_noise.normal_(0, 0)
    output = netG(Variable(syn_noise, volatile=True), Variable(syn_att, volatile=True))
    syn_feature.copy_(output.data.cpu())

    return syn_feature


# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    #print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, Variable(input_att))

    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty


# train a classifier on seen classes, obtain \theta of Equation (4)
pretrain_cls = classifier.CLASSIFIER(data.train_feature, util.map_label(data.train_label, data.seenclasses), data.seenclasses.size(0), opt.resSize, opt.cuda, 0.001, 0.5, 50, 100, opt.pretrain_classifier)

# freeze the classifier during the optimization
for p in pretrain_cls.model.parameters(): # set requires_grad to False
    p.requires_grad = False



fea2att = torch.load("fea2att.pkl")
h=0
for epoch in range(opt.nepoch):
    FP = 0 
    mean_lossD = 0
    mean_lossG = 0

    for i in range(0, data.ntrain, opt.batch_size):
        ############################
        # (1) Update D network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        for iter_d in range(opt.critic_iter):
            sample()
            netD.zero_grad()
            # train with realG
            # sample a mini-batch
            sparse_real = opt.resSize - input_res[1].gt(0).sum()
            input_resv = Variable(input_res)
            input_attv = Variable(input_att)

            criticD_real = netD(input_resv, input_attv)
            criticD_real = criticD_real.mean()


            # train with fakeG
            noise.normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev, input_attv)
            fake_norm = fake.data[0].norm()
            sparse_fake = fake.data[0].eq(0).sum()
            criticD_fake = netD(fake.detach(), input_attv)
            criticD_fake = criticD_fake.mean()


            # gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)

            l2loss = 0
            for param in netD.parameters():
                l2loss += torch.norm(param)

            Wasserstein_D = criticD_real - criticD_fake
            D_cost = criticD_fake - criticD_real + gradient_penalty#+0.01*l2loss
            D_cost.backward()
            optimizerD.step()

        ############################
        # (2) Update G network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = False # avoid computation

        netG.zero_grad()
        input_attv = Variable(input_att)
        noise.normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev, input_attv)
        criticG_fake = netD(fake, input_attv)
        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake
        # classification loss
        c_errG = cls_criterion(pretrain_cls.model(fake), Variable(input_label_org))

        l2loss = 0
        l1loss = 0
        for param in netG.parameters():
            l2loss += torch.norm(param)
            l1loss += torch.norm(param,p=1)


        errG = G_cost + opt.cls_weight*c_errG+0.01*l2loss#+0.000001*l1loss
        errG.backward()
        optimizerG.step()

    mean_lossG /=  data.ntrain / opt.batch_size 
    mean_lossD /=  data.ntrain / opt.batch_size 
    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, c_errG:%.4f'
              % (epoch, opt.nepoch, D_cost, G_cost, Wasserstein_D, c_errG))

    # evaluate the model, set G to evaluation mode
    netG.eval()
    # Generalized zero-shot learning


    if opt.gzsl:
        syn_feature, syn_label_new, syn_label = generate_syn_feature(netG, data.new_unseenclasses, data.new_att, opt.syn_num)

        #syn_feature_org, syn_label_new_org, syn_label_org = generate_syn_feature(netG, data.unseenclasses, data.attribute,opt.syn_num)

        pro_feature = prototype_generate_syn_feature(netG, data.attribute)

        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y_new = torch.cat((data.new_train_label, syn_label_new), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)




        if opt.cuda:
            train_X = train_X.cuda()

        train_X_att = fea2att(train_X)
        train_X_att = train_X_att.cpu()
        train_X_att_gan = prototype_generate_syn_feature(netG, train_X_att).cpu()
        train_X = train_X.cpu()
        #train_X = torch.cat((train_X, train_X_att_gan), 1)
        train_X = torch.cat((train_X, train_X_att_gan, train_X_att.cpu()), 1)
        test_seen_feature = data.test_seen_feature
        test_unseen_feature = data.test_unseen_feature
        if opt.cuda:
            test_seen_feature = test_seen_feature.cuda()
            test_unseen_feature = test_unseen_feature.cuda()
        test_seen_feature_att = fea2att(test_seen_feature)
        test_unseen_feature_att = fea2att(test_unseen_feature)
        test_seen_feature_att_gan = prototype_generate_syn_feature(netG, test_seen_feature_att)
        test_unseen_feature_att_gan = prototype_generate_syn_feature(netG, test_unseen_feature_att)
        test_seen_feature_att_gan = test_seen_feature_att_gan.cpu()
        test_unseen_feature_att_gan = test_unseen_feature_att_gan.cpu()
        test_seen_feature=test_seen_feature.cpu()
        test_unseen_feature = test_unseen_feature.cpu()

        #test_seen_feature = torch.cat((test_seen_feature, test_seen_feature_att_gan), 1)
        #test_unseen_feature = torch.cat((test_unseen_feature, test_unseen_feature_att_gan), 1)
        test_seen_feature = torch.cat((test_seen_feature, test_seen_feature_att_gan, test_seen_feature_att.cpu()), 1)
        test_unseen_feature = torch.cat((test_unseen_feature, test_unseen_feature_att_gan, test_unseen_feature_att.cpu()), 1)





        nclass = opt.nclass_all*opt.n_clusters
        #cls = classifier2.CLASSIFIER(train_X, train_Y_new,train_Y,pro_feature, opt.n_clusters,data, nclass, opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, True)

        cls = classifier2.CLASSIFIER(train_X, train_Y_new, train_Y, pro_feature, opt.n_clusters, data,test_seen_feature , test_unseen_feature,nclass, opt.cuda,opt.classifier_lr, 0.5, 25, opt.syn_num, True)

        print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))
        if h < cls.H:
            h = cls.H
            print("best H is {0}".format(h))
            torch.save(netG,"model/unseen"+str(cls.acc_unseen)+"seen"+str(cls.acc_seen)+"H"+str(cls.H)+str(opt.dataset)+".pkl")
    # Zero-shot learning

    # reset G to training mode
    netG.train()

