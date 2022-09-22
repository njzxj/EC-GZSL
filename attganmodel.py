import torch.nn as nn
import torch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



class MLP_D(nn.Module):
    def __init__(self, opt):
        super(MLP_D, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.sigmoid(self.fc2(h))
        return h

class MLP_G(nn.Module):
    def __init__(self, opt):
        super(MLP_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.attSize*2)
        self.fc2 = nn.Linear(opt.attSize*2, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, noise, att):

        h = torch.cat((noise, att), 1)
        h = self.relu(self.fc1(h))
        h = self.lrelu(self.fc2(h))
        return h

    def set_mean_std(self,mean_std):
        self.mean_std = mean_std



class FEA2ATT(nn.Module):
    def __init__(self,opt):
        super(FEA2ATT, self).__init__()
        self.fc1 = nn.Linear(opt.resSize,opt.resSize*2)
        self.fc2 = nn.Linear(opt.resSize*2,opt.attSize*2)
        self.fc3 = nn.Linear(opt.attSize* 2, opt.attSize)

        self.lrelu = nn.LeakyReLU(0.2,True)
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self ,h):
        h = self.relu(self.fc1(h))
        h = self.lrelu(self.fc2(h))
        h = self.relu(self.fc3(h))
        return h




class MLP_CRITIC(nn.Module):
    def __init__(self, opt):
        super(MLP_CRITIC, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.attSize, opt.ndh)
        #self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        return h



class ATT2FEA(nn.Module):
    def __init__(self, opt):
        super(ATT2FEA, self).__init__()
        self.fc1 = nn.Linear(opt.attSize , opt.attSize*2)
        self.fc2 = nn.Linear(opt.attSize*2, opt.resSize*2)
        self.fc3 = nn.Linear(opt.resSize * 2, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, att):
        h = att
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        h = self.lrelu(self.fc3(h))
        return h