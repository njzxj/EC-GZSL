import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import sys

import numpy as np

def getLocal(choselabels,alllabels):
    choseloc=np.argwhere(alllabels==choselabels)

    return choseloc


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def changeAtt(att,label,list):
    c = np.ones(att.size(1))
    g_label = []
    for j in list:
        c[j] = 0
    c = np.asarray(c)
    c = torch.from_numpy(c).float()
    attchange = att.clone().detach()
    attchange = torch.mul(attchange, c).float()
    labels = np.ones(att.size(0)).astype(int)
    labels = int(label) * labels
    g_label.extend(labels)
    return attchange,labels


class classifierSSL(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(classifierSSL, self).__init__()
        self.in_dim = in_dim
        self.out_dim=out_dim
        self.fc=nn.Linear(self.in_dim,self.out_dim)


    def forward(self,x):
        x = F.sigmoid(self.fc(x))

        return x


class classifierSSL_all(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(classifierSSL_all, self).__init__()
        self.in_dim = in_dim
        self.out_dim=out_dim
        self.fc=nn.Linear(self.in_dim,self.out_dim)


    def forward(self,x):
        x = F.sigmoid(F.relu(self.fc(x)))

        return x