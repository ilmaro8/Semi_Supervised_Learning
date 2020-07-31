import torch
from torch.utils import data
import torch.nn.functional as F
import os, sys, getopt

class TeacherModel(torch.nn.Module):
    def __init__(self):

        super(TeacherModel, self).__init__()
        self.pretrained = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_swsl')
        self.pretrained.fc = torch.nn.Sequential(torch.nn.Linear(in_features=2048,out_features=4))
        #self.pretrained = torch.nn.DataParallel(self.pretrained)

    def forward(self, x):
        output=self.pretrained(x)
        return output

class StudentModel(torch.nn.Module):
    def __init__(self):

        super(StudentModel, self).__init__()
        self.pretrained = torch.hub.load('pytorch/vision:v0.4.2', 'densenet121')
        self.pretrained.classifier = torch.nn.Sequential(torch.nn.Linear(in_features=1024,out_features=4))
        #self.pretrained = torch.nn.DataParallel(self.pretrained)

    def forward(self, x):
        output=self.pretrained(x)
        return output

if __name__ == "__main__":
	teacher = TeacherNetwork()
	student = StudentModel()