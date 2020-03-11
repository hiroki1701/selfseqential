from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random
import torch
import skimage
import skimage.io
import scipy.misc
import cv2
import pdb
from torchvision import transforms as trn

preprocess = trn.Compose([
    # trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

from misc.resnet_utils import myResnet
import misc.resnet


class DataLoaderRaw():
    def __init__(self):

        # Load resnet
        self.size = 100
        self.cnn_model =  'resnet101'
        self.my_resnet = getattr(misc.resnet, self.cnn_model)()
        self.my_resnet.load_state_dict(torch.load('./data/imagenet_weights/' + self.cnn_model + '.pth'))
        self.my_resnet = myResnet(self.my_resnet)
        self.my_resnet.cuda()
        self.my_resnet.eval()
        self.path = os.listdir('/mnt/poplin/tmp/nakamura_M1/MMexercise/dataset/tmp')
        self.path.sort()
        self.imgs = np.zeros((len(self.path),3,self.size,self.size),dtype = np.float32)
        self.tmp_atts = np.zeros((len(self.path),14,14,2048))
        self.tmp_fcs = np.zeros((len(self.path),2048))
        self.batch = 10
        i = 0
        for i in range(len(self.path)):
            name = '/mnt/poplin/tmp/nakamura_M1/MMexercise/dataset/tmp/' + self.path[i]
            img = cv2.imread(name)
            img = cv2.resize(img,(self.size,self.size))
            img = img.astype('float32') / 255.0
            img = img.transpose([2, 0, 1])
            self.imgs[i] += img


    def get_batch(self):
        i = 0
        for i in range(len(self.path)):
            img = torch.from_numpy(self.imgs[i,:,:,:]).cuda()
            # pdb.set_trace()
            tmp_fc, tmp_att = self.my_resnet(img)
            tmp_fc = tmp_fc.cpu().detach().numpy()
            tmp_att = tmp_att.cpu().detach().numpy()
            self.tmp_atts[i] += tmp_att
            self.tmp_fcs[i] += tmp_fc
            print('----------')
# len(self.path)

dataload = DataLoaderRaw()
dataload.get_batch()

np.savez_compressed('/mnt/poplin/tmp/nakamura_M1/MMexercise/dataset/resnet_feature.npz', array_1=dataload.tmp_atts, array_2=dataload.tmp_fcs)


