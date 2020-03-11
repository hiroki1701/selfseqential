from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
from six.moves import cPickle
import numpy as np
import torch
import torchvision.models as models
import skimage.io
import sys
sys.path.append('/home/nakamura/project/python3_selfsequential')

from torchvision import transforms as trn
preprocess = trn.Compose([
        #trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

from misc.resnet_utils import myResnet
import misc.resnet as resnet

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def main(opt):
    params = vars(opt)
    sys.path.append('/home/nakamura/project/selfsequential')
    sys.path.append('/home/nakamura/project/python3_selfsequential')
    net = getattr(resnet, params['model'])()
    net.load_state_dict(torch.load(os.path.join(params['model_root'], params['model'] + '.pth')))
    my_resnet = myResnet(net)
    my_resnet.cuda()
    my_resnet.eval()

    imgs = json.load(open(params['input_json'], 'r'))
    imgs = imgs['images']

    N = len(imgs)

    seed(123)  # make reproducible

    dir_fc = params['output_dir'] + '_fc'
    dir_att = params['output_dir'] + '_att'

    bu_infos = json.load(open(opt.bu_info))

    for num, bu_info in enumerate(bu_infos):
        id = bu_info['id']

        if os.path.exists(opt.image_root + '/' + str(id) + '.jpg'):
            I = skimage.io.imread(opt.image_root + '/' + str(id) + '.jpg')
        else:
            I = skimage.io.imread(opt.image_root + '_2/' + str(id) + '.jpg')

        if len(I.shape) == 2:
            I = I[:, :, np.newaxis]
            I = np.concatenate((I, I, I), axis=2)

        rate = I.shape[1] / bu_info['shape'][1]

        I = I.astype('float32') / 255.0

        boxes = bu_info['box']
        dets = boxes.astype(np.int32)

        att_feats = np.zeros((36, 2048))

        for k, region in enumerate(dets):
            I_ = torch.from_numpy(I.transpose([2, 0, 1])).cuda()
            I_ = preprocess(I_)

            if np.ceil((region[3]) * rate) - I.shape[0] > 0 and np.ceil((region[2]) * rate) - I.shape[1] > 0:
                I__ = I_ * 0.0
                print('zero')
            else:
                y_0 = int(np.floor(region[1] * rate))
                y_1 = int(np.ceil((region[3]) * rate))
                x_0 = int(np.floor(region[0] * rate))
                x_1 = int(np.ceil((region[2]) * rate))
                I__ = I_[:, y_0:y_1, x_0:x_1]

            try:
                with torch.no_grad():
                    tmp_fc, tmp_att = my_resnet(I__, params['att_size'])
            except RuntimeError:
                I__ = I_ * 0.0
                with torch.no_grad():
                    tmp_fc, tmp_att = my_resnet(I__, params['att_size'])
            att_feats[k] += tmp_fc.data.cpu().float().numpy()

        np.savez_compressed(os.path.join(dir_att, str(id)), feat=att_feats)

        if num % 1000 == 0:
            print('processing %d/%d (%.2f%% done)' % (num, len(bu_infos), num * 100.0 / len(bu_infos)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', default=None, type=str)
    parser.add_argument('--output_dir', default=None, type=str)
    parser.add_argument('--images_root', default=None, type=str)
    parser.add_argument('--att_size', default=14, type=int)
    parser.add_argument('--model', default='resnet101', type=str)
    parser.add_argument('--model_root', default='/mnt/workspace2018/nakamura/imagenet_weights/', type=str)
    parser.add_argument('--bu_info', default='/mnt/poplin/share/2018/nakamura_M1/self_sequential/data/data/bbox_info.json', type=str)

    # You should make bu_info by faster-r-cnn before do it!

    params = parser.parse_args()

    main(params)