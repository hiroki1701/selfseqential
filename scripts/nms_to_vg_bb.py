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

sys.path.append('/home/nakamura/project/python3_selfsequential/vg')
import visual_genome.local as vg
dir = '/mnt/poplin/share/dataset/visualgenome'
all_images = vg.get_all_image_data(dir)
all_discriptions = vg.get_all_region_descriptions(dir)
print('vg_loaded!')

import numpy as np
import torch
import pdb
import copy


# NMS の関数
def nms_cpu(dets, thresh):
    dets = np.array(dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores = areas
    order = scores.argsort()[::-1]
    #     pdb.set_trace()
    #     order = np.arange(len(scores))

    keep = []
    set_region = {}
    #     pdb.set_trace()
    while order.size > 0:
        i = order.item(0)
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order])
        yy1 = np.maximum(y1[i], y1[order])
        xx2 = np.minimum(x2[i], x2[order])
        yy2 = np.minimum(y2[i], y2[order])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #         ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ovr = inter / (areas[i] + areas[order] - inter)

        inds = np.where(ovr <= thresh)[0]
        if len(inds) > 0:
            if inds[0] != 0:
                inds = np.concatenate([np.zeros(1).astype(inds.dtype), inds])

        non_inds__ = np.where(ovr > thresh)[0]
        if len(non_inds__) > 0:
            if non_inds__[0] == 0:
                non_inds__ = np.delete(non_inds__, 0, 0)

        non_inds_ = copy.copy(order[non_inds__.tolist()])
        order = order[inds[1:]]
        set_region[i] = non_inds_

    return np.array(keep), set_region

def main(opt):
    all_dets_ = []
    all_set_regions = []
    count = 0
    r_count = 0
    # for i,img in enumerate(imgs):
    #     for j, vg_image in enumerate(all_images):
    for j in range(len(all_discriptions)):
        vg_image = all_images[j]

        dets = []
        for k, region in enumerate(all_discriptions[j]):
            axis = []
            axis.append(region.x)
            axis.append(region.y)
            axis.append(region.x + region.width)
            axis.append(region.y + region.height)
            axis.append(0)
            dets.append(axis)
            count += 1

        dets, set_region = nms_cpu(dets, opt.threshold)
        r_count += len(dets)
        all_images[j].nms_regions = dets
        all_images[j].set_regions = set_region
        all_dets_.append(dets)
        all_set_regions.append(set_region)

        if j % 1000 == 0:
            print('processing %d/%d (%.2f%% done)' % (j, len(all_discriptions), j * 100.0 / len(all_discriptions)))
            print('average regions {}'.format(count / len(all_dets_)))
            print('average deleted regions {}'.format(r_count / len(all_dets_)))
            print('---------------------------------------------------------')

    for i in range(len(all_set_regions)):
        for key in all_set_regions[i].keys():
            A = all_set_regions[i][key]
            A = A.tolist()
            all_set_regions[i][key] = A

    for i in range(len(all_dets_)):
        det = all_dets_[i]
        det = det.tolist()
        all_dets_[i] = det

    json.dump(all_dets_, open('/mnt/workspace2018/nakamura/selfsequential/data/region_dets_allvg_' + opt.name +  '.json', 'w'))
    json.dump(all_set_regions, open('/mnt/workspace2018/nakamura/selfsequential/data/marged_set_allvg_' + opt.name +  '.json', 'w'))

def concat(opt):
    import os
    region_info = json.load(open('/mnt/workspace2018/nakamura/selfsequential/data/region_dets_allvg_' + opt.name +  '.json'))
    img_dir = '/mnt/workspace2018/nakamura/vg_feature/feature_frc'
    save_dir = '/mnt/workspace2019/nakamura/vg_feature/resnet_region_nms' + opt.name +  '/'
    # img_list = os.listdir(img_dir)
    for i in range(len(all_discriptions)):
        att_feats = np.zeros((len(region_info[i]), 2048))
        for j, region_num in enumerate(region_info[i]):
            feature = np.load(img_dir + '/' + str(i) + '_' + str(region_num) + '.npy')
            att_feats[j] = feature
        np.savez_compressed(os.path.join(save_dir, str(i)), feat=att_feats)

        if i % 1000 == 0:
            print('processing %d/%d (%.2f%% done)' % (i, len(all_discriptions), i * 100.0 / len(all_discriptions)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', default=0.7, type=float)
    parser.add_argument('--name', default='07', type=str)
    parser.add_argument('--concat', default=0, type=int)
    params = parser.parse_args()

    main(params)
    if params.concat == 1:
        concat(params)
    # main()