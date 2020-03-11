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
import pdb

from torchvision import transforms as trn
preprocess = trn.Compose([
        #trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

info = json.load(open('/mnt/poplin/share/2018/nakamura_M1/self_sequential/data/data/cocotalk_2.json'))
ix_to_word = info['ix_to_word']
word_to_ix = dict()
size = len(ix_to_word)
pdb.set_trace()
#
for i in range(size):
    id = str(i+1)
    word = ix_to_word[id]
    word_to_ix.update({word:id})
    print(word,id)

info.update({'word_to_ix':word_to_ix})
pdb.set_trace()
f = open('/mnt/poplin/share/2018/nakamura_M1/self_sequential/data/data/cocotalk_2.json', 'w')
json.dump(info, f)
