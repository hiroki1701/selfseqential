import torch
import torch.nn as nn
from torch.optim import SGD
import math
import numpy as np
import torch
import torch.nn as nn
import json
import pdb
import torch.optim as optim
import torch.nn.functional as F
import argparse
from dataloader import *
import pdb
import opts
import sys
sys.path.append('/home/nakamura/project/self_seqential/vg/')
import visual_genome.local as vg
# import vg.visual_genome.local as vg
dir = '/mnt/poplin/share/dataset/visualgenome'
print('visual genome data loading...')
all_images = vg.get_all_image_data(dir)
all_discriptions = vg.get_all_region_descriptions(dir)
print('visual genome data loaded !')

class CriticModel_vg(nn.Module):
    def __init__(self, opt):
        super(CriticModel_vg, self).__init__()
        self.critic_lay_1 = nn.Linear(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)
        self.critic_lay_2 = nn.Linear(opt.rnn_size,opt.rnn_size)
        self.critic_lay_3 = nn.Linear(opt.rnn_size, 2)
        self.embed = nn.Sequential(nn.Embedding(9487 + 1, 512),
                                   nn.ReLU(),
                                   nn.Dropout(0.1))

    def forward(self,prev_h, pre_att, inputs):
        xt = self.embed(inputs)
        input = torch.cat([prev_h, pre_att, xt], 1)
        x = self.critic_lay_1(input)
        x = F.dropout(self.critic_lay_2(x))
        output = self.critic_lay_3(x)

        return output

def train(opt):
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    info = json.load(open('/mnt/poplin/share/2018/nakamura_M1/self_sequential/data/data/cocotalk_2.json'))
    opt.cocoid_to_id = info['cocoid_to_id']
    opt.word_to_ix = info['word_to_ix']

    epoch = 100
    loader.reset_iterator('vg')
    for i in range(epoch):
        data = loader.get_batch('vg')
        coco_ids = []
        vg_ids = []
        discriptions = []
        for id_info in data['infos']:
            coco_ids.append(id_info['id'])
            vg_ids.append(opt.cocoid_to_id[str(id_info['id'])])
            discriptions.append(all_discriptions[opt.cocoid_to_id[str(id_info['id'])]])

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        fc_feats, att_feats, labels, masks, att_masks = tmp

        pdb.set_trace()



    # model = model.train().cuda()
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters())


opt = opts.parse_opt()
train(opt)

