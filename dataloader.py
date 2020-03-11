from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random
from functools import reduce

import torch
import torch.utils.data as data

import multiprocessing
import copy
import pdb

class DataLoader(data.Dataset):

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split, self, split=='train')
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = opt.seq_per_img
        
        # feature related options
        self.use_att = getattr(opt, 'use_att', True)
        self.use_box = getattr(opt, 'use_box', 0)
        self.norm_att_feat = getattr(opt, 'norm_att_feat', 0)
        self.norm_box_feat = getattr(opt, 'norm_box_feat', 0)
        self.bbox_flg = 0
        self.max_att_len = opt.max_att_len
        self.weight_deterministic_flg = opt.weight_deterministic_flg
        if opt.caption_model == 'hcatt_hard' and opt.sum_reward_rate > 0:
            self.bbox_flg = 1
        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)

        self.info = json.load(open(self.opt.input_json))

        # pdb.set_trace()
        if opt.selected_region_file is not None:
            if os.path.exists(opt.selected_region_file):
                selected_region_info_json = json.load(open(opt.selected_region_file))
                selected_region_info = selected_region_info_json['acrions']
                selected_region_flg = selected_region_info_json['flg']

                if len(self.info['images']) <= len(selected_region_info):
                    for i in range(len(self.info['images'])):
                        self.info['images'][i]['selected_id_frn'] = selected_region_info[i]
                        if selected_region_flg[i] == 0 and self.weight_deterministic_flg == 5:
                            self.info['images'][i]['split'] = 'no_use'

        self.ix_to_word = self.info['ix_to_word']
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is ', self.vocab_size)
        # open the hdf5 file
        print('DataLoader loading h5 file: ', opt.input_fc_dir, opt.input_att_dir, opt.input_box_dir, opt.input_label_h5)
        self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')

        self.input_fc_dir = self.opt.input_fc_dir
        self.input_att_dir = self.opt.input_att_dir
        self.input_box_dir = self.opt.input_box_dir
        if self.opt.input_bu_feature is not None:
            self.input_bu_feature = json.load(open(self.opt.input_bu_feature))
        else:
            self.input_bu_feature = None
        self.input_subatt_dir = self.opt.input_subatt_dir

        # load in the sequence data
        seq_size = self.h5_label_file['labels'].shape
        # if opt.seq_length < seq_size[1]:
        #     self.seq_length = seq_size[1]
        # else:
        #     self.seq_length = opt.seq_length
        if opt.seq_length > 0:
            self.seq_length = opt.seq_length
        else:
            self.seq_length = seq_size[1]
        print('max sequence length in data is', self.seq_length)
        # load the pointers in full to RAM (should be small enough)
        self.label_start_ix = self.h5_label_file['label_start_ix'][:]
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]
        try:
            self.label_start_ix_er = self.h5_label_file['label_start_ix_er'][:]
            self.label_end_ix_er = self.h5_label_file['label_end_ix_er'][:]
        except KeyError:
            self.label_start_ix_er = None
            self.label_end_ix_er = None

        self.num_images = self.label_start_ix.shape[0]
        print('read %d image features' %(self.num_images))

        # separate out indexes for each of the provided splits
        if self.opt.input_json == '/mnt/poplin/share/2018/nakamura_M1/self_sequential/data/data/cocotalk_2.json':
            self.cocoid_to_id = self.info['cocoid_to_id']
            self.split_ix = {'train': [], 'val': [], 'test': [], 'vg':[], 'nouse':[]}
            for ix in range(len(self.info['images'])):
                img = self.info['images'][ix]
                if img['split'] == 'train':
                    self.split_ix['train'].append(ix)
                elif img['split'] == 'val':
                    self.split_ix['val'].append(ix)
                elif img['split'] == 'test':
                    self.split_ix['test'].append(ix)
                elif img['split'] == 'nouse':
                    self.split_ix['nouse'].append(ix)
                elif opt.train_only == 0:  # restval
                    self.split_ix['train'].append(ix)

                flg = self.cocoid_to_id.get(str(img['id']))
                if flg is not None and img['split'] == 'train':
                    self.split_ix['vg'].append(ix)
            print('assigned %d images to split train' % len(self.split_ix['train']))
            print('assigned %d images to split val' % len(self.split_ix['val']))
            print('assigned %d images to split test' % len(self.split_ix['test']))
            print('assigned %d images to split nouse' % len(self.split_ix['nouse']))
            print('assigned %d images to split vg' % len(self.split_ix['vg']))
            self.iterators = {'train': 0, 'val': 0, 'test': 0, 'vg': 0}
        else:
            self.split_ix = {'train': [], 'val': [], 'test': [], 'vg': [], 'nouse': []}
            for ix in range(len(self.info['images'])):
                img = self.info['images'][ix]
                if img['split'] == 'train':
                    self.split_ix['train'].append(ix)
                elif img['split'] == 'val':
                    self.split_ix['val'].append(ix)
                elif img['split'] == 'test':
                    self.split_ix['test'].append(ix)
                elif img['split'] == 'nouse':
                    self.split_ix['nouse'].append(ix)
                elif opt.train_only == 0: # restval
                    self.split_ix['train'].append(ix)
            print('assigned %d images to split train' % len(self.split_ix['train']))
            print('assigned %d images to split val' % len(self.split_ix['val']))
            print('assigned %d images to split test' % len(self.split_ix['test']))
            print('assigned %d images to split nouse' % len(self.split_ix['nouse']))
            self.iterators = {'train': 0, 'val': 0, 'test': 0}

        # '/mnt/workspace2018/nakamura/selfsequential/data/region_labels.json'
        self.region_labels_json = json.load(open('/mnt/workspace2018/nakamura/selfsequential/data/region_labels_all.json'))
        self.region_labels = self.region_labels_json

        # self.region_labels = []
        # for i in range(len(self.region_labels_json)):
        #     region_labels_element = np.zeros((self.max_att_len, 50))
        #     for j in range(min(len(self.region_labels_json[i]), len(region_labels_element))):
        #         sent_ix = self.region_labels_json[i][j]['ix']
        #         region_labels_element[j] += np.array(sent_ix).astype(np.int32)
        #     self.region_labels.append(np.array(region_labels_element))
        # self.region_labels = np.array(self.region_labels)

        self._prefetch_process = {} # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split=='train')
            # Terminate the child process when the parent exists

        self.region_bleu_flg = opt.region_bleu_flg

        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]
        import atexit
        atexit.register(cleanup)

    def get_captions(self, ix, seq_per_img, weight_index, tmp_att, split, weight_id_info=None):
        # fetch the sequence labels
        ix1 = self.label_start_ix[ix] - 1 #label_start_ix starts from 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1 # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        # 1: select 1 area, 2: selsect 2~4area, 4: 4area, 5: use_selsected_area and swich every 5 word, 6: random and swich every 5 word
        if self.weight_deterministic_flg == 2 or self.weight_deterministic_flg == 4:
            if split != 'train':
                np.random.seed(seed=32)
            max_region = 4
            seq = np.zeros([seq_per_img, self.seq_length], dtype='int')
            for q in range(seq_per_img):
                if self.weight_deterministic_flg == 2:
                    use_region = np.random.randint(max_region - 1) + 2
                    # use_region = 3
                else:
                    use_region = 4
                pre_ixl = -1
                switch_num = 0
                weight_ids = []
                switch_nums = []
                q_i = 0
                while self.seq_length - switch_num > 0 and q_i < use_region:
                    ixl = np.random.randint(self.label_start_ix_er[ix].shape[0])
                    # ixl = np.random.randint(tmp_att.shape[0])
                    while self.label_start_ix_er[ix, ixl] == 0 or ixl == pre_ixl:
                        ixl = np.random.randint(self.label_start_ix_er[ix].shape[0])

                    seq_cand = self.h5_label_file['labels'][
                               self.label_start_ix_er[ix, ixl] - 1: self.label_end_ix_er[ix, ixl]]
                    ixl_2 = np.random.randint(len(seq_cand))
                    seq[q, switch_num:switch_num + len(seq_cand[ixl_2])] = seq_cand[ixl_2, :(self.seq_length - switch_num)]

                    weight_ids.append(ixl)
                    switch_nums.append(switch_num)
                    pre_ixl = ixl
                    switch_num = np.sum(seq[q]>0)
                    q_i += 1
                while len(weight_ids) < max_region:
                    weight_ids.append(-1)
                    switch_nums.append(self.seq_length)
                weight_index.append([weight_ids, switch_nums])
        elif self.weight_deterministic_flg == 1 or self.weight_deterministic_flg == 3:
            cap_exist = (self.label_start_ix_er[ix]>0).sum()
            if split != 'train':
                np.random.seed(seed=32)
            max_region = 1
            seq = np.zeros([seq_per_img, self.seq_length], dtype='int')
            for q in range(seq_per_img):
                use_region = 1
                pre_ixl = -1
                switch_num = 0
                weight_ids = []
                switch_nums = []
                q_i = 0
                while self.seq_length - switch_num > 0 and q_i < use_region:
                    ixl = np.random.randint(self.label_start_ix_er[ix].shape[0])
                    # ixl = np.random.randint(tmp_att.shape[0])
                    while self.label_start_ix_er[ix, ixl] == 0 or ixl == pre_ixl:
                        ixl = np.random.randint(self.label_start_ix_er[ix].shape[0])

                    seq_cand = self.h5_label_file['labels'][
                               self.label_start_ix_er[ix, ixl] - 1: self.label_end_ix_er[ix, ixl]]
                    ixl_2 = np.random.randint(len(seq_cand))
                    seq[q, switch_num:switch_num + len(seq_cand[ixl_2])] = seq_cand[ixl_2, :(self.seq_length - switch_num)]

                    weight_ids.append(ixl)
                    # switch_nums.append(self.seq_length)
                    switch_nums.append(0)
                    pre_ixl = ixl
                    switch_num = self.seq_length
                    q_i += 1
                while len(weight_ids) < max_region:
                    weight_ids.append(-1)
                    switch_nums.append(0)
                weight_index.append([weight_ids, switch_nums])
        elif self.weight_deterministic_flg == 7 or self.weight_deterministic_flg == 8 or self.weight_deterministic_flg == 9:
            # we need to subsample (with replacement)
            seq = np.zeros([seq_per_img, self.seq_length], dtype='int')
            if split != 'train':
                np.random.seed(seed=32)
            # weight_id_info: [id_1, id_2, .., id_max]
            use_region = 0
            switch_num = 0
            weight_ids = []
            switch_nums = []
            max_region = len(weight_id_info)
            for i in range(max_region):
                weight_ids.append(weight_id_info[i])
                if weight_id_info[i] != -1:
                    if self.label_start_ix_er is not None:
                        ixl = weight_id_info[i]
                        seq_cand = self.h5_label_file['labels'][
                                   self.label_start_ix_er[ix, ixl] - 1: self.label_end_ix_er[ix, ixl]]
                        ixl_2 = np.random.randint(len(seq_cand))


                        if (self.weight_deterministic_flg == 8 or self.weight_deterministic_flg == 9) and i > 0:
                            if switch_num + 1 < self.seq_length:
                                seq[:, switch_num+1:switch_num + len(seq_cand[ixl_2])+1] = seq_cand[ixl_2, :(
                                self.seq_length - switch_num - 1)].reshape(
                                    (1, -1)).repeat(5, axis=0)
                        else:
                            if switch_num < self.seq_length:
                                seq[:, switch_num:switch_num + len(seq_cand[ixl_2])] = seq_cand[ixl_2, :(
                                    self.seq_length - switch_num)].reshape(
                                    (1, -1)).repeat(5, axis=0)
                        switch_nums.append(switch_num)
                        if split == 'train':
                            if self.weight_deterministic_flg == 8 and i == 0:
                                switch_num += np.sum(seq_cand[ixl_2] > 0)
                            elif (self.weight_deterministic_flg == 9 and i == 0) or self.weight_deterministic_flg != 9:
                                switch_num += np.sum(seq_cand[ixl_2] > 0) + 1
                            else:
                                switch_num += np.sum(seq_cand[ixl_2] > 0) + 2
                        else:
                            switch_num = np.sum(5 * (i + 1))
                    else:
                        switch_nums.append(5 * (i))
                    use_region += 1
                else:
                    switch_nums.append(switch_num)
            for i in range(seq_per_img):
                weight_index.append([weight_ids, switch_nums])

        elif self.weight_deterministic_flg == 0:
            if ncap < seq_per_img:
                # we need to subsample (with replacement)
                seq = np.zeros([seq_per_img, self.seq_length], dtype = 'int')
                for q in range(seq_per_img):
                    ixl = random.randint(ix1,ix2)
                    seq[q, :] = self.h5_label_file['labels'][ixl, :self.seq_length]
                    # seq[q, :] = self.h5_label_file['labels_50'][ixl, :self.seq_length]
            else:
                ixl = random.randint(ix1, ix2 - seq_per_img + 1)
                seq = self.h5_label_file['labels'][ixl: ixl + seq_per_img, :self.seq_length]

        else:
            seq = np.zeros([seq_per_img, self.seq_length], dtype='int')
            if self.weight_deterministic_flg == 5 or self.weight_deterministic_flg == 6:
                if split != 'train':
                    np.random.seed(seed=32)
                # weight_id_info: [id_1, id_2, .., id_max]
                max_region = 3
                use_region = 0
                switch_num = 0
                weight_ids = []
                switch_nums = []
                if self.weight_deterministic_flg == 5 and weight_id_info is not None:
                    max_region = len(weight_id_info)
                    for i in range(max_region):
                        weight_ids.append(weight_id_info[i])
                        if weight_id_info[i] != -1:
                            if self.label_start_ix_er is not None:
                                ixl = weight_id_info[i]
                                seq_cand = self.h5_label_file['labels'][
                                           self.label_start_ix_er[ix, ixl] - 1: self.label_end_ix_er[ix, ixl]]
                                ixl_2 = np.random.randint(len(seq_cand))
                                if switch_num < self.seq_length:
                                    seq[:, switch_num:switch_num + len(seq_cand[ixl_2])] = seq_cand[ixl_2, :(self.seq_length - switch_num)].reshape(
                                        (1, -1)).repeat(5, axis=0)
                                switch_nums.append(switch_num)
                                if split == 'train':
                                    switch_num = np.sum(seq[0] > 0)
                                else:
                                    switch_num = np.sum(5 * (i + 1))
                            else:
                                switch_nums.append(5 * (i))
                            use_region += 1
                        else:
                            switch_nums.append(switch_num)

                elif self.weight_deterministic_flg == 6:
                    for i in range(max_region):
                        weight_ids.append(np.random.randint(low=0, high=36))
                        switch_nums.append(5 * (i))
                else:
                    pdb.set_trace()
                for i in range(seq_per_img):
                    weight_index.append([weight_ids, switch_nums])

        return seq, weight_index

    def get_batch(self, split, batch_size=None, seq_per_img=None):
        batch_size = batch_size or self.batch_size
        seq_per_img = seq_per_img or self.seq_per_img

        fc_batch = [] # np.ndarray((batch_size * seq_per_img, self.opt.fc_feat_size), dtype = 'float32')
        att_batch = [] # np.ndarray((batch_size * seq_per_img, 14, 14, self.opt.att_feat_size), dtype = 'float32')
        label_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype = 'int')
        mask_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype = 'float32')
        label_regions = []

        wrapped = False

        infos = []
        gts = []
        gts_each_region_all = []
        weight_index = []
        fixed_region_coordinate = []

        for i in range(batch_size):
            # fetch image
            tmp_fc, tmp_att,\
                ix, tmp_wrapped = self._prefetch_process[split].get()
            if 'selected_id_frn' in self.info['images'][ix].keys():
                weight_id_info = self.info['images'][ix]['selected_id_frn']
            else:
                weight_id_info = None

            # pdb.set_trace()
            if weight_id_info is not None:
                weight_id_info = weight_id_info[:10]

            label_batch[i * seq_per_img : (i + 1) * seq_per_img, 1 : self.seq_length + 1], weight_index = \
                self.get_captions(ix, seq_per_img, weight_index, tmp_att, split, weight_id_info=weight_id_info)
            if self.weight_deterministic_flg == 3:
                tmp_fc = tmp_att[weight_index[-1 * seq_per_img:]]

            fc_batch.append(tmp_fc)
            att_batch.append(tmp_att)
            if tmp_wrapped:
                wrapped = True

            # if (self.region_bleu_flg == 1 or self.opt.wbleu_reward_weight > 0) and self.label_start_ix_er is not None:
            if False:
                label_regions.append(self.region_labels[ix])
            # Used for reward evaluation
            # gts.append(self.h5_label_file['labels_50'][self.label_start_ix[ix] - 1: self.label_end_ix[ix]])
            gts.append(self.h5_label_file['labels'][self.label_start_ix[ix] - 1: self.label_end_ix[ix]])

            # Use for weighted bleu
            if (self.region_bleu_flg == 1 or self.opt.wbleu_reward_weight > 0) and self.label_start_ix_er is not None:
                gts_each_region = []
                for j in range(self.label_start_ix_er[ix].shape[0]):
                    if self.label_start_ix_er[ix, j] > 0:
                        gts_each_region.append(self.h5_label_file['labels'][self.label_start_ix_er[ix, j] - 1: self.label_end_ix_er[ix, j]])
                gts_each_region_copy = copy.copy(gts_each_region)
                gts_each_region_all.append(gts_each_region_copy)

            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            # pdb.set_trace()
            # info_dict['file_path'] = self.info['images'][ix]['file_path']
            if 'fixed_region' in self.info['images'][ix].keys():
                for j in range(seq_per_img):
                    fixed_region_np = np.zeros((self.max_att_len, 6)).astype(np.int32)
                    att_len = min(len(self.info['images'][ix]['fixed_region']), self.max_att_len)
                    fixed_region_np[:att_len] += np.array(self.info['images'][ix]['fixed_region'])[:att_len]
                    fixed_region_coordinate.append(fixed_region_np)

            infos.append(info_dict)

        # #sort by att_feat length
        # fc_batch, att_batch, label_batch, gts, infos = \
        #     zip(*sorted(zip(fc_batch, att_batch, np.vsplit(label_batch, batch_size), gts, infos), key=lambda x: len(x[1]), reverse=True))
        if len(gts_each_region_all) == 0:
            fc_batch, att_batch, label_batch, gts, infos = \
                zip(*sorted(zip(fc_batch, att_batch, np.vsplit(label_batch, batch_size), gts, infos), key=lambda x: 0, reverse=True))
        else:
            fc_batch, att_batch, label_batch, gts, gts_each_region_all, infos = \
                zip(*sorted(zip(fc_batch, att_batch, np.vsplit(label_batch, batch_size), gts, gts_each_region_all, infos), key=lambda x: 0,
                            reverse=True))
        data = {}
        if self.weight_deterministic_flg == 3:
            data['fc_feats'] = np.stack(reduce(lambda x, y: x + y, [[_] for _ in fc_batch]))
            data['fc_feats'] = data['fc_feats'].reshape(-1, data['fc_feats'].shape[-1])
        else:
            data['fc_feats'] = np.stack(reduce(lambda x,y:x+y, [[_]*seq_per_img for _ in fc_batch]))

        # merge att_feats
        # max_att_len = max([_.shape[0] for _ in att_batch])
        data['att_feats'] = np.zeros([len(att_batch)*seq_per_img, self.max_att_len, att_batch[0].shape[1]], dtype = 'float32')
        for i in range(len(att_batch)):
            if att_batch[i].shape[0] != self.max_att_len:
                att_batch_ = np.zeros((self.max_att_len, att_batch[i].shape[1]))
                att_batch_[:att_batch[i].shape[0], :] = att_batch_[:att_batch[i].shape[0], :] + att_batch[i][:self.max_att_len, :]
            else:
                att_batch_ = att_batch[i]
            # data['att_feats'][i*seq_per_img:(i+1)*seq_per_img, :att_batch[i].shape[0]] = att_batch[i]
            data['att_feats'][i * seq_per_img:(i + 1) * seq_per_img, :att_batch_.shape[0]] = att_batch_
        data['att_masks'] = np.zeros(data['att_feats'].shape[:2], dtype='float32')

        # if self.weight_deterministic_flg == 1:
        #     index_arange = np.arange(len(data['att_feats']))
        #     data['fc_feats'] = data['att_feats'][index_arange, weight_index]

        for i in range(len(att_batch)):
            data['att_masks'][i*seq_per_img:(i+1)*seq_per_img, :att_batch[i].shape[0]] = 1
        # set att_masks to None if attention features have same length
        # print(data['att_masks'].sum(),data['att_masks'].size)
        if data['att_masks'].sum() == data['att_masks'].size:
            data['att_masks'] = None
        data['att_masks'] = None
        # exit()
        data['labels'] = np.vstack(label_batch)
        data['weight_index'] = weight_index

        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum()+2, data['labels'])))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        data['masks'] = mask_batch

        data['gts'] = gts # all ground truth captions of each images
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        # pdb.set_trace()
        if self.input_bu_feature is not None and self.bbox_flg == 1:
            data['bbox'] = np.zeros((len(att_batch)*seq_per_img, 36 ,196), dtype = 'int32')
            data['sub_att'] = None
            for i in range(len(infos)):
                bbox_weights_id = np.array(self.input_bu_feature[str(infos[i]['id'])]['bbox_fix']) #(36, 4)
                bbox_weights = np.zeros((1, 36, 14, 14))
                for j in range(bbox_weights.shape[1]):
                    bbox_weights[0, j, bbox_weights_id[j, 0]:bbox_weights_id[j, 2], bbox_weights_id[j, 1]:bbox_weights_id[j, 3]] = 1
                bbox_weights = np.reshape(bbox_weights, (1, 36, 196))
                bbox_weights = np.repeat(bbox_weights, seq_per_img, axis=0)
                data['bbox'][i * seq_per_img:(i + 1) * seq_per_img, :, :] = bbox_weights

            # data['sub_att'] = np.zeros((len(att_batch)*seq_per_img, 196, att_batch[0].shape[1]), dtype = 'int32')
            # data['S'] = np.zeros((len(att_batch) * seq_per_img, 36), dtype='float32')
            # for i in range(len(infos)):
                # bbox_weights = np.load(self.input_bu_feature + '/{}.npy'.format(infos[i]['id'])) #(36, 196)
                # bbox_weights = np.resize(bbox_weights, (1, bbox_weights.shape[0], bbox_weights.shape[1]))
                # bbox_weights = np.repeat(bbox_weights, seq_per_img, axis=0)
                #
                # A = bbox_weights.sum(axis=2, keepdims=True)
                # A[A==0] = 1
                # data['bbox'][i * seq_per_img:(i + 1) * seq_per_img, :, :] = bbox_weights/A
                # # data['S'][i * seq_per_img:(i + 1) * seq_per_img, :] = bu_S
                # att_feat_sub = np.load(os.path.join(self.input_subatt_dir, str(self.info['images'][ix]['id']) + '.npz'))['feat']
                # att_feat_sub = np.resize(att_feat_sub,
                #                          (1, att_feat_sub.shape[0]*att_feat_sub.shape[1], att_feat_sub.shape[2]))
                # att_feat_sub = np.repeat(att_feat_sub, seq_per_img, axis=0)
                # data['sub_att'][i * seq_per_img:(i + 1) * seq_per_img, :, :] = att_feat_sub
        else:
            data['bbox'] = None
            data['sub_att'] = None

        data['label_region'] = None
        if (self.region_bleu_flg == 1 or self.opt.wbleu_reward_weight > 0) and self.label_start_ix_er is not None:
            # data['label_region'] = np.array(label_regions)
            data['label_region'] = gts_each_region_all
        else:
            data['label_region'] = None
        if 'fixed_region' in self.info['images'][0].keys():
            data['fixed_region'] = np.array(fixed_region_coordinate)
        else:
            data['fixed_region'] = None
        return data

    # It's not coherent to make DataLoader a subclass of Dataset, but essentially, we only need to implement the following to functions,
    # so that the torch.utils.data.DataLoader can load the data according the index.
    # However, it's minimum change to switch to pytorch data loading.
    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = index #self.split_ix[index]
        if self.use_att:
            att_feat = np.load(os.path.join(self.input_att_dir, str(self.info['images'][ix]['id']) + '.npz'))['feat']
            # print(self.input_att_dir, str(self.info['images'][ix]['id']) + '.npz')
            att_feat = att_feat.reshape(-1, att_feat.shape[-1])
            if self.norm_att_feat:
                att_feat = att_feat / np.linalg.norm(att_feat, 2, 1, keepdims=True)
            if self.use_box:
                box_feat = np.load(os.path.join(self.input_box_dir, str(self.info['images'][ix]['id']) + '.npy'))
                # devided by image width and height
                x1,y1,x2,y2 = np.hsplit(box_feat, 4)
                h,w = self.info['images'][ix]['height'], self.info['images'][ix]['width']
                box_feat = np.hstack((x1/w, y1/h, x2/w, y2/h, (x2-x1)*(y2-y1)/(w*h))) # question? x2-x1+1??
                if self.norm_box_feat:
                    box_feat = box_feat / np.linalg.norm(box_feat, 2, 1, keepdims=True)
                att_feat = np.hstack([att_feat, box_feat])
                # sort the features by the size of boxes
                att_feat = np.stack(sorted(att_feat, key=lambda x:x[-1], reverse=True))
        else:
            att_feat = np.zeros((1,1,1))

        return (np.load(os.path.join(self.input_fc_dir, str(self.info['images'][ix]['id']) + '.npy')),
                att_feat,
                ix)

    def __len__(self):
        return len(self.info['images'])

class SubsetSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, split, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle

    # Add more in the queue
    def reset(self):
        """
        Two cases for this function to be triggered:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 1, the merge is done in DataLoader class
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,
                                            batch_size=1,
                                            sampler=SubsetSampler(self.dataloader.split_ix[self.split][self.dataloader.iterators[self.split]:]),
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=4, # 4 is usually enough
                                            collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped
    
    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()

        assert tmp[2] == ix, "ix not equal"

        return tmp + [wrapped]