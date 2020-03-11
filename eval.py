from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import sys, codecs
# sys.stdout = codecs.getwriter("utf-8")(sys.stdout)

import time
import os
import _pickle as cPickle
# from six.moves import cPickle

import opts
import models
from dataloader import *
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch
import similality.cos_distance as sim
import Discriminator.utils as dis_utils
import Discriminator.dataloader_for_dis as dis_dataloader

# Input arguments and options
parser = argparse.ArgumentParser()

dir = '/mnt/poplin/share/2018/nakamura_M1/self_sequential/data/'
# Input paths
parser.add_argument('--model', type=str, default='',
                help='path to model to evaluate')
parser.add_argument('--discriminator', type=str, default=None,
                help='path to model to evaluate')
parser.add_argument('--manager', type=str, default='None',
                help='path to manger to evaluate')
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='',
                help='path to infos to evaluate')
# Basic options
parser.add_argument('--batch_size', type=int, default=0,
                help='if > 0 then overrule, otherwise load from checkpoint.')
parser.add_argument('--num_images', type=int, default=-1,
                help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser.add_argument('--language_eval', type=int, default=0,
                help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
parser.add_argument('--dump_images', type=int, default=1,
                help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
parser.add_argument('--dump_json', type=int, default=1,
                help='Dump json with predictions into vis folder? (1=yes,0=no)')
parser.add_argument('--dump_path', type=int, default=0,
                help='Write image paths along with predictions into vis json? (1=yes,0=no)')

# Sampling options
parser.add_argument('--sample_max', type=int, default=1,
                help='1 = sample argmax words. 0 = sample from distributions.')
parser.add_argument('--max_ppl', type=int, default=0,
                help='beam search by max perplexity or max probability.')
parser.add_argument('--beam_size', type=int, default=2,
                help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
parser.add_argument('--group_size', type=int, default=1,
                help='used for diverse beam search. if group_size is 1, then it\'s normal beam search')
parser.add_argument('--diversity_lambda', type=float, default=0.5,
                help='used for diverse beam search. Usually from 0.2 to 0.8. Higher value of lambda produces a more diverse list')
parser.add_argument('--temperature', type=float, default=1.0,
                help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
parser.add_argument('--decoding_constraint', type=int, default=0,
                help='If 1, not allowing same word in a row')
# For evaluation on a folder of images:
parser.add_argument('--image_folder', type=str, default='', 
                help='If this is nonempty then will predict on the images in this folder path')
parser.add_argument('--image_root', type=str, default='', 
                help='In case the image paths have to be preprended with a root path to an image folder')
# For evaluation on MSCOCO images from some split:
parser.add_argument('--input_fc_dir', type=str, default=dir + 'data/cocotalk_fc',
                    help='path to the directory containing the preprocessed fc feats')
parser.add_argument('--input_att_dir', type=str, default=dir + 'data/cocotalk_att',
                help='path to the directory containing the preprocessed att feats')
parser.add_argument('--input_box_dir', type=str, default=dir + 'data/cocotalk_box',
                help='path to the directory containing the boxes of att feats')
parser.add_argument('--input_label_h5', type=str, default=dir + 'data/coco_label.h5',
                    help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_json', type=str, default='', 
                help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
parser.add_argument('--input_bu_feature', type=str, default='/mnt/poplin/share/2018/nakamura_M1/self_sequential/data/data/bbox_info.json',
                        help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_subatt_dir', type=str, default=dir + 'data/cocotalk_att',
                        help='path to the directory containing the preprocessed att feats')
parser.add_argument('--split', type=str, default='test',
                help='if running on MSCOCO images, which split to use: val|test|train')
parser.add_argument('--coco_json', type=str, default='', 
                help='if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')
parser.add_argument('--sim_pred_type', type=int, default=2, help='select similarity predictor')
# misc
parser.add_argument('--id', type=str, default='', 
                help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
parser.add_argument('--verbose_beam', type=int, default=1, 
                help='if we need to print out all beam search beams.')
parser.add_argument('--verbose_loss', type=int, default=0, 
                help='if we need to calculate loss.')
parser.add_argument('--attention_output', type=bool, default=False,
                help='if we need to output attention.')

parser.add_argument('--internal_model', type=str, default='',
                help='P or R')
parser.add_argument('--internal_dir', type=str, default='')
parser.add_argument('--output', type=int, default=0)
parser.add_argument('--seq_length', type=int, default=20)
parser.add_argument('--sub_seq_flg', type=int, default=0)
parser.add_argument('--region_bleu_flg', type=int, default=0)
parser.add_argument('--dataset', type=str, default='coco')
parser.add_argument('--use_weight_probability', type=int, default=0)
parser.add_argument('--max_att_len', type=int, default=36)
parser.add_argument('--weight_deterministic_flg', type=int, default=0)
parser.add_argument('--whole_att_flg', type=int, default=0, help='To evaluate baseline method, whole image feature is used as a attention')
parser.add_argument('--baseline_concat', type=int, default=0)

parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--prohibit_flg', type=int, default=0, help='use prohibition of return saliency')
parser.add_argument('--prohibit_flg_hard', type=int, default=0, help='use prohibition of return saliency for hard attention')

parser.add_argument('--min_seq_length', type=int, default=-1,
                        help='minimun number of sequrnse length')
parser.add_argument('--bleu_option', type=str, default='closest',
                        help='closest ot times')
parser.add_argument('--p_switch', type=int, default=0)
parser.add_argument('--area_feature_use', type=int, default=0)
parser.add_argument('--selected_region_file', type=str, default=None)
parser.add_argument('--use_next_region', type=int, default=0)

opt = parser.parse_args()

# opt.model = '/mnt/poplin/share/2018/nakamura_M1/self_sequential/log/' + opt.model
# opt.infos_path ='/mnt/poplin/share/2018/nakamura_M1/self_sequential/log/' + opt.infos_path
# opt.manager = '/mnt/poplin/share/2018/nakamura_M1/self_sequential/log/' + opt.manager
# opt.internal_dir = '/mnt/poplin/share/2018/nakamura_M1/self_sequential/log/' + opt.internal_dir

mnt_path = '/mnt/workspace2019/nakamura/selfsequential/log_python3/'

opt.model = mnt_path + opt.model
if opt.discriminator is not None:
    opt.discriminator = mnt_path + opt.discriminator
opt.sim_model_dir = mnt_path + 'log_' + opt.id + '/sim_model' + opt.model[-13:-4] + '.pth'
opt.infos_path = mnt_path + opt.infos_path
# opt.manager = '/mnt/poplin/share/2018/nakamura_M1/self_sequential/log/' + opt.manager
opt.internal_dir = mnt_path + opt.internal_dir

torch.cuda.set_device(opt.gpu)

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

# Load infos
# with open(opt.infos_path) as f:
#     infos = cPickle.load(f)
with open(opt.infos_path, mode='rb') as f:
    infos = cPickle.load(f, encoding='latin1')

# with open('/mnt/workspace2018/nakamura/IAPR/iapr_talk_mod.json', mode='rb')
#     infos = cPickle.load(f, encoding='latin1')

# override and collect parameters
if len(opt.input_fc_dir) == 0:
    opt.input_fc_dir = infos['opt'].input_fc_dir
    opt.input_att_dir = infos['opt'].input_att_dir
    # opt.input_box_dir = infos['opt'].input_box_dir
    opt.input_label_h5 = infos['opt'].input_label_h5
if len(opt.input_json) == 0:
    opt.input_json = infos['opt'].input_json
if opt.batch_size == 0:
    opt.batch_size = infos['opt'].batch_size
if len(opt.id) == 0:
    opt.id = infos['opt'].id
# opt.input_bu_feature = infos['opt'].input_bu_feature
opt.sum_reward_rate = 1
# ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval","gpu"]
ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval","gpu", "seq_length", "dataset", 'region_bleu_flg',
          'input_label_h5', "input_box_dir", "input_json", "internal_model", "prohibit_flg_hard", "input_bu_feature", "sum_reward_rate"]
for k in vars(infos['opt']).keys():
    if k not in ignore:
        if k in vars(opt):
            # pdb.set_trace()
            print('-----')
            print(k)
            print(vars(opt)[k])
            print(vars(infos['opt'])[k])
            print('-----')
            # assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
        else:
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

vocab = infos['vocab'] # ix -> word mapping

opt.multi_learn_flg = 0
opt.critic_probabilistic = 0
opt.ppo_flg = 0
opt.ppo = 0
opt.critic_encode = 0
opt.actor_critic_flg = 0

# Setup the model
print('Setup the model')
model = models.setup(opt)
model.load_state_dict(torch.load(opt.model, map_location='cuda:0'))
model.cuda()
model.eval()

# if infos['opt'].discriminator_weight > 0:
#     Discriminator = dis_utils.Discriminator(infos['opt'])
#     # discrimiantor_model_dir = '/mnt/workspace2018/nakamura/selfsequential/discriminator_log/coco/discriminator_150.pth'
#     discrimiantor_model_dir = '/mnt/workspace2018/nakamura/selfsequential/discriminator_log/iapr_dict/discriminator_125.pth'
#     Discriminator.load_state_dict(torch.load(discrimiantor_model_dir, map_location='cuda:' + str(opt.gpu)))
#     Discriminator = Discriminator.cuda()
#     Discriminator.eval()
#
#     Discriminator_learned = dis_utils.Discriminator(infos['opt'])
#     discrimiantor_model_dir = opt.discriminator
#     Discriminator_learned.load_state_dict(torch.load(discrimiantor_model_dir, map_location='cuda:' + str(opt.gpu)))
#     Discriminator_learned = Discriminator_learned.cuda()
#     Discriminator_learned.eval()
# else:
#     Discriminator = None
#     Discriminator_learned = None



if opt.discriminator is not None:
    infos['opt'].cut_length = 5
    Discriminator = dis_utils.Discriminator(infos['opt'])
    # discrimiantor_model_dir = '/mnt/workspace2018/nakamura/selfsequential/discriminator_log/coco/discriminator_150.pth'
    # discrimiantor_model_dir = '/mnt/workspace2019/nakamura/selfsequential/discriminator_log/sew/discriminator_900.pth'
    discrimiantor_model_dir = '/mnt/workspace2019/nakamura/selfsequential/discriminator_log/vg_cut5/discriminator_200.pth'
    Discriminator.load_state_dict(torch.load(discrimiantor_model_dir, map_location='cuda:' + str(opt.gpu)))
    Discriminator = Discriminator.cuda()
    Discriminator.eval()

    Discriminator_learned = dis_utils.Discriminator(infos['opt'])
    discrimiantor_model_dir = opt.discriminator
    if not os.path.isfile(discrimiantor_model_dir):
        # discrimiantor_model_dir = '/mnt/workspace2019/nakamura/selfsequential/log_python3/log_xe_with_disc500_vg_woic_wdf4_rand/discriminator_30_48000.pth'
        discrimiantor_model_dir = '/mnt/workspace2019/nakamura/selfsequential/log_python3/log_xe_with_disc500_vg_simple_wdf5_lr01_rand/discriminator_30_48000.pth'
    # discrimiantor_model_dir = '/mnt/workspace2019/nakamura/selfsequential/log_python3/log_xe_wirh_disc1000_vg_woic_limitlength_spuls/discriminator_36_57000.pth'
    # discrimiantor_model_dir = '/mnt/workspace2019/nakamura/selfsequential/log_python3/log_xe_wirh_disc1000_vg_woic_limitlength/discriminator_42_66000.pth'
    Discriminator_learned.load_state_dict(torch.load(discrimiantor_model_dir, map_location='cuda:' + str(opt.gpu)))
    Discriminator_learned = Discriminator_learned.cuda()
    Discriminator_learned.eval()
else:
    Discriminator = None
    Discriminator_learned = None

# if opt.manager != 'manager_model':
#     if opt.manager_model == 'manager':
#         manager = models.ManagerModel(opt)
#     elif opt.manager_model == 'manager_lstm':
#         manager = models.ManagerModel_lstm(opt)
#     elif opt.manager_model == 'manager_fc':
#         manager = models.ManagerModel_fc(opt)
#     if vars(opt).get('start_from', None) is not None:
#         # check if all necessary files exist
#         print('manager_load')
#         assert os.path.isdir(opt.start_from), " %s must be a a path" % opt.start_from
#         assert os.path.isfile(os.path.join(opt.start_from,
#                                            "infos_" + opt.id + ".pkl")), "infos.pkl file does not exist in path %s" % opt.start_from

if infos['opt'].caption_model == 'hcatt_hard' or infos['opt'].caption_model == 'hcatt_hard_nregion' or \
                infos['opt'].caption_model == 'basicxt_hard_nregion':
    crit = utils.LanguageModelCriterion_hard()
else:
    crit = utils.LanguageModelCriterion()
opt.sim_model = None


# Create the Data Loader instance
print('Create the Data Loader instance')
if len(opt.image_folder) == 0:
  loader = DataLoader(opt)
else:
  loader = DataLoaderRaw({'folder_path': opt.image_folder, 
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size,
                            'cnn_model': opt.cnn_model})
  print('Data load!')
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.

loader.ix_to_word = infos['vocab']

if opt.internal_model == 'sim' or opt.internal_model == 'sim_dammy' or infos['opt'].caption_model == 'hcatt_hard':
    sim_model = sim.Sim_model(opt.input_encoding_size, opt.rnn_size, vocab_size=loader.vocab_size)
    if opt.sim_pred_type == 0:
        # model_root = '/mnt/workspace2018/nakamura/vg_feature/model_cossim2/model_13_1700.pt'
        model_root = '/mnt/workspace2018/nakamura/vg_feature/model_cossim_bu/model_6_0.pt'
    elif opt.sim_pred_type == 1:
        model_root = '/mnt/workspace2018/nakamura/vg_feature/model_cossim_noshuffle04model_71_1300.pt'
    elif opt.sim_pred_type == 2:
        model_root = opt.sim_model_dir

    try:
        checkpoint = torch.load(model_root, map_location='cuda:0')
        sim_model.load_state_dict(checkpoint['model_state_dict'])
    except KeyError:
        sim_model.load_state_dict(torch.load(model_root, map_location='cuda:' + str(opt.gpu)))
    sim_model.cuda()
    sim_model.eval()
    opt.sim_model = sim_model
    for param in sim_model.parameters():
        param.requires_grad = False

    if infos['opt'].caption_model == 'hcatt_hard':
        internal = None
    else:
        internal = models.CriticModel_sim(opt)
        internal = internal.cuda()
        if opt.internal_dir != mnt_path:
            internal.load_state_dict(torch.load(opt.internal_dir, map_location='cuda:0'))
        internal.eval()

else:
    internal = None

if infos['opt'].caption_model == 'hcatt_hard':
    opt.sim_reward_flg = 1

opt.internal = internal

# /mnt/workspace2018/nakamura/selfsequential/vis

import pickle
# Set sample options
print('Set sample options')
if opt.output == 1:
    splits = ['test']
    for split in splits:
        opt.split = split
        loss, split_predictions, lang_stats = eval_utils.eval_split_output(model, crit, loader, vars(opt), Discriminator=Discriminator, Discriminator_learned=Discriminator_learned)
        # with open('/mnt/poplin/share/2018/nakamura_M1/self_sequential/vis/vis_hcatt_04.json', 'w') as f:
        # print(model.training, internal.training, sim_model.training)

        dir = '/mnt/workspace2018/nakamura/selfsequential/vis'
        if not os.path.isdir(dir):
            os.mkdir(dir)
        if opt.beam_size > 1:
            filename = opt.id + '_' + opt.model[-13:-4] + '_py3_' + 'beam' + str(opt.beam_size)
        # else:
        #     filename = opt.id + '_' + opt.model[-13:-4] + '_urs'
        elif opt.baseline_concat == 1:
            filename = opt.id + '_' + opt.model[-13:-4] + '_py3_baselineA_lessthan08_wbf5'
        elif opt.whole_att_flg == 1:
            filename = opt.id + '_' + opt.model[-13:-4] + '_py3_baselineB_lessthan08_wbf5'
        elif opt.internal_model == 'sim':
            filename = opt.id + '_' + opt.model[-13:-4] + '_py3_nouselessthan08'
        else:
            filename = opt.id + '_' + opt.model[-13:-4] + '_py3_re'
            # filename = opt.id + '_' + opt.model[-13:-4] + '_att_fasten'

        if opt.language_eval == 1:
            text = eval_utils.write_record(infos['opt'].id + '_' + opt.model[-13:-4], split_predictions, lang_stats)
            with open(dir + '/result_' + filename + '_' + opt.dataset + '.txt', 'w') as f:
                f.write(text)

        with open(dir + '/' + filename + '_' + split + '.json', 'wb') as f:
            pickle.dump(split_predictions, f)
        print('json are saved to ' +  dir + '/' + filename + '_' + split + '.json')
    exit()
else:
    loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader, vars(opt))

print('loss: ', loss)
if lang_stats:
  print(lang_stats)

if opt.dump_json == 1:
    # dump the json
    json.dump(split_predictions, open('vis/vis.json', 'w'))
