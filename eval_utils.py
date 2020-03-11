from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image
import cv2

import torch
import torch.nn as nn
from collections import OrderedDict

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils
import copy
import pdb
from misc.rewards import init_scorer, get_self_critical_reward, get_internal_reward, \
    get_self_critical_and_similarity_reward, get_double_reward, get_attnorm_reward, get_hardatt_reward

def language_eval(dataset, preds, model_id, split, detail_flg=False, wbleu_set=None, option='closest'):
    import sys
    sys.path.append("coco-caption")
    if dataset == 'coco':
        annFile = 'coco-caption/annotations/captions_val2014.json'
    elif dataset == 'vg':
        annFile = '/mnt/poplin/share/dataset/visualgenome/captions_vg.json'
    elif dataset == 'iapr':
        annFile = '/mnt/workspace2018/nakamura/IAPR/captions_iapr.json'

    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes, wbleu_set=wbleu_set)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    if detail_flg:
        out_detail_scores = {}
        # for i in range(len(cocoEval.imgToEval.items())):
        #     pdb.set_trace()
        #     out_detail_scores[str(cocoEval.imgToEval.items()[i][0])] = cocoEval.imgToEval.items()[i][1]
        for key in cocoEval.imgToEval.keys():
            out_detail_scores[str(key)] = cocoEval.imgToEval[key]
        return [out, out_detail_scores]

    return out

def eval_writer(model, iteration, loader, writer, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    internal = eval_kwargs.get('internal', None)


    model.eval()

    loader.reset_iterator(split)

    path_name_v = '/mnt/poplin/share/dataset/MSCOCO/val2014'

    n = 0
    predictions = []
    for i in range(10):
        # get input data
        data = loader.get_batch(split)
        n = n + loader.batch_size

        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data[
                                                                                           'att_masks'] is not None else None]
        tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, att_masks = tmp
        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq = model(fc_feats, att_feats, att_masks, opt=eval_kwargs, mode='sample')[0].data

        # Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(loader.batch_size):
                print('\n'.join(
                    [utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                print('--' * 10)
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        count = 0
        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}

            predictions.append(entry)

            if verbose:
                # pdb.set_trace()
                print('image %s: %s' % (entry['image_id'], entry['caption']))
                id = "COCO_val2014_" + "{0:012d}".format(entry['image_id']) + ".jpg"
                img = cv2.imread(path_name_v + '/' + id)
                img = img[:, :, [2, 1, 0]]
                img = np.transpose(img, (2,0,1))
                img = torch.from_numpy(img)
                txt = str(entry['image_id']) + ': '+ entry['caption'].encode()
                writer.add_text('caption',txt, iteration)
                writer.add_image('image',img, iteration)


def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    internal = eval_kwargs.get('internal', None)
    sim_model = eval_kwargs.get('sim_model', None)
    caption_model = eval_kwargs.get('caption_model', None)
    baseline = eval_kwargs.get('baseline', None)
    gts = eval_kwargs.get('gts', None)
    weight_deterministic_flg = eval_kwargs.get('weight_deterministic_flg', 0)

    # Make sure in the evaluation mode
    model.eval()
    if internal is not None:
        internal.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    predictions_2 = []
    predictions_bleu = {}
    sents_label_eval = {}
    while True:
        # get input data
        data = loader.get_batch(split)
        n = n + loader.batch_size

        if data.get('labels', None) is not None and verbose_loss:
            # forward the model to get loss
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks = tmp

            if weight_deterministic_flg > 0:
                weight_index = np.array(data['weight_index'])
            else:
                weight_index = None

            with torch.no_grad():
                if caption_model != 'hcatt_hard' and caption_model != 'hcatt_hard_nregion'and caption_model != 'basicxt_hard_nregion' and caption_model != 'basicxt_hard':
                    loss = crit(model(fc_feats, att_feats, labels, att_masks,internal, weight_index=weight_index), labels[:,1:], masks[:,1:]).item()
                else:
                    output = model(fc_feats, att_feats, labels, att_masks, internal)
                    loss, baseline = crit(output, labels[:, 1:],
                                masks[:, 1:], baseline, model.weights_p, model.weights)
                    loss = loss.item()
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # data['att_masks'] = None
        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None]
        tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]

        if weight_deterministic_flg > 0:
            weight_index = np.array(data['weight_index'])
            weight_index = weight_index[np.arange(loader.batch_size) * loader.seq_per_img]
        else:
            weight_index = None
        # sents_label = utils.decode_sequence(loader.get_vocab(), labels[:,1:])
    # sents_label_eval = {}
        # for li in range(len(sents_label)//loader.seq_per_img):
        #     block = sents_label[loader.seq_per_img*li:loader.seq_per_img*(li+1)]
        #     sents_label_eval[li] = block

        fc_feats, att_feats, att_masks = tmp
        bbox = None
        sub_att = None



        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq = model(fc_feats, att_feats, att_masks, internal, opt=eval_kwargs, weight_index=weight_index, sim_model=sim_model, bbox=bbox,
                        sub_att=sub_att, label_region=data['label_region'], mode='sample')[0].data
            # Print beam search

        if beam_size > 1 and verbose_beam:
            for i in range(loader.batch_size):
                print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                print('--' * 10)

        sents = utils.decode_sequence(loader.get_vocab(), seq)
        # modify around here

        for i in range(len(data['gts'])):
            gts_sents = utils.decode_sequence(loader.get_vocab(), torch.from_numpy(data['gts'][0].astype(np.int32)))
            sents_label_eval[data['infos'][i]['id']] = gts_sents

        count = 0
        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            entry_ = {'image_id': data['infos'][k]['id'], 'caption': [sent]}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            predictions_2.append(entry_)
            predictions_bleu[data['infos'][k]['id']] = [sent]
            # sents_label_eval[data['infos'][k]['id']] = data['gts']
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/cifar10_output/img' + str(len(predictions)) + '.jpg' # bit gross
                os.system(cmd)

            if verbose:
                print('image %s: %s' %(entry['image_id'], entry['caption']))

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()
            # predictions_2.pop

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break



    lang_stats = None
    if lang_eval == 1:
        r_weights = model.pre_weights_p.data.cpu().numpy().mean(axis=1)
        lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)
    elif lang_eval == 2:
        lang_stats = utils.language_eval_excoco(predictions_2, predictions_bleu, sents_label_eval, loader)

    tmp = None

    # Switch back to training mode
    model.train()
    if internal is not None:
        internal.train()
    return loss_sum/loss_evals, predictions, lang_stats

def eval_split_output(model, crit, loader, eval_kwargs={}, Discriminator=None, Discriminator_learned=None):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    internal = eval_kwargs.get('internal', 1)
    sim_model = eval_kwargs.get('sim_model', None)
    bleu_option = eval_kwargs.get('bleu_option', 'closest')
    weight_deterministic_flg = eval_kwargs.get('weight_deterministic_flg', 0)
    cut_length = eval_kwargs.get('cut_length', -1)
    baseline_concat = eval_kwargs.get('baseline_concat', 0)

    # Make sure in the evaluation mode
    model.eval()
    # if internal is not None:
    #     internal.eval()

    loader.reset_iterator(split)
    init_scorer('coco-train-idxs', len(loader.ix_to_word))

    n = 0
    count = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    predictions_for_eval = []
    gts_for_wb = {}
    res_for_wb = {}
    while True:
        # get input data
        data = loader.get_batch(split)
        n = n + loader.batch_size

        # data['att_masks'] = None
        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        # try:
            # tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
            #        data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
            #        data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None,
            #        data['bbox'][np.arange(loader.batch_size) * loader.seq_per_img]]
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None,
               data['bbox'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None]

        tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, att_masks, bbox= tmp

        if weight_deterministic_flg > 0:
            weight_index = np.array(data['weight_index'])
            weight_index = weight_index[np.arange(loader.batch_size) * loader.seq_per_img]
        else:
            weight_index = None

        sub_att = None
        # forward the model to also get generated samples for each image
        with torch.no_grad():
            if baseline_concat == 0:
                seq = model(fc_feats, att_feats, att_masks, internal, opt=eval_kwargs, sim_model=sim_model, bbox=bbox,
                            sub_att=sub_att, label_region=data['label_region'], mode='sample', weight_index=weight_index, test_flg=True)[0].data
            else:
                seq, model = make_baseline_result(model, fc_feats, att_feats, eval_kwargs, data, weight_index)

            if Discriminator is not None:
                hokan = torch.zeros((len(seq), 1)).type(torch.LongTensor).cuda()
                # fake_data = torch.cat((hokan, seq, hokan), 1)
                seq = seq.type(hokan.type())
                fake_data = torch.cat((seq, hokan), 1)
                if cut_length > 0:
                    _, dis_score = Discriminator.fixed_length_forward(fake_data, cut_length)
                    _, dis_leanrned_score = Discriminator_learned.fixed_length_forward(fake_data, cut_length)
                else:
                    dis_score = Discriminator(fake_data)
                    dis_leanrned_score = Discriminator_learned(fake_data)
                dis_score = dis_score.data.cpu().numpy()
                dis_leanrned_score = dis_leanrned_score.data.cpu().numpy()
                dis_score = dis_score[:, 1]
                dis_leanrned_score = dis_leanrned_score[:, 1]
            else:
                dis_score = np.zeros(len(seq))
                dis_leanrned_score = np.zeros(len(seq))

        # Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(loader.batch_size):
                print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                print('--' * 10)
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        # informations cal for att
        gen_result_ = seq.data.cpu().numpy()
        word_exist = (gen_result_ > 0).astype(np.int).reshape(gen_result_.shape[0], gen_result_.shape[1], 1)
        weights_for_att = model.weights_p.data.cpu()
        # att_score = get_attnorm_reward(word_exist, weights_for_att).mean(axis=1)
        att_score = np.zeros(len(gen_result_))
        if bbox is not None:
            att_score_hard = get_hardatt_reward(bbox, model.weights.data.cpu(), seq.data.cpu())
        else:
            att_score_hard = None

        for k, sent in enumerate(sents):
            if len(model.attentions.shape) == 3:
                model.attentions = model.attentions.reshape(1, model.attentions.shape[0],
                                                            model.attentions.shape[1], model.attentions.shape[2])
            if len(model.similarity.size()) == 1:
                model.similarity = model.similarity.view(1, model.similarity.size(0))
                model.region_b1 = model.region_b1.view(1, model.region_b1.size(0))
                model.region_b4 = model.region_b4.view(1, model.region_b4.size(0))
                model.region_cider = model.region_cider.view(1, model.region_cider.size(0))

            if att_score_hard is not None:
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent, 'attention':model.attentions[:,:,k,:],
                         'similarity':model.similarity[k].numpy(), 'att_score':att_score[k], 'att_score_hard':att_score_hard[k],
                         'dis_score': dis_score[k], 'dis_learned_score': dis_leanrned_score[k],
                         'region_b1': model.region_b1[k].numpy(), 'region_b4':model.region_b4[k].numpy(), 'region_cider':model.region_cider[k].numpy(),
                         'region_rouge': model.region_rouge[k].numpy(), 'region_meteor': model.region_meteor[k].numpy()}
            else:
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent, 'attention': model.attentions[:, :, k, :],
                         'similarity': model.similarity[k].numpy(), 'att_score': att_score[k],
                         'att_score_hard': None, 'dis_score':dis_score[k], 'dis_learned_score':dis_leanrned_score[k],
                         'region_b1': model.region_b1[k].numpy(), 'region_b4': model.region_b4[k].numpy(),
                         'region_cider': model.region_cider[k].numpy(), 'region_rouge': model.region_rouge[k].numpy(), 'region_meteor': model.region_meteor[k].numpy()}
            entry_for_leval = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            predictions_for_eval.append(entry_for_leval)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/cifar10_output/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                exit()
                os.system(cmd)

            if verbose:
                print('image %s: %s' %(entry['image_id'], entry['caption']))

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()
            predictions_for_eval.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

        corrent_weights = model.weights_p.data.cpu().numpy()

        # if data['label_region'] is not None:
        #     gts_for_wb, res_for_wb, count, corrent_weights = preprocess_wbleu(seq.data.cpu().numpy(), data['gts'],
        #                                                         data['label_region'], corrent_weights, count, gts_for_wb, res_for_wb)
        if model.pre_weights_p is None or baseline_concat == 1:
            r_weights = corrent_weights
        else:
            r_weights = np.concatenate([r_weights, corrent_weights], axis=0)


    lang_stats = None
    if lang_eval == 1:

        lang_stats = language_eval(dataset, predictions_for_eval,
                                   eval_kwargs['id'], split, detail_flg=True, wbleu_set=[gts_for_wb, res_for_wb, r_weights], option=bleu_option)
        for j in range(len(predictions)):
            predictions[j].update(lang_stats[1][str(predictions[j]['image_id'])])
        similarity_calculator(predictions)
        att_score_calculator(predictions)
    elif lang_eval == 2:
        labels = data['labels']
        lang_stats = utils.language_eval_excoco(sents, labels, loader)

    # pdb.set_trace()
    # Switch back to training mode
    # model.train()

    return loss_sum/loss_evals, predictions, lang_stats

def make_baseline_result(model, fc_feats, att_feats, eval_kwargs, data, weight_index):
    seq_length = 100
    att_length = 88
    seq_concat = np.zeros((fc_feats.size(0), seq_length)).astype(np.int32)
    att_concat = np.zeros((2, seq_length, fc_feats.size(0), att_length))
    all_similarity = np.zeros((fc_feats.size(0), seq_length))
    all_region_b1 = np.zeros((fc_feats.size(0), seq_length))
    all_region_b4 = np.zeros((fc_feats.size(0), seq_length))
    all_region_cider = np.zeros((fc_feats.size(0), seq_length))
    all_region_rouge = np.zeros((fc_feats.size(0), seq_length))
    all_region_meteor = np.zeros((fc_feats.size(0), seq_length))
    for i in range(weight_index.shape[2]):
        corrent_weight_index_ = np.copy(weight_index[:, :, i])  # (batch, 2)
        corrent_weight_index = np.reshape(corrent_weight_index_, (weight_index.shape[0], 2, 1))

        use_index = np.where(corrent_weight_index_[:, 0] > -1)
        if len(use_index[0]) == 0:
            break

        label_region = []
        if data['label_region'] is not None:
            for id in use_index[0]:
                label_region.append(data['label_region'][id])
        else:
            label_region = None

        corrent_weight_index[:, 1] = 0
        seq = model(fc_feats[use_index], att_feats[use_index], None, None, opt=eval_kwargs, sim_model=None, bbox=None,
                    sub_att=None, label_region=label_region, mode='sample', weight_index=corrent_weight_index[use_index],
                    test_flg=True)[0].data.cpu().numpy()

        att = np.copy(model.attentions)
        similarity = np.copy(model.similarity)
        region_b1 = np.copy(model.region_b1)
        region_b4 = np.copy(model.region_b4)
        region_cider = np.copy(model.region_cider)
        region_rouge = np.copy(model.region_rouge)
        region_meteor = np.copy(model.region_meteor)

        for j in range(len(use_index[0])):
            count_used = (seq_concat[use_index[0][j]] > 0).sum()
            corrent_used = min((seq[j] > 0).sum(), max(seq_length - count_used, 0))

            if corrent_used > 0:
                seq_concat[use_index[0][j], count_used:count_used + corrent_used] = seq[j, :corrent_used]
                att_concat[:, count_used:count_used + corrent_used, use_index[0][j]] = att[:, :corrent_used, j]
                all_similarity[use_index[0][j], count_used:count_used + corrent_used] = similarity[j, :corrent_used]
                all_region_b1[use_index[0][j], count_used:count_used + corrent_used] = region_b1[j, :corrent_used]
                all_region_b4[use_index[0][j], count_used:count_used + corrent_used] = region_b4[j, :corrent_used]
                all_region_cider[use_index[0][j], count_used:count_used + corrent_used] = region_cider[j, :corrent_used]
                all_region_rouge[use_index[0][j], count_used:count_used + corrent_used] = region_rouge[j, :corrent_used]
                all_region_meteor[use_index[0][j], count_used:count_used + corrent_used] = region_meteor[j, :corrent_used]

    seq_concat = torch.from_numpy(seq_concat).cuda()
    all_similarity = torch.from_numpy(all_similarity)
    all_region_b1 = torch.from_numpy(all_region_b1)
    all_region_b4 = torch.from_numpy(all_region_b4)
    all_region_cider = torch.from_numpy(all_region_cider)
    all_region_rouge = torch.from_numpy(all_region_rouge)
    all_region_meteor = torch.from_numpy(all_region_meteor)

    model.similarity = all_similarity
    model.region_b1 = all_region_b1
    model.region_b4 = all_region_b4
    model.region_cider = all_region_cider
    model.region_rouge = all_region_rouge
    model.region_meteor = all_region_meteor
    model.attentions = att_concat

    return seq_concat, model

def array_to_str(arr):
    arr = arr.tolist()
    while 0 in arr: arr.remove(0)
    out = ' '.join(map(str, arr))
    return out

def preprocess_wbleu(gen_result, data_gts, label_region, weights, count, gts_for_wb, res_for_wb):
    label_region = label_region.data.cpu().numpy().astype(np.int32)
    for i in range(len(data_gts)):
        seq_gt = []
        for j in range(weights.shape[1]):
            seq_gt.append(array_to_str(label_region[i][j]))
        seq_gt_ = copy.copy(seq_gt)
        gts_for_wb[i + count] = seq_gt_

    for i in range(len(gen_result)):
        res_for_wb[i + count] = [array_to_str(gen_result[i])]

    count += len(data_gts)

    word_exist = (gen_result > 0).astype(np.int).reshape(gen_result.shape[0], gen_result.shape[1], 1)
    weights = weights[:, ::word_exist.shape[1], :] * word_exist
    weights = weights.sum(axis=1) / word_exist.sum(1)

    return gts_for_wb, res_for_wb, count, weights

def write_record(name, prediction, lang_stats):
    text = name + ' \n'
    text += evaluation_calculator(lang_stats) + ' \n'
    text += similarity_calculator(prediction) + ' \n'
    text += att_score_calculator(prediction) + ' \n'
    text += discriminator_score(prediction) + ' \n'
    return text

def similarity_calculator(info):
    print('computing Similarity score...')
    ave = 0
    ave_b1 = 0
    ave_b4 = 0
    ave_c =0
    ave_rouge = 0
    ave_meteor = 0
    for j in range(len(info)):
        ave += cal_avg_sim(info[j]['similarity'])
        ave_b1 += cal_avg_sim(info[j]['region_b1'])
        ave_b4 += cal_avg_sim(info[j]['region_b4'])
        ave_c += cal_avg_sim(info[j]['region_cider'])
        ave_rouge += cal_avg_sim(info[j]['region_rouge'])
        ave_meteor += cal_avg_sim(info[j]['region_meteor'])
    text = 'Similarity: {}'.format(np.round(ave / len(info), 3)) + ' \n'
    text += 'region_b1: {}'.format(np.round(ave_b1 / len(info), 5)) + ' \n'
    text += 'region_b4: {}'.format(np.round(ave_b4 / len(info), 5)) + ' \n'
    text += 'region_cider: {}'.format(np.round(ave_c / len(info), 5)) + ' \n'
    text += 'region_rouge: {}'.format(np.round(ave_rouge / len(info), 5)) + ' \n'
    text += 'region_meteor: {}'.format(np.round(ave_meteor / len(info), 5)) + ' \n'

    print(text)
    return text

def att_score_calculator(info):
    print('computing Att score...')
    ave = 0
    ave_hard = 0
    for j in range(len(info)):
        ave += info[j]['att_score']
        if info[j]['att_score_hard'] is not None:
            ave_hard += info[j]['att_score_hard']
    text = 'Att_score: {} \n'.format(np.round(ave / len(info), 3))
    if info[j]['att_score_hard'] is not None:
        text += 'Att_score_hard: {}'.format(np.round(ave_hard / len(info), 3))
    print(text)
    return text

def discriminator_score(info):
    print('computing discriminator score...')
    ave = 0
    ave_learned = 0
    for j in range(len(info)):
        ave += info[j]['dis_score']
        ave_learned += info[j]['dis_learned_score']
    text = 'discriminator_score: {} \n'.format(np.round(ave / len(info), 3))
    text += 'discriminator_learned_score: {} \n'.format(np.round(ave_learned / len(info), 3))
    print(text)
    return text

def evaluation_calculator(lang_stats):
    lang_stats = lang_stats[0]
    text = 'CIDEr: {}'.format(np.round(lang_stats['CIDEr'], 5)) + ' \n' + \
           'Bleu_1: {}'.format(np.round(lang_stats['Bleu_1'], 5)) + ' \n' + \
           'Bleu_4: {}'.format(np.round(lang_stats['Bleu_4'], 5)) + ' \n' + \
           'Recall_1: {}'.format(np.round(lang_stats['Recall_1'], 5)) + ' \n' + \
           'Recall_4: {}'.format(np.round(lang_stats['Recall_4'], 5)) + ' \n' + \
           'METEOR: {}'.format(np.round(lang_stats['METEOR'], 5)) + ' \n' + \
           'ROUGE_L: {}'.format(np.round(lang_stats['ROUGE_L'], 5)) + ' \n'
    if 'WBleu_1' in lang_stats.keys():
        text += 'WBleu_1: {}'.format(np.round(lang_stats['WBleu_1'], 5)) + ' \n' + \
           'WBleu_4: {}'.format(np.round(lang_stats['WBleu_4'], 5)) + ' \n'

    return text

def cal_avg_sim(sim_score):
    count_nonzero = (sim_score != 0.0) * 1
    count_nonzero = np.sum(count_nonzero)
    sim_score = (sim_score.sum() / max(count_nonzero, 1))

    return sim_score