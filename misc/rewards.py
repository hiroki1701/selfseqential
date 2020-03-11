from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
from collections import OrderedDict
import torch
import copy
from sklearn import preprocessing

import sys

sys.path.append("/home/nakamura/project/python3_selfsequential/cider")
sys.path.append("cider")
print(sys.path)
from pyciderevalcap.ciderD.ciderD import CiderD

sys.path.append("coco-caption")
# from pycocoevalcap.cider.cider import Cider

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.recall.recall import Recall
from pycocoevalcap.weighted_bleu.weighted_bleu import Weighted_Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

# from speaksee.evaluation import Bleu
# from speaksee.evaluation import Cider as CiderD

sys.path.append("/home/nakamura/project/python3_selfsequential/similality")
sys.path.append("/home/nakamura/project/selfsequential/similality")
from cos_distance import Sim_model

import pdb

CiderD_scorer = None
Bleu_scorer = None
Recall_scorer = None
Weighted_Bleu_scorer = None
Weighted_Bleu1_scorer = None
Global_sim_scorer = None
# CiderD_scorer = CiderD(df='corpus')

def init_scorer(cached_tokens, vocab_size):
    global CiderD_scorer
    # CiderD_scorer = Cider_scorer or Cider
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)
    global Recall_scorer
    Recall_scorer = Recall_scorer or Recall(4)
    global Weighted_Bleu_scorer
    Weighted_Bleu_scorer = Weighted_Bleu_scorer or Weighted_Bleu(4)
    global Weighted_Bleu1_scorer
    Weighted_Bleu1_scorer = Weighted_Bleu(1)
    global Rouge_scorer
    Rouge_scorer = Rouge()
    global Meteor_scorer
    Meteor_scorer = Meteor()
    global Global_sim_scorer
    Global_sim_scorer = Sim_model(512, 512, vocab_size=vocab_size)
    # checkpoint = torch.load('/mnt/workspace2019/nakamura/selfsequential/sim_model/no_shuffle_/model_26_8900.pt')
    # checkpoint = torch.load('/mnt/workspace2019/nakamura/selfsequential/sim_model/no_shuffle_newloss/model_23_8000.pt',  map_location='cuda:0')
    checkpoint = torch.load('/mnt/workspace2019/nakamura/selfsequential/sim_model/vg_noshuffle2/model_106_36000.pt')
    Global_sim_scorer.load_state_dict(checkpoint['model_state_dict'])
    Global_sim_scorer.eval()
    Global_sim_scorer.cuda()

def array_to_str(arr):
    arr = arr.tolist()
    while 0 in arr: arr.remove(0)
    out = ' '.join(map(str, arr))
    return out

def count_seqlength(r_weights):
    index = np.where(r_weights[:,:,0]==1)
    seq_length = r_weights.shape[1]
    weight_length_array = np.zeros(len(r_weights))
    for i in range(len(r_weights.shape[0])):
        weight_length_array[i] = seq_length - len(np.where(index[0]==0)[0])
    return weight_length_array

def cal_avg_sim(similarity):
    sim_score = similarity.data.numpy()
    count_nonzero = (sim_score != 0.0) * 1
    count_nonzero = np.sum(count_nonzero, axis=1)
    index = np.where(count_nonzero == 0)
    count_nonzero[index] = 1
    sim_score = (sim_score.sum(axis=1) / count_nonzero)

    return sim_score

def cal_sum_sim(similarity):
    sim_score = similarity.data.numpy()
    sim_score = sim_score.sum(axis=1)

    return sim_score

def cal_avg_sim_sparse(similarity):
    sim_score = similarity.data.numpy()
    count_nonzero = (sim_score != 0.0) * 1
    count_nonzero = np.sum(count_nonzero, axis=1)
    index = np.where(count_nonzero == 0)
    count_nonzero[index] = 1
    sim_score = sim_score / count_nonzero.reshape(count_nonzero.shape[0], 1)

    return sim_score


def get_internal_reward_(data, gen_result, opt):
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])
    # get greedy decoding baselin

    res = OrderedDict()

    gen_result = gen_result.data.cpu().numpy()

    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    res_ = [{'image_id': i, 'caption': res[i]} for i in range(batch_size)]
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(batch_size)}
    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_, option=opt.cider_option)
        print('Cider scores:', _)
    else:
        cider_scores = 0

    scores = opt.cider_reward_weight * cider_scores
    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards


def get_internal_reward(model, fc_feats, att_feats, att_masks, data, gen_result, opt, internal=None, sim_model=None):
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])
    # get greedy decoding baselin

    internal.eval()
    with torch.no_grad():
        greedy_res, _ = model(fc_feats, att_feats, att_masks=att_masks, Critic=internal, sim_model=sim_model,
                              mode='sample')

    internal.train()

    res = OrderedDict()

    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()

    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    res_ = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
    res__ = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_, option=opt.cider_option)
        print('Cider scores:', _)
    else:
        cider_scores = 0

    if opt.recall_reward_weight > 0:
        _, recall_scores = Recall_scorer.compute_score(gts, res__)
        recall_scores = np.array(recall_scores[3])
        print('Recall scores:', _[3])
    else:
        recall_scores = 0

    if opt.bleu_reward_weight > 0:
        if opt.dataset == 'coco':
            option = 'closest'
        elif opt.dataset == 'vg':
            option = opt.bleu_option
        # _, b_scores = Bleu_scorer.compute_score(gts, res__, option=option)
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__, option=option)
        bleu_scores = np.array(bleu_scores[3])
        print('Bleu scores:', _[3])
    else:
        bleu_scores = 0

    scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores + opt.recall_reward_weight * recall_scores
    scores = scores[:batch_size] - scores[batch_size:]
    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards


def get_self_critical_and_similarity_reward(model, fc_feats, att_feats, att_masks, data, gen_result, opt,
                                            similarity, internal=None, sim_model=None, bbox=None, label_region=None,
                                            sub_att=None, D=None, weight_index=None, fixed_region=None, target_model=None):
    if opt.att_reward_flg==1:
        weights = model.weights_p.data.cpu() #(batch, seq, att_num)
    if opt.caption_model == 'hcatt_hard' and opt.sum_reward_rate > 0.0 :
        weights = model.weights.data.cpu()

    # --Get CIDEr score--
    cider_reward, c_score, caption_train, greedy_res = get_self_critical_reward_(model, fc_feats, att_feats, att_masks, data,
                                                                     gen_result, opt,
                                                                     internal=internal, sim_model=sim_model, label_region=label_region, bbox=bbox,
                                                                     sub_att=sub_att, D=D, weight_index=weight_index, fixed_region=fixed_region, target_model=target_model)

    att_score = np.zeros(1)
    att_reward = 0
    att_lambda = 0
    actor_critic_reward = None

    # --Get similarity score--

    # cider_reward, (batch, seq_length)
    # c_score, (batch)
    if opt.sim_reward_flg == 1:
        if opt.sim_sum_flg == 1:
            # sum
            pdb.set_trace()
            sim_score = cal_sum_sim(similarity)  # (batch, )
            eval_sim_score = cal_sum_sim(model.similarity)  # (batch, )
        else:
            # mean
            sim_score = cal_avg_sim(similarity)  # (batch, )
            if sim_score.mean() != sim_score.mean():
                pdb.set_trace()
            if target_model is not None:
                eval_sim_score = cal_avg_sim(target_model.similarity)
            else:
                eval_sim_score = cal_avg_sim(model.similarity)  # (batch, )
        diff_sim = sim_score - eval_sim_score  # (batch, )
        # diff_sim = diff_sim.data.numpy().mean(axis=1)  # (batch,)

        print('similarity : ', sim_score.mean())
        diff_sim = np.repeat(diff_sim[:, np.newaxis], gen_result.shape[1], 1)  # (batch, 16)
        # alpha = max(opt.cider_reward_weight, opt.bleu_reward_weight, opt.recall_reward_weight)
        alpha = 1.0
        if opt.critic_weight > 0.0:
            beta = opt.critic_weight
        else:
            beta = 1.0 - alpha
        if opt.critic_cider_reward_weight is None:
            c_alpha = alpha
        else:
            c_alpha = opt.critic_cider_reward_weight
        ganma = opt.discriminator_weight
    else:
        sim_score = 0
        diff_sim = 0
        alpha = max(1.0, opt.recall_reward_weight)
        beta = 0
        ganma = opt.discriminator_weight

    pure_reward = np.zeros(np.shape(c_score))

    # --Distribution of CIDEr score and similarity--
    if opt.l_score is not None:
        c_high_index = np.where(c_score > opt.l_score)
        pure_reward += c_score
        pure_reward[c_high_index] = alpha * c_score[c_high_index] + (1 - alpha) * sim_score[c_high_index]
        cider_reward[c_high_index] = alpha * cider_reward[c_high_index] + (1 - alpha) * diff_sim[c_high_index]
        reward = cider_reward
    elif opt.separate_reward == 1:
        if caption_train:
            pure_reward = alpha * c_score + beta * sim_score
            reward = alpha * cider_reward + beta * diff_sim
        else:
            pure_reward = sim_score
            reward = diff_sim
        try:
            print('pure_reward: {}, global: {}, local: {}'.format(pure_reward.mean(), alpha * c_score.mean(), beta * sim_score.mean()))
        except AttributeError:
            print('pure_reward: {}, global: {}, local: {}'.format(pure_reward.mean(), alpha * c_score.mean(), beta * sim_score))
    else:
        pure_reward = alpha * c_score + beta * sim_score
        reward = alpha * cider_reward + beta * diff_sim
        try:
            print('pure_reward: {}, global: {}, local: {}'.format(pure_reward.mean(), alpha * c_score.mean(), beta * sim_score.mean() ))
        except AttributeError:
            print('pure_reward: {}, global: {}, local: {}'.format(pure_reward.mean(), alpha * c_score.mean(),
                                                                  beta * sim_score))
    # Penalty to train internal critic easily
    if not caption_train:
        if opt.penalty_type == 'cases':
            # if number of change the attention is larger than T * 0.7, model is given penalty
            word_exist_critic = gen_result.data.cpu() > 0
            word_exist_critic = word_exist_critic.type(torch.FloatTensor)
            critic_penalty = (internal.pre_same_action_flg / word_exist_critic.sum(dim=1)) > 0.3
            critic_penalty_ = (internal.pre_same_action_flg / word_exist_critic.sum(dim=1)) < 0.1
            critic_penalty += critic_penalty_
            critic_penalty = critic_penalty.type(torch.FloatTensor)
            critic_penalty = critic_penalty.numpy()
            critic_penalty = np.repeat(critic_penalty[:, np.newaxis], gen_result.shape[1], 1)
            reward = reward - 1 * critic_penalty

        elif opt.penalty_type == 'compare':
            word_exist_critic = gen_result.data.cpu() > 0
            word_exist_critic = word_exist_critic.type(torch.FloatTensor)
            word_exist_critic[np.where(word_exist_critic < 1)] = 1
            critic_penalty = -1 * torch.pow((internal.pre_same_action_flg / word_exist_critic.sum(dim=1)), 2)

            word_exist_critic_eval = torch.from_numpy(greedy_res) > 0
            word_exist_critic_eval = word_exist_critic_eval.type(torch.FloatTensor)
            word_exist_critic_eval[np.where(word_exist_critic_eval < 1)] = 1
            critic_penalty_eval = -1 * torch.pow((internal.same_action_flg / word_exist_critic_eval.sum(dim=1)), 2)

            critic_penalty = critic_penalty - critic_penalty_eval
            critic_penalty = critic_penalty.numpy()
            critic_penalty = np.repeat(critic_penalty[:, np.newaxis], gen_result.shape[1], 1)
            reward = reward + 0.7 * critic_penalty

    # Get Area reward
    if opt.caption_model == 'hcatt_hard' and opt.sum_reward_rate > 0.0:
        if bbox is None:
            print('bounding box is None!')
            exit()
        else:
            hard_att_reward = get_hardatt_reward(bbox, weights, gen_result)
            hard_att_reward_eval = get_hardatt_reward(bbox, model.weights.data.cpu(), gen_result)
            hard_att_reward = hard_att_reward - hard_att_reward_eval
            hard_att_reward = np.repeat(hard_att_reward.numpy()[:, np.newaxis], gen_result.shape[1], 1)
            reward += opt.sum_reward_rate * hard_att_reward

    # clipping reward
    # reward = np.clip(reward, -0.1, 0.1)
    if np.mean(reward)> -0.05:
        target_update_flg = True
    else:
        target_update_flg = False

    return reward, pure_reward, actor_critic_reward, target_update_flg

def get_hardatt_reward(bbox, weights, gen_result):
    bbox = bbox.data.cpu()
    seq_length = gen_result.size(1)
    selected_index = np.where(weights.numpy()>0)
    selected_weights = bbox[selected_index[0], selected_index[2], :] #(batch * seq_length, att_size)
    selected_weights = selected_weights.view(-1, seq_length, selected_weights.size(-1)) #(batch, seq_length, att_size)
    hard_att_reward = torch.clamp(selected_weights.sum(dim=1), 0.0, 1.0).sum(dim=1).type(torch.FloatTensor)/selected_weights.size(-1) #(batch, )

    return hard_att_reward

def get_self_critical_and_similarity_reward_for_actor_critic(model, fc_feats, att_feats, att_masks, data, gen_result, opt,
                                            similarity, internal=None, sim_model=None, bbox=None, sub_att=None, label_region=None, D=None):
    if opt.att_reward_flg==1:
        weights = model.weights_p.data.cpu() #(batch, seq, att_num)

    cider_reward, c_score, caption_train, greedy_res = get_self_critical_reward_for_actor_critic(model, fc_feats, att_feats, att_masks, data,
                                                                     gen_result, opt,
                                                                     internal=internal, sim_model=sim_model, bbox=bbox,
                                                                     sub_att=sub_att, label_region=label_region, D=D)

    att_score = np.zeros(1)
    att_reward = 0
    att_lambda = 0

    # cider_reward, (batch, seq_length)
    # c_score, (batch)
    if opt.sim_reward_flg == 1:
        if opt.sim_sum_flg == 1:
            # sum
            sim_score = cal_sum_sim(similarity)  # (batch, )
        else:
            # mean
            sim_score = cal_avg_sim(similarity)  # (batch, )
            if sim_score.mean() != sim_score.mean():
                pdb.set_trace()
        # mean
        if sim_score.mean() != sim_score.mean():
            pdb.set_trace()
        diff_sim = sim_score # (batch, )
        # diff_sim = diff_sim.data.numpy().mean(axis=1)  # (batch,)

        print('similarity : ', sim_score.mean())
        diff_sim = np.repeat(diff_sim[:, np.newaxis], gen_result.shape[1], 1)  # (batch, 16)
        alpha = max(opt.cider_reward_weight, opt.bleu_reward_weight, opt.bleu_reward_weight)
        if opt.critic_weight > 0:
            beta = opt.critic_weight
        else:
            beta = 1.0 - alpha
    else:
        sim_score = 0
        diff_sim = 0
        alpha = 1.0
        beta = 0.0

    # pdb.set_trace()

    pure_reward = alpha * c_score + beta * sim_score + att_lambda * att_score.mean()
    try:
        print('purereward: {}, gloabal: {}, local: {}'.format(pure_reward.mean(), alpha * c_score.mean(), beta * sim_score.mean()))
    except AttributeError:
        print('purereward: {}, gloabal: {}, local: {}'.format(pure_reward.mean(), alpha * c_score.mean(),
                                                              beta * sim_score))

    reward = alpha * cider_reward + beta * diff_sim + att_lambda * att_reward
    return reward, pure_reward

def get_double_reward(model, fc_feats, att_feats, att_masks, data, gen_result, opt,
                                            similarity, internal=None, sim_model=None, bbox=None, sub_att=None):
    if opt.att_reward_flg==1:
        weights = model.weights_p.data.cpu() #(batch, seq, att_num)

    cider_reward, c_score, caption_train, greedy_res = get_self_critical_reward_(model, fc_feats, att_feats, att_masks, data,
                                                                     gen_result, opt,
                                                                     internal=internal, sim_model=sim_model, bbox=bbox,
                                                                     sub_att=sub_att)

    if opt.att_reward_flg==100:
        eval_weights = model.weights_p.data.cpu()
        gen_result_ = gen_result.data.cpu().numpy()
        word_exist = (gen_result_>0).astype(np.int).reshape(gen_result_.shape[0], gen_result_.shape[1], 1)
        word_exist_eval = (greedy_res > 0).astype(np.int).reshape(greedy_res.shape[0], greedy_res.shape[1], 1)
        # word_exist = np.ones((gen_result_.shape[0], gen_result_.shape[1], 1))
        # word_exist_eval = np.ones((gen_result_.shape[0], gen_result_.shape[1], 1))
        att_score = get_attnorm_reward(word_exist, weights)
        eval_att_score = get_attnorm_reward(word_exist_eval, eval_weights)
        att_reward = att_score - eval_att_score #(batch, )
        att_lambda = opt.att_lambda
        model.att_score = att_score.mean()
    else:
        att_score = np.zeros(1)
        att_reward = 0
        att_lambda = 0

    # cider_reward, (batch, seq_length)
    # c_score, (batch)
    if opt.sim_reward_flg == 1:
        # mean
        sim_score = cal_avg_sim(similarity)  # (batch, )
        if sim_score.mean() != sim_score.mean():
            pdb.set_trace()

        # mean
        eval_sim_score = cal_avg_sim(model.similarity)  # (batch, )
        diff_sim = sim_score - eval_sim_score  # (batch, )
        # diff_sim = diff_sim.data.numpy().mean(axis=1)  # (batch,)

        print('similarity : ', sim_score.mean())
        diff_sim = np.repeat(diff_sim[:, np.newaxis], gen_result.shape[1], 1)  # (batch, 16)
        alpha = max(opt.cider_reward_weight, opt.bleu_reward_weight)
        if opt.critic_cider_reward_weight is None:
            c_alpha = alpha
        else:
            c_alpha = opt.critic_cider_reward_weight
    else:
        sim_score = 0
        diff_sim = 0
        alpha = 1.0

    pure_reward = alpha * c_score + (1.0 - alpha) * sim_score + att_lambda * att_score.mean()
    reward = alpha * cider_reward + (1.0 - alpha) * diff_sim + att_lambda * att_reward
    pure_reward_critic = c_alpha * c_score + (1.0 - c_alpha) * sim_score
    reward_critic = c_alpha * cider_reward + (1.0 - c_alpha) * diff_sim
    # if internal is not None:
    #     same_action_flg = internal.pre_same_action_flg
    #     reward_critic[np.where(same_action_flg)==1.0] = -10.0
    #     if same_action_flg.sum() > 0:
    #         print('penalty!!')
    #     else:
    #         print('safe')
    return reward, pure_reward, reward_critic, pure_reward_critic

def get_self_critical_and_similarity_reward_process(model, fc_feats, att_feats, att_masks, data, gen_result, opt,
                                                    similarity, internal=None, sim_model=None):
    cider_reward = get_self_critical_reward(model, fc_feats, att_feats, att_masks, data, gen_result, opt, internal,
                                            sim_model)
    sim_score = cal_avg_sim(model.similarity_for_cal)
    # mean
    diff_sim = similarity.data.numpy()  # (batch, 16)

    print('similarity : ', sim_score.mean())

    alpha = max(opt.cider_reward_weight, opt.bleu_reward_weight)
    reward = cider_reward + alpha * diff_sim

    return reward


def get_self_critical_and_similarity_reward_process_only_prob(model, fc_feats, att_feats, att_masks, data, gen_result,
                                                              opt, similarity, internal=None, sim_model=None):
    # gen_result : result of model(tansaku)
    # greedy_res : result of model (greedy, baseline)
    # reward : cider(gen_result) - cider(greedy_res)
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])

    res = OrderedDict()

    gen_result = gen_result.data.cpu().numpy()

    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    res_ = [{'image_id': i, 'caption': res[i]} for i in range(batch_size)]
    res__ = {i: res[i] for i in range(batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(batch_size)}

    _, cider_scores = CiderD_scorer.compute_score(gts, res_, option=opt.cider_option)
    print('Cider scores:', _)

    scores = cider_scores

    cider_reward = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    sim_score = cal_avg_sim(model.similarity_for_cal)
    # mean
    diff_sim = similarity.data.numpy()  # (batch, 16)

    print('similarity : ', sim_score.mean())

    alpha = max(opt.cider_reward_weight, opt.bleu_reward_weight)
    reward = cider_reward + alpha * diff_sim

    return reward


def get_self_critical_and_similarity_att_reward(model, fc_feats, att_feats, att_masks, data, gen_result,
                                                opt, similarity, weights, weights_num, internal=None, sim_model=None):
    # pdb.set_trace()
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    # get greedy decoding baseline
    model.eval()
    with torch.no_grad():
        greedy_res, _ = model(fc_feats, att_feats, att_masks=att_masks, Critic=internal, sim_model=sim_model,
                              mode='sample')
    model.train()

    cider_reward = get_self_critical_reward(model, fc_feats, att_feats, att_masks, data, gen_result, opt,
                                            internal=internal, sim_model=sim_model, greedy_res=greedy_res)
    eval_weights = model.weights
    eval_weihgts_num = internal.output_action.sum(dim=-1)

    # cal similarity
    sim_score = cal_avg_sim(similarity)  # (batch, )
    if sim_score.mean() != sim_score.mean():
        pdb.set_trace()
    eval_sim_score = cal_avg_sim(model.similarity)  # (batch, )
    diff_sim = sim_score - eval_sim_score  # (batch, )
    # diff_sim = diff_sim.data.numpy().mean(axis=1)  # (batch,)
    print('similarity : ', sim_score.mean())
    diff_sim = np.repeat(diff_sim[:, np.newaxis], gen_result.shape[1], 1)  # (batch, 16)

    # cal att loss
    att_score = get_attnorm_reward(gen_result, weights, weights_num, internal=None)
    eval_att_score = get_attnorm_reward(greedy_res, eval_weights, eval_weihgts_num, internal=None)
    diff_att_score = att_score - eval_att_score  # (batch, 16)
    print('att_score : ', att_score.mean())

    alpha = max(opt.cider_reward_weight, opt.bleu_reward_weight)
    beta = opt.att_reward_weight
    reward = alpha * cider_reward + (1 - alpha) * diff_sim + beta * diff_att_score
    print('rate: ', cider_reward.mean(), diff_sim.mean(), diff_att_score.mean())

    return reward


def get_self_critical_reward(model, fc_feats, att_feats, att_masks, data, gen_result, opt, internal=None,
                             sim_model=None, greedy_res=None, bbox=None, sub_att=None):
    # gen_result : result of model(tansaku)
    # greedy_res : result of model (greedy, baseline)
    # reward : cider(gen_result) - cider(greedy_res)
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])
    if greedy_res is None:
        # get greedy decoding baseline
        model.eval()
        with torch.no_grad():
            greedy_res, _ = model(fc_feats, att_feats, att_masks=att_masks, Critic=internal, sim_model=sim_model,
                                  bbox=bbox, sub_att=sub_att, mode='sample')

        model.train()

    res = OrderedDict()

    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    res_ = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
    res__ = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}

    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_, option=opt.cider_option)
        model.c_score = _
        print('Cider scores:', _)
    else:
        cider_scores = 0

    if opt.recall_reward_weight > 0:
        _, recall_scores = Recall_scorer.compute_score(gts, res__)
        recall_scores = np.array(recall_scores[3])
        print('Recall scores:', _[3])
    else:
        recall_scores = 0

    if opt.bleu_reward_weight > 0:
        if opt.dataset == 'coco':
            option = 'closest'
        elif opt.dataset == 'vg':
            option = opt.bleu_option
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__, option=option)
        bleu_scores = np.array(bleu_scores[3])
        print('Bleu scores:', _[3])
    else:
        bleu_scores = 0
    if opt.log_flg == 0:
        scores = opt.cider_reward_weight * cider_scores + opt.bleu_reward_weight * bleu_scores + opt.recall_reward_weight * recall_scores
    else:
        scores = opt.cider_reward_weight * np.log(cider_scores)

    scores = scores[:batch_size] - scores[batch_size:]
    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)
    # pdb.set_trace()
    return rewards


def calculate_region_bleu(t, seq, label_region, weight, seq_per_img, cc, index_z, ended_sequense_list, top_n=3):
    # seq += 1
    index_one = np.where(cc.data.cpu().numpy() == 1)
    calculate_id_list = list(index_one[0]) + list(index_z[0])
    calculate_id_list = set(calculate_id_list) - ended_sequense_list
    calculate_id_list = list(calculate_id_list)
    calculate_id_list.sort()

    weight_numpy = weight.data.cpu().numpy()
    wbleu_scores_all = np.zeros(len(weight_numpy))
    weight_captions = []
    # index_att = np.argsort(weight_numpy, axis=1)[::-1]
    # index_top = []
    # for i in range(top_n):
    #     index_top.append(np.where(index_att == i))

    gts_att = {}
    res = {}

    # for i in range(len(seq)):
    #     res[i] = [array_to_str(seq[i])]

    # seq_perimg について計算する．
    if len(seq) == len(label_region):
        seq_per_img = 1

    cap_count = 0
    for i in range(len(seq)):
        if i in calculate_id_list :
            seq_gt = []
            weight_caption = []
            for j in range(len(label_region[(i % len(seq)) // seq_per_img])):
                max_att = np.max(weight_numpy[i, :len(label_region[(i % len(seq)) // seq_per_img])])
                if weight_numpy[i, j] > 1.0/weight_numpy.shape[1] or weight_numpy[i, j] == max_att:
                    region_seq = [array_to_str(l) for l in label_region[(i % len(seq)) // seq_per_img][j]]
                    region_seq_ = copy.copy(region_seq)
                    seq_gt += region_seq_
                    weight_caption += [weight_numpy[i, j]] * len(region_seq_)
            if len(seq_gt) == 0:
                pdb.set_trace()
            seq_gt_ = copy.copy(seq_gt)
            gts_att[cap_count] = seq_gt_
            weight_caption_ = copy.copy(weight_caption)
            weight_captions.append(weight_caption_)
            res[cap_count] = [array_to_str(seq[i])]
            cap_count += 1

    # print(index_z, ended_sequense_list)
    # pdb.set_trace()

    _, wbleu_scores = Weighted_Bleu1_scorer.compute_score(gts_att, res, weight_captions)
    if len(calculate_id_list) > 0:
        min_index_b4 = np.where(wbleu_scores == 0)
        if len(min_index_b4[0]) > 0:
            wbleu_scores[min_index_b4] = sys.float_info.min
    wbleu_scores_all[calculate_id_list] = np.array(wbleu_scores[0])
    # _, wbleu_scores = Weighted_Bleu_scorer.compute_score(gts_att, res, weight_captions)
    # wbleu_scores_all[calculate_id_list] = np.array(wbleu_scores[3])
    # # pdb.set_trace()

    return wbleu_scores_all

def calculate_region_cider(t, seq, label_region, weight, seq_per_img, cc, index_z, ended_sequense_list, top_n=3):
    index_one = np.where(cc.data.cpu().numpy() == 1)
    calculate_id_list = list(index_one[0]) + list(index_z[0])
    calculate_id_list = set(calculate_id_list) - ended_sequense_list
    calculate_id_list = list(calculate_id_list)
    calculate_id_list.sort()

    weight_numpy = weight.data.cpu().numpy()
    region_cider_scores_all = np.zeros(len(weight_numpy))
    weight_captions = []

    gts_att = {}
    res = []

    # for i in range(len(seq)):
    #     res[i] = [array_to_str(seq[i])]

    # seq_perimg について計算する．
    if len(seq) == len(label_region):
        seq_per_img = 1

    cap_count = 0
    for i in range(len(seq)):
        if i in calculate_id_list:
            seq_gt = []
            weight_caption = []
            for j in range(len(label_region[(i % len(seq)) // seq_per_img])):
                max_att = np.max(weight_numpy[i, :len(label_region[(i % len(seq)) // seq_per_img])])
                if weight_numpy[i, j] > 0.0 or weight_numpy[i, j] == max_att:
                    region_seq = [array_to_str(l) for l in label_region[(i % len(seq)) // seq_per_img][j]]
                    region_seq_ = copy.copy(region_seq)
                    seq_gt += region_seq_
                    weight_caption += [weight_numpy[i, j]] * len(region_seq_)
                    break
            if len(seq_gt) == 0:
                pdb.set_trace()
            seq_gt_ = copy.copy(seq_gt)
            gts_att[cap_count] = seq_gt_
            weight_caption_ = copy.copy(weight_caption)
            weight_captions.append(weight_caption_)
            res_element = {}
            res_element['image_id'] = cap_count
            res_element['caption'] = [array_to_str(seq[i])]
            res.append(res_element)
            cap_count += 1

    _, region_cider_scores = CiderD_scorer.compute_score(gts_att, res)
    if len(calculate_id_list) > 0:
        min_index_c = np.where(region_cider_scores == 0)
        if len(min_index_c) > 0:
            region_cider_scores[min_index_c] = 1e-8
    region_cider_scores_all[calculate_id_list] = np.array(region_cider_scores)

    return region_cider_scores_all

def calculate_all_region_metrix(t, seq, label_region, weight, seq_per_img, cc, index_z, ended_sequense_list, top_n=3):
    index_one = np.where(cc.data.cpu().numpy() == 1)
    calculate_id_list = list(index_one[0]) + list(index_z[0])
    calculate_id_list = set(calculate_id_list) - ended_sequense_list
    calculate_id_list = list(calculate_id_list)
    calculate_id_list.sort()

    weight_numpy = weight.data.cpu().numpy()
    wbleu1_scores_all = np.zeros(len(weight_numpy))
    wbleu4_scores_all = np.zeros(len(weight_numpy))
    wrouge_scores_all = np.zeros(len(weight_numpy))
    wmeteor_scores_all = np.zeros(len(weight_numpy))
    wcider_scores_all = np.zeros(len(weight_numpy))
    weight_captions = []
    # index_att = np.argsort(weight_numpy, axis=1)[::-1]
    # index_top = []
    # for i in range(top_n):
    #     index_top.append(np.where(index_att == i))

    gts_att = {}
    res = {}
    res_cider = []

    # for i in range(len(seq)):
    #     res[i] = [array_to_str(seq[i])]

    # seq_perimg について計算する．
    if len(seq) == len(label_region):
        seq_per_img = 1

    cap_count = 0
    for i in range(len(seq)):
        if i in calculate_id_list:
            seq_gt = []
            weight_caption = []
            for j in range(len(label_region[(i % len(seq)) // seq_per_img])):
                max_att = np.max(weight_numpy[i, :len(label_region[(i % len(seq)) // seq_per_img])])
                if  weight_numpy[i, j] > 0.0 or weight_numpy[i, j] == max_att:
                    region_seq = [array_to_str(l) for l in label_region[(i % len(seq)) // seq_per_img][j]]
                    region_seq_ = copy.copy(region_seq)
                    seq_gt += region_seq_
                    weight_caption += [weight_numpy[i, j]] * len(region_seq_)
                    break
            if len(seq_gt) == 0:
                pdb.set_trace()

            # make gt
            seq_gt_ = copy.copy(seq_gt)
            gts_att[cap_count] = seq_gt_
            weight_caption_ = copy.copy(weight_caption)
            weight_captions.append(weight_caption_)
            # make res caption of bleu
            res[cap_count] = [array_to_str(seq[i])]
            # make res caption of cider
            res_element = {}
            res_element['image_id'] = cap_count
            res_element['caption'] = [array_to_str(seq[i])]
            res_cider.append(res_element)

            cap_count += 1

    # print(index_z, ended_sequense_list)
    # pdb.set_trace()

    # BLEU:
    _, wbleu4_scores = Weighted_Bleu_scorer.compute_score(gts_att, res, weight_captions)
    if len(calculate_id_list) > 0:
        min_index_b4 = np.where(wbleu4_scores == 0)
        if len(min_index_b4[0]) > 0:
            wbleu4_scores[min_index_b4] = sys.float_info.min
    wbleu1_scores_all[calculate_id_list] = np.array(wbleu4_scores[0])
    wbleu4_scores_all[calculate_id_list] = np.array(wbleu4_scores[3])

    # ROUGE
    _, wrouge_scores = Rouge_scorer.compute_score(gts_att, res)
    if len(calculate_id_list) > 0:
        min_index_rouge = np.where(wrouge_scores == 0)
        if len(min_index_rouge[0]) > 0:
            wrouge_scores[min_index_rouge] = sys.float_info.min
    wrouge_scores_all[calculate_id_list] = np.array(wrouge_scores)

    # METEOR
    if len(calculate_id_list) > 0:
        _, wmeteor_scores = Meteor_scorer.compute_score(gts_att, res)
        if len(calculate_id_list) > 0:
            min_index_meteor = np.where(wmeteor_scores == 0)
            if len(min_index_meteor[0]) > 0:
                wmeteor_scores[min_index_meteor] = sys.float_info.min
        wmeteor_scores_all[calculate_id_list] = np.array(wmeteor_scores)

    # CIDEr
    _, region_cider_scores = CiderD_scorer.compute_score(gts_att, res_cider)
    if len(calculate_id_list) > 0:
        min_index_c = np.where(region_cider_scores == 0)
        if len(min_index_c[0]) > 0:
            region_cider_scores[min_index_c] = 1e-8
    wcider_scores_all[calculate_id_list] = np.array(region_cider_scores)

    return wbleu1_scores_all, wbleu4_scores_all, wcider_scores_all, wrouge_scores_all, wmeteor_scores_all

def get_self_critical_reward_(model, fc_feats, att_feats, att_masks, data, gen_result, opt, internal=None,
                              sim_model=None, bbox=None, sub_att=None, label_region=None, greedy_res=None,
                              D=None, weight_index=None, fixed_region=None, target_model=None):
    weights = model.weights.data.cpu().numpy()
    model_gate = model.gate.data.cpu().numpy()
    if model.training:
        caption_train = True
    else:
        caption_train = False

    if internal is not None:
        if internal.training:
            internal_train = True
            internal.eval()
        else:
            internal_train = False

    # gen_result : result of model(tansaku)
    # greedy_res : result of model (greedy, baseline)
    # reward : cider(gen_result) - cider(greedy_res)
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])
    if greedy_res is None:
        if target_model is None:
            # get greedy decoding baseline
            model.eval()
            with torch.no_grad():
                greedy_res, _, word_exist_seq = model(fc_feats, att_feats, att_masks=att_masks, Critic=internal, sim_model=sim_model,
                                      bbox=bbox, sub_att=sub_att, label_region=label_region, weight_index=weight_index, mode='sample')

            if internal is not None:
                if caption_train and internal_train:
                    model.train()
                    internal.train()
                elif internal_train:
                    internal.train()
                else:
                    model.train()
            else:
                model.train()
        else:
            target_model.eval()
            greedy_res, _ = target_model(fc_feats, att_feats, att_masks=att_masks, Critic=internal, sim_model=sim_model,
                                  bbox=bbox, sub_att=sub_att, label_region=label_region, weight_index=weight_index,
                                  mode='sample')
    else:
        greedy_res = gen_result


    res = OrderedDict()

    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    res_ = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
    res__ = {i: res[i] for i in range(2 * batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}

    model.c_score = 0
    if opt.cider_reward_weight > 0:
        _, c_scores = CiderD_scorer.compute_score(gts, res_, option=opt.cider_option)
        print('Cider scores:', _)
        model.c_score += _
    else:
        c_scores = 0

    if opt.bleu_reward_weight > 0:
        if opt.dataset == 'coco':
            option = 'closest'
        elif opt.dataset == 'vg':
            option = opt.bleu_option
        _, b_scores = Bleu_scorer.compute_score(gts, res__, option=option)
        # _, b_scores = Bleu_scorer.compute_score(gts, res__)
        b_scores = np.array(b_scores[3])
        print('Bleu scores:', _[3])
        model.c_score += b_scores[batch_size:].mean()
        print('Bleu scores train: {}, Bleu scores test: {}'.format(b_scores[:batch_size].mean(), b_scores[batch_size:].mean()))
    else:
        b_scores = 0

    if opt.recall_reward_weight > 0:
        _, recall_scores = Recall_scorer.compute_score(gts, res__)
        recall_scores = np.array(recall_scores[3])
        print('Recall scores:', _[3])
        model.c_score += _[3]
    else:
        recall_scores = 0

    if opt.wbleu_reward_weight > 0:
        gts_for_wb = {}

        r_weights = model.pre_weights_p.data.cpu().numpy().mean(axis=1)
        r_weights_ = model.weights_p.data.cpu().numpy().mean(axis=1)
        r_weights = np.concatenate([r_weights, r_weights_], 0)

        # normalization
        r_weights = preprocessing.minmax_scale(r_weights.astype(float), axis=1)

        label_region = label_region.data.cpu().numpy().astype(np.int32)
        for i in range(len(res__)):
            seq_gt = []
            for j in range(model.pre_weights_p.size(1)):
                if j > len(label_region[(i % batch_size) // seq_per_img])-1:
                    seq_gt.append(array_to_str(label_region[(i % batch_size) // seq_per_img][0]))
                    r_weights[i, j] = 0.0
                else:
                    seq_gt.append(array_to_str(label_region[(i % batch_size) // seq_per_img][j]))
            seq_gt_ = copy.copy(seq_gt)
            gts_for_wb[i] = seq_gt_

        _, wbleu_scores = Weighted_Bleu_scorer.compute_score(gts_for_wb, res__, r_weights)
        wbleu_scores = np.array(wbleu_scores[3])
        print('Weighted bleu scores:', _[3])
        model.c_score += _[3]
    else:
        wbleu_scores = 0

    # if opt.discriminator_weight > 0 and D is not None and model.training:
    if opt.discriminator_weight > 0 and D is not None:
        gen_result_torch = torch.from_numpy(gen_result).cuda()
        greedy_res_torch = torch.from_numpy(greedy_res).cuda()
        res_d = torch.cat((gen_result_torch, greedy_res_torch), 0)
        if opt.cut_length < 0:
            d_score = D(res_d)
            d_score = d_score.data.cpu().numpy()
        elif weight_index is not None and opt.random_disc == 1:
            weight_index_double = np.concatenate([weight_index, weight_index], 0)
            model_gate_eval = model.gate.data.cpu().numpy()
            if model_gate[0].sum() > 0:
                model_gate_double = np.concatenate([model_gate, model_gate_eval], 0)
            else:
                model_gate_double = None
            d_score, _ = D.discriminate_switch_time(res_d, opt.cut_length,
                                                    weight_index=weight_index_double[:, 1, :], model_gate=model_gate_double)
        elif opt.all_switch_dis == 1:
            model_gate_eval = model.gate.data.cpu().numpy()
            if model_gate[0].sum() > 0:
                model_gate_double = np.concatenate([model_gate, model_gate_eval], 0)
            else:
                model_gate_double = None
            d_score, _ = D.discriminate_all_switch(res_d, opt.cut_length, model_gate=model_gate_double)
        elif opt.all_switch_end_dis == 1:
            model_gate_eval = model.gate.data.cpu().numpy()
            if model_gate[0].sum() > 0:
                model_gate_double = np.concatenate([model_gate, model_gate_eval], 0)
            else:
                model_gate_double = None
            d_score, _ = D.discriminate_all_switch_and_end(res_d, opt.cut_length, model_gate=model_gate_double)
        else:
            d_score, _ = D.fixed_length_forward(res_d, opt.cut_length)
        d_scores = d_score[:, 1]
        print('Discriminator scores train: {}, Discriminator scores test: {}'.format(d_scores[:batch_size].mean(), d_scores[batch_size:].mean()))
        model.d_score = d_scores.mean()
    else:
        d_scores = 0
        model.d_score = 0

    if opt.gsim_weight > 0:
        gen_result_torch = torch.from_numpy(gen_result).cuda()
        greedy_res_torch = torch.from_numpy(greedy_res).cuda()
        input_cap_feature = torch.cat((gen_result_torch, greedy_res_torch), 0)
        input_im_feature = fc_feats.repeat(2, 1)
        maekd_att_mapped, maked_cap_mapped = Global_sim_scorer(input_im_feature, input_cap_feature)
        sim_maked = Global_sim_scorer.cal_sim(maekd_att_mapped, maked_cap_mapped)
        sim_maked = sim_maked.cpu().data.numpy()
        print('Sim scores:', sim_maked.mean())
        model.c_score += sim_maked.mean()
    else:
        sim_maked = 0

    if opt.used_area_weight > 0:
        word_exist = (gen_result>0).reshape(gen_result.shape[0], gen_result.shape[1], 1)
        word_exist_eval = (greedy_res>0).reshape(gen_result.shape[0], gen_result.shape[1], 1)
        weights = weights * word_exist
        weights_eval = model.weights.data.cpu().numpy() * word_exist_eval
        used_area_score = cal_used_area_score(weights, fixed_region)
        used_area_score_eval = cal_used_area_score(weights_eval, fixed_region)
        used_area_reward = np.concatenate([used_area_score, used_area_score_eval])
        model.c_score += used_area_score.mean()
    else:
        used_area_reward = 0

    scores = c_scores + b_scores + opt.discriminator_weight * d_scores + recall_scores + \
             wbleu_scores + opt.gsim_weight * sim_maked + opt.used_area_weight * used_area_reward
    if type(scores) == float:
        scores = np.zeros(batch_size * 2)

    # if opt.discriminator_weight == 0.0:
    #     scores_diff = scores[:batch_size] - scores[batch_size:]  # (batchsize, )
    # else:
    #     scores_diff = scores[:batch_size]
    scores_diff = scores[:batch_size] - scores[batch_size:]
    rewards = np.repeat(scores_diff[:, np.newaxis], gen_result.shape[1], 1)  # (batchsize, seq_length)

    return rewards, scores[:batch_size], caption_train, greedy_res

def cal_used_area_score(weights, fixed_region):
    score_sheet = np.zeros((weights.shape[0], 100, 100))
    used_area_batch = np.sum(weights, axis=1) #(150, region_num)
    for i in range(len(used_area_batch)):
        regions = fixed_region[i]
        att_length = min(regions.shape[0], used_area_batch.shape[1])
        used_area = regions[np.where(used_area_batch[i, :att_length]>0)]
        for area in used_area:
            score_sheet[i, area[0]:area[2], area[1]:area[3]] = 1
    score_sheet = np.reshape(score_sheet, (score_sheet.shape[0], -1))
    score = score_sheet.mean(axis=1)
    return score

def get_self_critical_reward_for_actor_critic(model, fc_feats, att_feats, att_masks, data, gen_result, opt, internal=None,
                              sim_model=None, bbox=None, label_region=None, sub_att=None, greedy_res=None, D=None):

    if model.training:
        caption_train = True
    else:
        caption_train = False

    # gen_result : result of model(tansaku)
    # greedy_res : result of model (greedy, baseline)
    # reward : cider(gen_result) - cider(greedy_res)
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])
    if internal is not None:
        internal.reset()

    res = OrderedDict()
    model.c_score = 0
    gen_result = gen_result.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    res_ = [{'image_id': i, 'caption': res[i]} for i in range(batch_size)]
    res__ = {i: res[i] for i in range(batch_size)}
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(batch_size)}

    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_, option=opt.cider_option)
        print('Cider scores:', _)
        model.c_score += _
    else:
        cider_scores = 0

    if opt.recall_reward_weight > 0:
        _, recall_scores = Recall_scorer.compute_score(gts, res__)
        recall_scores = np.array(recall_scores[3])
        print('Recall scores:', _[3])
    else:
        recall_scores = 0

    if opt.bleu_reward_weight > 0:
        if opt.dataset == 'coco':
            option = 'closest'
        elif opt.dataset == 'vg':
            option = opt.bleu_option
        # _, b_scores = Bleu_scorer.compute_score(gts, res__, option=option)
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__, option=option)
        bleu_scores = np.array(bleu_scores[3])
        print('Bleu scores:', _[3])
        model.c_score += bleu_scores.mean()
    else:
        bleu_scores = 0

    if opt.wbleu_reward_weight > 0:
        gts_for_wb = {}
        word_exist = (gen_result > 0).astype(np.int).reshape(gen_result.shape[0], gen_result.shape[1], 1)
        r_weights = model.weights_p.data.cpu().numpy()
        r_weights = r_weights[:, :word_exist.shape[1], :] * word_exist
        r_weights = r_weights.sum(axis=1)/word_exist.sum(1)
        # r_weights = np.concatenate([r_weights, r_weights_], 0)

        # normalization
        # r_weights = preprocessing.minmax_scale(r_weights.astype(float), axis=1)

        label_region = label_region.data.cpu().numpy().astype(np.int32)
        for i in range(len(res__)):
            seq_gt = []
            for j in range(model.pre_weights_p.size(1)):
                if j > len(label_region[(i % batch_size) // seq_per_img])-1:
                    seq_gt.append(array_to_str(label_region[(i % batch_size) // seq_per_img][0]))
                    r_weights[i, j] = 0.0
                else:
                    seq_gt.append(array_to_str(label_region[(i % batch_size) // seq_per_img][j]))
            seq_gt_ = copy.copy(seq_gt)
            gts_for_wb[i] = seq_gt_

        _, wbleu_scores = Weighted_Bleu_scorer.compute_score(gts_for_wb, res__, r_weights)
        wbleu_scores = np.array(wbleu_scores[3])
        print('Weighted bleu scores:', _[3])
        model.c_score += _[3]
    else:
        wbleu_scores = 0

    if opt.recall_reward_weight > 0:
        _, recall_scores = Recall_scorer.compute_score(gts, res__)
        recall_scores = np.array(recall_scores[3])
        print('Recall scores:', _[3])
        model.c_score += _[3]
    else:
        recall_scores = 0

    if opt.discriminator_weight > 0 and D is not None:
        gen_result_cuda = torch.from_numpy(gen_result).cuda()
        d_score = D(gen_result_cuda)
        d_score_ = d_score[:, 0]
        print('Discriminator scores:', d_score_.mean())
        d_scores = d_score_.data.cpu().numpy()
        model.c_score += d_score_.mean() * opt.discriminator_weight
    else:
        d_scores = 0

    # pdb.set_trace()
    scores = cider_scores + bleu_scores + d_scores + recall_scores + wbleu_scores

    scores = scores[:batch_size]  # (batchsize, )
    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)  # (batchsize, seq_length)

    return rewards, scores[:batch_size], caption_train, greedy_res

# calculate attention normalization reward
def get_attnorm_reward(word_exist, weights):
    # gen_result: caption vector (batch, seq_length)
    # weights: weights (batch, seq_length, att_num)
    # weight_num: number of output weights (batch, )

    weights_cpu = weights * torch.from_numpy(word_exist).type(torch.FloatTensor)
    weight_num = weights_cpu.sum(dim=2).sum(dim=1, keepdim=True) # (batch, 1)
    weight_num[np.where(weight_num<1)] = 1
    # keisu is att_num/number_of_weights

    att_reward = 1.0/weights_cpu.size(-1) - torch.sum(weights_cpu, dim=1)/weight_num  # (150, 196)
    att_reward = att_reward * att_reward  # (150, 196)
    att_reward = torch.sum(att_reward, dim=1)  # (150)
    att_reward = att_reward.numpy()
    att_reward = np.repeat(att_reward[:, np.newaxis], word_exist.shape[1], 1)  # (150, 16)

    if att_reward.sum() != att_reward.sum():
        pdb.set_trace()

    reward = -1 * att_reward

    return reward


def get_reward_similarity(sim_model, gen_result, internal):
    pass


def get_next_reward(model, fc_feats, att_feats, att_masks, data, gen_result, opt):
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])
    caption_length = len(gen_result[0])

    res = OrderedDict()
    gen_result = gen_result.data.cpu().numpy()
    for i in range(batch_size * caption_length):
        res[i] = [array_to_str(gen_result[i // caption_length][:(i % caption_length + 1)])]

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    res_ = [{'image_id': i // caption_length, 'caption': res[i]} for i in range(batch_size * caption_length)]
    gts = {i: gts[i // (seq_per_img * caption_length)] for i in range(batch_size * caption_length)}

    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_, option=opt.cider_option)
        ave_scores = np.average(cider_scores[15:][::16])
        print('Cider scores:', ave_scores)
    else:
        cider_scores = 0
    scores = opt.cider_reward_weight * cider_scores
    f = np.zeros(np.shape(scores))
    scores_ = np.zeros(np.shape(scores))
    gamma = 1.0
    for i in range(len(scores)):
        if i % 16 != 0:
            f[i] = scores[i] - scores[i - 1]

    for i in range(len(scores)):
        scores_[i] = np.sum(f[i:(i + 15 - (i % 16))])
    rewards = np.reshape(scores_, (-1, gen_result.shape[1]))
    # rewards = np.reshape(f, (-1, gen_result.shape[1]))

    return rewards


def get_next_reward_baseline(model, fc_feats, att_feats, att_masks, data, gen_result, opt):
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])
    caption_length = len(gen_result[0])

    model.eval()
    with torch.no_grad():
        greedy_res, _ = model(fc_feats, att_feats, att_masks=att_masks, mode='sample')

    model.train()

    res = OrderedDict()
    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    for i in range(batch_size * caption_length):
        res[i] = [array_to_str(gen_result[i // caption_length][:(i % caption_length + 1)])]
    for i in range(batch_size * caption_length):
        res[batch_size * caption_length + i] = [
            array_to_str(greedy_res[i // caption_length][:(i % caption_length + 1)])]

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    res_ = [{'image_id': i // caption_length, 'caption': res[i]} for i in range(2 * batch_size * caption_length)]
    gts = {i: gts[(i // (seq_per_img * caption_length)) % len(data['gts'])] for i in
           range(2 * batch_size * caption_length)}
    # pdb.set_trace()

    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_, option=opt.cider_option)
        ave_scores = np.average(cider_scores[15:][::16])
        print('Cider scores:', ave_scores)
    else:
        cider_scores = 0
    scores = opt.cider_reward_weight * cider_scores
    f = np.zeros(np.shape(scores))
    scores_ = np.zeros(np.shape(scores))
    gamma = 1.0
    for i in range(len(scores)):
        if i % 16 != 0:
            f[i] = scores[i] - scores[i - 1]

    for i in range(len(scores)):
        scores_[i] = np.sum(f[i:(i + 15 - (i % 16))])
    scores__ = scores[:batch_size * caption_length] - scores[batch_size * caption_length:]
    rewards = np.reshape(scores__, (-1, gen_result.shape[1]))
    # rewards = np.reshape(f, (-1, gen_result.shape[1]))

    return rewards