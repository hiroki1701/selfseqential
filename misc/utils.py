from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from misc.rewards import cal_avg_sim, cal_sum_sim

import sys
sys.path.append("/home/nakamura/project/python3_selfsequential/cider")
sys.path.append("/home/nakamura/project/selfsequential/cider")
from pyciderevalcap.ciderD.ciderD import CiderD
sys.path.append("/home/nakamura/project/python3_selfsequential/coco-caption")
sys.path.append("/home/nakamura/project/selfsequential/coco-caption")
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

from collections import OrderedDict

import pdb
try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

def calculate_loss(opt, rl_crit, rl_crit_hard, rl_crit_hard_base, sample_logprobs, gen_result,
                   reward, baseline, model, reward_coefficient=None):
    if opt.softplus_flg == 1:
        reward = np.log(1 + np.exp(reward))
    if opt.caption_model != 'hcatt_hard' and opt.caption_model != 'hcatt_hard_nregion' and \
                    opt.caption_model != 'basicxt_hard_nregion' and opt.caption_model != 'basicxt_hard':
        loss = rl_crit(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().cuda())
    elif opt.sim_reward_flg == 0:
        if model.p_switch == 1:
            loss = rl_crit_hard(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().cuda(),
                                model.pre_weights_p, model.pre_weights, reward_coefficient=reward_coefficient,
                                p_switch=1)
        else:
            if baseline is None:
                pdb.set_trace()
                baseline = torch.zeros((gen_result.size()[0], gen_result.size()[1])) / gen_result.size()[1]
                baseline = baseline.cuda()
            loss, baseline = rl_crit_hard_base(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().cuda(),
                                     model.pre_weights_p, model.pre_weights, baseline)
    else:
        loss = rl_crit_hard(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().cuda(),
                            model.pre_weights_p, model.pre_weights, reward_coefficient=reward_coefficient, p_switch=None)
    return loss, baseline

def if_use_att(caption_model):
    # Decide if load attention feature according to caption model
    if caption_model in ['show_tell', 'all_img', 'fc']:
        return False
    return True

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                if ix_to_word[str(ix.item())][-1] == '.':
                    txt = txt + ix_to_word[str(ix.item())][:-1]
                else:
                    txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        out.append(txt)
    return out

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

def language_eval_excoco(predictions, predictions_bleu, sents_label_eval, loader):

    Scorer = CiderD()
    Bleu_scorer = Bleu(4)
    METEOR_scorer = Meteor()
    ROUGE_scorer = Rouge()

    c_score, _ = Scorer.compute_score(sents_label_eval, predictions)
    b_score, _ = Bleu_scorer.compute_score(sents_label_eval, predictions_bleu)
    m_score, _ = METEOR_scorer.compute_score(sents_label_eval, predictions_bleu)
    r_score, _ = ROUGE_scorer.compute_score(sents_label_eval, predictions_bleu)

    print('Evaluating {} samples'.format(len(predictions)))

    print('Bleu_1 : ' + str(b_score[0]))
    print('Bleu_2 : ' + str(b_score[1]))
    print('Bleu_3 : ' + str(b_score[2]))
    print('Bleu_4 : ' + str(b_score[3]))
    print('METEOR : ' + str(m_score))
    print('ROUGE_L : ' + str(r_score))
    print('CIDEr : ' + str(c_score))

    lang_stat = {}
    lang_stat['BLEU_1'] = b_score[0]
    lang_stat['BLEU_2'] = b_score[1]
    lang_stat['BLEU_3'] = b_score[2]
    lang_stat['BLEU_4'] = b_score[3]
    lang_stat['METEOR'] = m_score
    lang_stat['ROUGE_L'] = r_score
    lang_stat['CIDEr'] = c_score

    return lang_stat

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward, reward_coefficient=None):
        # input: (batch, seq_length)
        # reward: (batch, seq_length)

        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        if reward_coefficient is not None:
            # use PPO
            epsilon = 0.2
            reward_coefficient = to_contiguous(reward_coefficient).view(-1)
            reward_coefficient = reward_coefficient.type(torch.cuda.FloatTensor)
            input = torch.exp(input) * reward_coefficient
            print(input)
            input_clip = torch.clamp(input, max=1+epsilon, min=1-epsilon)
            output = (-input * reward * mask).view(-1, 1)
            output_clip = (-input_clip * reward * mask).view(-1, 1)
            output, _ = torch.max(torch.cat((output, output_clip), 1), 1)
        else:
            output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class RewardCriterion_conly(nn.Module):
    def __init__(self):
        super(RewardCriterion_conly, self).__init__()

    def forward(self, input, seq, reward, reward_coefficient=None, c_count=10000):
        # input: (batch, seq_length)
        # reward: (batch, seq_length)
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        # mask_curriculum = torch.zeros(seq.size()).cuda()
        # mask_curriculum[:, :c_count//1000+1] = 1
        # mask = mask * mask_curriculum
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        if reward_coefficient is not None:
            # use PPO
            epsilon = 0.2
            reward_coefficient = to_contiguous(reward_coefficient).view(-1)
            reward_coefficient = reward_coefficient.type(torch.cuda.FloatTensor)
            input = torch.exp(input) * reward_coefficient
            print(input)
            input_clip = torch.clamp(input, max=1+epsilon, min=1-epsilon)
            output = (-input * reward * mask).view(-1, 1)
            output_clip = (-input_clip * reward * mask).view(-1, 1)
            output, _ = torch.max(torch.cat((output, output_clip), 1), 1)
        else:
            output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class RewardCriterion_hard(nn.Module):
    def __init__(self):
        super(RewardCriterion_hard, self).__init__()

    def forward(self, input, seq, reward, weight_p, weight, reward_coefficient=None, p_switch=None):
        # input: (batch, seq_length)
        # reward: (batch, seq_length)
        # weight: (batch, att_size, seq_length)
        # weight_p: (batch, att_size, seq_length)


        att_index = np.where(weight.data.cpu() > 0)
        weight_p_ = weight_p[att_index].view(weight_p.size(0), weight_p.size(1)) #(batch, seq_length)
        weight_p_ = to_contiguous(weight_p_).view(-1) #(batch*seq_length)
        blank_index = np.where(weight_p_.data.cpu() == 0.0)
        weight_p_[blank_index] = 1.0
        input = to_contiguous(input).view(-1) #(batch*seq_length)
        reward = to_contiguous(reward).view(-1) #(batch*seq_length)
        mask = (seq > 0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1) #(batch*seq_length)
        if p_switch is not None:
            input = torch.exp(input) * weight_p_
        else:
            input = weight_p_

        if reward_coefficient is not None:
            # use PPO
            epsilon = 0.2
            reward_coefficient = to_contiguous(reward_coefficient).view(-1)
            reward_coefficient = reward_coefficient.type(torch.cuda.FloatTensor)
            input = input * reward_coefficient
            print(input.min().item(), input.max().item())
            input_clip = torch.clamp(input, max=1+epsilon, min=1-epsilon)
            output = (-input * reward * mask).view(-1, 1)
            output_clip = (-input_clip * reward * mask).view(-1, 1)
            output, _ = torch.max(torch.cat((output, output_clip), 1), 1)
        else:
            input = torch.log(input)
            output = - input * reward * mask

        output = torch.sum(output) / torch.sum(mask)

        return output

class RewardCriterion_hard_baseline(nn.Module):
    def __init__(self):
        super(RewardCriterion_hard_baseline, self).__init__()

    def forward(self, input_, seq, reward, weight_p, weight, baseline):
        # print(baseline.size())
        # print(input_.size())
        # print(weight.size())
        # truncate to the same size

        att_index = np.where(weight.data.cpu().numpy() > 0)  # (batch, 17, 36)
        if baseline.size(1) > weight_p.size(1):
            baseline = baseline[:, :weight_p.size(1)]
        weight_p_ = weight_p[att_index].view(weight_p.size(0), weight_p.size(1))  # (batch, seq_length)
        weight_p_ = to_contiguous(weight_p_).view(-1)  # (batch*seq_length)
        blank_index = np.where(weight_p_.data.cpu() == 0.0)
        weight_p_[blank_index] = 1.0
        baseline = to_contiguous(baseline).view(-1)
        input = to_contiguous(input_).view(-1)  # (batch*seq_length)
        reward = to_contiguous(reward).view(-1)  # (batch*seq_length)
        mask = (seq > 0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)  # (batch*seq_length)
        loss_1 = input * mask
        if loss_1.size() == baseline.size():
            coefficient = (loss_1 - baseline).data.cpu().numpy()
        else:
            coefficient = (loss_1).data.cpu().numpy()
        loss_2 = torch.from_numpy(coefficient).cuda() * torch.log(weight_p_ + 1e-8) * mask  # (batch*seq_length)
        output = reward * (loss_1 + 1.0 * loss_2)
        output = -1.0 * torch.sum(output) / torch.sum(mask)  # ()
        # print(loss_1.mean().item(), loss_2.mean().item(), baseline.mean().item(), (torch.log(weight_p_ + 1e-8) * mask).mean().item())

        # loss_1 = input.gather(2, target.unsqueeze(2)).squeeze(2) * mask   # (batch, 17)
        # weight_p_ = weight_p[att_index].view(weight_p.size(0), weight_p.size(1)) #(batch, 17)
        # lambda_r = 1e-6
        # baseline = baseline * 0
        # coefficient = (loss_1 - baseline).data.cpu().numpy()
        # loss_2 = lambda_r * torch.from_numpy(coefficient).cuda() * torch.log(weight_p_ + 1e-8) * mask#(batch, 17)
        # loss_2 = torch.from_numpy(coefficient).cuda() * torch.log(weight_p_ + 1e-8) * mask  # (batch, 17)
        # lambda_h = 1e-6

        # loss_3 = lambda_h * (-1 * weight_p * torch.log2(weight_p)).sum(dim=2) * mask #(batch, 17)
        # output = loss_1 + loss_2 + loss_3 #(batch, 17)
        # output = loss_1 + loss_2
        # output = -1.0 * torch.sum(output) / torch.sum(mask)  # ()

        pre_baseline = torch.sum(loss_1) / torch.sum(mask)
        baseline = baseline * mask
        baseline = torch.sum(baseline) / torch.sum(mask)  # ()
        # print(baseline.mean().item(), pre_baseline.mean().item())
        baseline = 0.9 * baseline + 0.1 * pre_baseline
        baseline = baseline.data.cpu().numpy()
        baseline = torch.from_numpy(baseline).repeat((input_.size(0), input_.size(1))).cuda()  # (batch, 17)

        return output, baseline

class AttentionCriterion(nn.Module):
    def __init__(self):
        super(AttentionCriterion, self).__init__()

    def forward(self, model, gen_result_):
        weights = model.weights_p
        word_exist = (gen_result_ > 0).astype(np.int).reshape(gen_result_.shape[0], gen_result_.shape[1], 1)
        att_score = self.get_attnorm_loss(word_exist, weights)
        att_loss = att_score.mean()

        return att_loss

    def get_attnorm_loss(self, word_exist, weights):
        # gen_result: caption vector (batch, seq_length)
        # weights: weights (batch, seq_length, att_num)
        # weight_num: number of output weights (batch, )

        weights = weights * torch.from_numpy(word_exist).type(torch.cuda.FloatTensor)
        weight_num = weights.sum(dim=2).sum(dim=1, keepdim=True)  # (batch, 1)
        weight_num[np.where(weight_num.data.cpu() < 1)] = 1
        weight_num = weight_num.data.cpu().numpy()
        weight_num = torch.from_numpy(weight_num).type(torch.cuda.FloatTensor)
        # keisu is att_num/number_of_weights

        att_reward = 1.0 / weights.size(-1) - torch.sum(weights, dim=1) / weight_num  # (150, 196)
        att_reward = att_reward * att_reward  # (150, 196)
        att_reward = torch.sum(att_reward, dim=1)  # (150)

        if att_reward.sum() != att_reward.sum():
            pdb.set_trace()

        return att_reward

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size

        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class LanguageModelCriterion_hard(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion_hard, self).__init__()

    def forward(self, input, target, mask, baseline, weight_p, weight):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]

        att_index = np.where(weight.data.cpu().numpy()>0) # (batch, 17, 36)

        loss_1 = input.gather(2, target.unsqueeze(2)).squeeze(2) * mask  # (batch, 17)
        weight_p_ = weight_p[att_index].view(weight_p.size(0), weight_p.size(1))  # (batch, 17)
        if loss_1.size() == baseline.size():
            coefficient = (loss_1 - baseline).data.cpu().numpy()
        else:
            coefficient = (loss_1).data.cpu().numpy()
        loss_2 = torch.from_numpy(coefficient).cuda() * torch.log(weight_p_ + 1e-8) * mask  # (batch, 17)
        output = loss_1 + 1.0*loss_2
        output = -1.0 * torch.sum(output) / torch.sum(mask)  # ()
        # print(loss_1.mean().item(), loss_2.mean().item(), baseline.mean().item(), (torch.log(weight_p_ + 1e-8) * mask).mean().item())

        # loss_1 = input.gather(2, target.unsqueeze(2)).squeeze(2) * mask   # (batch, 17)
        # weight_p_ = weight_p[att_index].view(weight_p.size(0), weight_p.size(1)) #(batch, 17)
        # lambda_r = 1e-6
        # baseline = baseline * 0
        # coefficient = (loss_1 - baseline).data.cpu().numpy()
        # loss_2 = lambda_r * torch.from_numpy(coefficient).cuda() * torch.log(weight_p_ + 1e-8) * mask#(batch, 17)
        # loss_2 = torch.from_numpy(coefficient).cuda() * torch.log(weight_p_ + 1e-8) * mask  # (batch, 17)
        # lambda_h = 1e-6

        # loss_3 = lambda_h * (-1 * weight_p * torch.log2(weight_p)).sum(dim=2) * mask #(batch, 17)
        # output = loss_1 + loss_2 + loss_3 #(batch, 17)
        # output = loss_1 + loss_2
        # output = -1.0 * torch.sum(output) / torch.sum(mask)  # ()

        if loss_1.size() == baseline.size():
            pre_baseline = input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
            pre_baseline = torch.sum(pre_baseline)/torch.sum(mask)
            baseline = baseline * mask
            baseline = torch.sum(baseline) / torch.sum(mask)  # ()
            # print(baseline.mean().item(), pre_baseline.mean().item())
            baseline = 0.9 * baseline + 0.1 * pre_baseline
            baseline = baseline.data.cpu().numpy()
            baseline = torch.from_numpy(baseline).repeat((input.size(0), input.size(1))).cuda() # (batch, 17)

        return output, baseline

# reward probability 
def cal_internal_reward(input, target, mask):
    input_ = torch.exp(input.data.cpu())
    target_ = target[:, :input.size(1)].data.cpu()
    mask_ = mask[:, :input.size(1)].data.cpu()
    output = input_.gather(2, target_.unsqueeze(2)).squeeze(2) * mask_
    for i in range(output.shape[1]):
        if torch.sum(output[:,i]) == 0:
            output = output[:,:i]
            return output
    return output

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    # pdb.set_trace()
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.requires_grad == True and param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def target_update(model_a, model_b,tau):
    model_a.fc_1.weight.data = (1 - tau) * model_a.fc_1.weight.data + tau * model_b.fc_1.weight.data
    model_a.fc_2.weight.data = (1 - tau) * model_a.fc_2.weight.data + tau * model_b.fc_2.weight.data
    model_a.fc_1.bias.data = (1 - tau) * model_a.fc_1.bias.data + tau * model_b.fc_1.bias.data
    model_a.fc_2.bias.data = (1 - tau) * model_a.fc_2.bias.data + tau * model_b.fc_2.bias.data

def target_grad(model_a,model_b):

    model_a.fc_1.weight.grad = model_b.fc_1.weight.grad
    model_a.fc_1.bias.grad = model_b.fc_1.bias.grad
    model_a.fc_2.weight.grad = model_b.fc_2.weight.grad
    model_a.fc_2.bias.grad = model_b.fc_2.bias.grad

def same_target(model_a,model_b):
    model_a.fc_1.weight = model_b.fc_1.weight
    model_a.fc_1.bias = model_b.fc_1.bias
    model_a.fc_2.weight = model_b.fc_2.weight
    model_a.fc_2.bias = model_b.fc_2.bias

def freeze_weight(module,freeze):
    count = 0
    for param in module.parameters():
        # if count == 0:
        #     pdb.set_trace()
        #     count+=1
        param.requires_grad = freeze

# reward baseline regressor
class Regressor(nn.Module):
    def __init__(self,opt):
        super(Regressor, self).__init__()
        self.input_size = opt.rnn_size
        self.hidden_size = 256
        self.output_size = 1
        self.fc_1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc_2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self,h_lang):
        hidden = self.fc_1(h_lang)
        output = self.fc_2(hidden)
        return output

def build_optimizer(params, opt):
    if opt.optim == 'rmsprop':
        return optim.RMSprop(params, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, weight_decay=opt.weight_decay)
    elif opt.optim == 'adagrad':
        return optim.Adagrad(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdm':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdmom':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay, nesterov=True)
    elif opt.optim == 'adam':
        return optim.Adam(filter(lambda p: p.requires_grad, params), opt.learning_rate, (opt.optim_alpha, opt.optim_beta),
                          opt.optim_epsilon, weight_decay=opt.weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))

def build_internal_optimizer(params, opt):
    if opt.c_optim == 'rmsprop':
        return optim.RMSprop(params, opt.c_learning_rate, opt.c_optim_alpha, opt.c_optim_epsilon, weight_decay=opt.c_weight_decay)
    elif opt.c_optim == 'adagrad':
        return optim.Adagrad(params, opt.c_learning_rate, weight_decay=opt.c_weight_decay)
    elif opt.c_optim == 'sgd':
        return optim.SGD(params, opt.c_learning_rate, weight_decay=opt.c_weight_decay)
    elif opt.c_optim == 'sgdm':
        return optim.SGD(params, opt.c_learning_rate, opt.c_optim_alpha, weight_decay=opt.c_weight_decay)
    elif opt.c_optim == 'sgdmom':
        return optim.SGD(params, opt.c_learning_rate, opt.c_optim_alpha, weight_decay=opt.c_weight_decay, nesterov=True)
    elif opt.c_optim == 'adam':
        return optim.Adam(filter(lambda p: p.requires_grad, params), opt.c_learning_rate, (opt.c_optim_alpha, opt.c_optim_beta),
                          opt.c_optim_epsilon, weight_decay=opt.c_weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(opt.c_optim))

def change_lr(opt, epoch, optimizer, model, internal_optimizer, dis_optimizer):
    if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
        frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
        decay_factor = opt.learning_rate_decay_rate ** frac
        opt.current_lr = opt.learning_rate * decay_factor
    else:
        opt.current_lr = opt.learning_rate
    set_lr(optimizer, opt.current_lr)
    # Assign the scheduled sampling prob
    if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
        frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
        opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
        model.ss_prob = opt.ss_prob

    if internal_optimizer is not None:
        if epoch > opt.c_learning_rate_decay_start and opt.c_learning_rate_decay_start >= 0:
            frac = (epoch - opt.c_learning_rate_decay_start) // opt.c_learning_rate_decay_every
            decay_factor = opt.c_learning_rate_decay_rate ** frac
            opt.c_current_lr = opt.c_learning_rate * decay_factor
        else:
            opt.c_current_lr = opt.c_learning_rate
        set_lr(internal_optimizer, opt.c_current_lr)

    # if dis_optimizer is not None:
    #     if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
    #         frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
    #         decay_factor = opt.learning_rate_decay_rate ** frac
    #         opt.current_lr = opt.learning_rate * decay_factor
    #     else:
    #         opt.current_lr = opt.learning_rate
    #     set_lr(dis_optimizer, opt.current_lr)

    return opt, optimizer, model, internal_optimizer, dis_optimizer

def record_tb_about_critic(model, internal_loss, iteration, tag, tb_summary_writer, reward, pure_reward,
                           entropy, sim_sum_flg, num_internal_switching, total_critic_reward=None):
    if model.similarity is not None:
        if sim_sum_flg == 1:
            avg_sim = cal_sum_sim(model.similarity)
        else:
            avg_sim = cal_avg_sim(model.similarity)
    else:
        avg_sim = None
    
    # record loss and reward to tensorboard
    internal_loss = torch.sum(internal_loss)
    add_summary_value(tb_summary_writer, 'internal_loss', internal_loss, iteration, tag)
    add_summary_value(tb_summary_writer, 'ave_reward', reward.mean(), iteration, tag)

    if pure_reward is not None:
        add_summary_value(tb_summary_writer, 'internal_reward', np.mean(pure_reward), iteration, tag)
        add_summary_value(tb_summary_writer, 'max_internal_reward', np.max(pure_reward), iteration, tag)
        add_summary_value(tb_summary_writer, 'min_internal_reward', np.min(pure_reward), iteration, tag)
    if model.similarity is not None:
        add_summary_value(tb_summary_writer, 'similarity', np.mean(avg_sim), iteration, tag)
    if model.c_score is not None:
        add_summary_value(tb_summary_writer, 'score', model.c_score, iteration, tag)
    if model.att_score is not None:
        add_summary_value(tb_summary_writer, 'att_score', model.att_score, iteration, tag)
    if total_critic_reward is not None:
        if iteration == 250:
            add_summary_value(tb_summary_writer, 'mean_internal_reward', total_critic_reward/251, 0, tag)
            total_critic_reward = 0.0
        elif (iteration - 250) % 500 == 0:
            add_summary_value(tb_summary_writer, 'mean_internal_reward', total_critic_reward/500, iteration - 250, tag)
            total_critic_reward = 0.0
    add_summary_value(tb_summary_writer, 'entropy', entropy, iteration, tag)
    add_summary_value(tb_summary_writer, 'num_switching', num_internal_switching, iteration, tag)

    return total_critic_reward
        
def record_tb_about_model(model, pure_reward, tb_summary_writer, iteration, tag, sim_sum_flg,
                          loss_mean, dis_accuracy, num_internal_switching):
    if model.similarity is not None:
        if sim_sum_flg == 1:
            avg_sim = cal_sum_sim(model.similarity)
        else:
            avg_sim = cal_avg_sim(model.similarity)
    else:
        avg_sim = None

    if pure_reward is not None:
        add_summary_value(tb_summary_writer, 'captioning_reward', np.mean(pure_reward), iteration, tag)
        add_summary_value(tb_summary_writer, 'max_captioning_reward', np.max(pure_reward), iteration, tag)
        add_summary_value(tb_summary_writer, 'min_captioning_reward', np.min(pure_reward), iteration, tag)
    if model.similarity is not None:
        add_summary_value(tb_summary_writer, 'similarity', np.mean(avg_sim),
                          iteration, tag)
    if model.c_score is not None:
        add_summary_value(tb_summary_writer, 'score', model.c_score, iteration, tag)
    if model.att_score is not None:
        add_summary_value(tb_summary_writer, 'att_score', model.att_score, iteration, tag)
    if model.d_score is not None:
        add_summary_value(tb_summary_writer, 'd_score', model.d_score, iteration, tag)
    if loss_mean > 0:
        add_summary_value(tb_summary_writer, 'Discriminator loss', loss_mean, iteration, tag)
        add_summary_value(tb_summary_writer, 'Discriminator acc', dis_accuracy, iteration, tag)
    weight_p_predicted = model.weights_p[:, ::4, :].data.cpu().numpy()
    weight_entropy = -1 * weight_p_predicted * np.log2(weight_p_predicted)
    weight_entropy = weight_entropy.sum(axis=-1).mean()
    add_summary_value(tb_summary_writer, 'weight_entropy', weight_entropy, iteration, tag)
    add_summary_value(tb_summary_writer, 'num_switching', num_internal_switching, iteration, tag)

def add_summary_value(writer, key, value, iteration, tag=None):
    if writer:
        if tag is None:
            writer.add_scalar(key, value, iteration)
        else:
            writer.add_scalar(key + '/' + tag, value, iteration)
