from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# sample

import pdb

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import similality.cos_distance as sim

import time
import os
#
import _pickle as cPickle
import six
import copy

import opts
import models
from dataloader import *
# from similality.sim_dataloader import *
import eval_utils
import misc.utils as utils
import Discriminator.utils as dis_utils
import Discriminator.dataloader_for_dis as dis_dataloader
import models.Critic as critic_utils
from misc.rewards import init_scorer, get_self_critical_reward, get_internal_reward, \
    get_self_critical_and_similarity_reward, get_double_reward, get_self_critical_and_similarity_reward_for_actor_critic
from misc.rewards import cal_avg_sim


# ------------ preparing tensorboard ---------- #
try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

def add_summary_value(writer, key, value, iteration, tag=None):
    if writer:
        if tag is None:
            writer.add_scalar(key, value, iteration)
        else:
            writer.add_scalar(key + '/' + tag, value, iteration)

#-----------------------------------------------#

def train(opt, num_switching=None):
    global internal
    if opt.gpu2 is None:
        torch.cuda.set_device(opt.gpu)
    RL_count = 0
    pure_reward = None

    # Deal with feature things before anything
    opt.use_att = utils.if_use_att(opt.caption_model)
    if opt.use_box: opt.att_feat_size = opt.att_feat_size + 5

    # set dataloder
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length
    opt.baseline_concat = 0

    # setting of record
    result_path = '/mnt/workspace2019/nakamura/selfsequential/log_python3/' + opt.checkpoint_path
    tb_summary_writer = tb and tb.SummaryWriter(result_path)

    infos = {}
    histories = {}


    # --- pretrained model loading --- #
    if opt.start_from is not None:
        opt.start_from = '/mnt/workspace2019/nakamura/selfsequential/log_python3/' + opt.start_from
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        infos = cPickle.load(open(os.path.join(opt.start_from, 'infos_' + opt.id + '.pkl'), mode='rb'))
        saved_model_opt = infos['opt']
        # need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
        need_be_same = ["rnn_type", "rnn_size", "num_layers"]
        for checkme in need_be_same:
            assert vars(saved_model_opt)[checkme] == vars(opt)[
                checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            histories = cPickle.load(open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl') , mode='rb'))
    if opt.sf_epoch is not None and opt.sf_itr is not None:
        iteration = opt.sf_itr
        epoch = opt.sf_epoch
    else:
        iteration = infos.get('iter', 0)
        epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    #---------------------------------------#

    # I forget about these parameter, they maybe are not used.
    b_regressor = None
    opt.regressor = b_regressor

    # model setting
    if opt.gpu2 is not None:
        model = models.setup(opt).cuda()
        dp_model = torch.nn.DataParallel(model)
    else:
        model = models.setup(opt).cuda()
        dp_model = model

    update_lr_flag = True
    # Assure in training mode
    dp_model.train()

    # set rl mode and internal critic and similairty model
    info_json = json.load(open(opt.input_json))
    sim_model = None
    new_internal = None
    if opt.internal_model == 'sim' or opt.internal_model == 'sim_newr'  or opt.internal_model == 'sim_dammy':

        # setting internal critic and similarity prediction network
        sim_model = sim.Sim_model(opt.input_encoding_size, opt.rnn_size, vocab_size=len(info_json['ix_to_word']))

        if opt.region_bleu_flg == 0:
            if opt.sim_pred_type == 0:
                # model_root = '/mnt/workspace2018/nakamura/vg_feature/model_cossim2/model_13_1700.pt'
                model_root = '/mnt/workspace2018/nakamura/vg_feature/model_cossim_bu/model_6_0.pt'
            elif opt.sim_pred_type == 1:
                model_root = '/mnt/workspace2018/nakamura/vg_feature/model_cossim_noshuffle04model_71_1300.pt'
            elif opt.sim_pred_type == 2:
                model_root = '/mnt/workspace2019/nakamura/selfsequential/sim_model/subset_similarity/model_0_3000.pt'
            else:
                print('select 0 or 1')
                exit()
            checkpoint = torch.load(model_root, map_location='cuda:0')
            sim_model.load_state_dict(checkpoint['model_state_dict'])
            sim_model.cuda()
            sim_model.eval()
            for param in sim_model.parameters():
                param.requires_grad = False
            sim_model_optimizer = None
        elif opt.region_bleu_flg == 1:
            sim_model.cuda()
            if opt.sf_internal_epoch is not None:
                sim_model.load_state_dict(
                    torch.load(os.path.join(opt.start_from, 'sim_model_' + str(opt.sf_internal_epoch) + '_' + str(
                        opt.sf_internal_itr) + '.pth')))
                # sim_model_optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'internal_optimizer_' + str(
                #     opt.sf_internal_epoch) + '_' + str(opt.sf_internal_itr) + '.pth')))
            sim_model_optimizer = utils.build_internal_optimizer(sim_model.parameters(), opt)
        else:
            print('not implimented')
            exit()


        if opt.only_critic_train == 1:
            random.seed(100)
        if opt.critic_encode==1:
            internal = models.CriticModel_with_encoder(opt)
        elif opt.bag_flg == 1:
            internal = models.CriticModel_bag(opt)
        elif opt.ppo == 1:
            # internal = models.CriticModel_sim(opt)
            internal = models.CriticModel_nodropout(opt)
            new_internal = models.CriticModel_nodropout(opt)
            internal.load_state_dict(new_internal.state_dict())
        elif opt.input_h_flg == 1:
            internal = models.CriticModel_sim(opt)
        else:
            internal = models.CriticModel_sim_h(opt)

        internal = internal.cuda()
        if new_internal is not None:
            new_internal = new_internal.cuda()

        if opt.ppo == 1:
            internal_optimizer = utils.build_internal_optimizer(new_internal.parameters(), opt)
        else:
            internal_optimizer = utils.build_internal_optimizer(internal.parameters(), opt)

        if opt.sf_internal_epoch is not None:
            internal.load_state_dict(torch.load(os.path.join(opt.start_from,'internal_' + str(opt.sf_internal_epoch) + '_' + str(
                                                                 opt.sf_internal_itr) + '.pth')))
            internal_optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'internal_optimizer_' + str(
                opt.sf_internal_epoch) + '_' + str(opt.sf_internal_itr) + '.pth')))
            # new_internal = models.CriticModel_nodropout(opt)
            new_internal.load_state_dict(torch.load(os.path.join(opt.start_from,'internal_' + str(opt.sf_internal_epoch) + '_' + str(
                                                                 opt.sf_internal_itr) + '.pth')))
        if opt.multi_learn_flg != 1:
            if opt.internal_rl_flg == 1:
                internal_rl_flg = True
                dp_model.eval()
            else:
                internal.eval()
                internal_rl_flg = False
        else:
            internal_rl_flg = True
    else:
        if opt.sim_reward_flg > 0:
            # setting internal critic and similarity prediction network
            sim_model = sim.Sim_model(opt.input_encoding_size, opt.rnn_size, vocab_size=len(info_json['ix_to_word']))
            if opt.sim_pred_type == 0:
                # model_root = '/mnt/workspace2018/nakamura/vg_feature/model_cossim2/model_13_1700.pt'
                # model_root = '/mnt/workspace2018/nakamura/vg_feature/model_cossim_bu/model_6_0.pt'
                model_root = '/mnt/workspace2019/nakamura/selfsequential/sim_model/no_shuffle_simforcoco/model_37_34000.pt'
            elif opt.sim_pred_type == 1:
                model_root = '/mnt/workspace2018/nakamura/vg_feature/model_cossim_noshuffle04model_71_1300.pt'
            elif opt.sim_pred_type == 2:
                model_root = '/mnt/workspace2019/nakamura/selfsequential/sim_model/subset_similarity/model_0_3000.pt'
            else:
                print('select 0 or 1')
                exit()

            if opt.region_bleu_flg == 0:
                if opt.sim_pred_type == 0:
                    # model_root = '/mnt/workspace2018/nakamura/vg_feature/model_cossim2/model_13_1700.pt'
                    opt.sim_model_dir = '/mnt/workspace2018/nakamura/vg_feature/model_cossim_bu/model_6_0.pt'
                elif opt.sim_pred_type == 1:
                    opt.sim_model_dir = '/mnt/workspace2018/nakamura/vg_feature/model_cossim_noshuffle04model_71_1300.pt'
                elif opt.sim_pred_type == 2:
                    opt.sim_model_dir = '/mnt/workspace2019/nakamura/selfsequential/sim_model/subset_similarity/model_0_3000.pt'
                else:
                    opt.sim_model_dir = '/mnt/workspace2019/nakamura/selfsequential/log_python3/log_' + opt.id + '/sim_model' + opt.model[-13:-4] + '.pth'

                checkpoint = torch.load(opt.sim_model_dir, map_location='cuda:0')
                sim_model.load_state_dict(checkpoint['model_state_dict'])
                sim_model.cuda()
                sim_model.eval()
                for param in sim_model.parameters():
                    param.requires_grad = False
                sim_model_optimizer = None
            elif opt.region_bleu_flg == 1:
                sim_model_optimizer = utils.build_internal_optimizer(sim_model.parameters(), opt)
                sim_model.cuda()

        internal = None
        internal_optimizer = None
        internal_rl_flg = False
        opt.c_current_lr = 0
    # opt.internal = internal

    # set Discriminator
    if opt.discriminator_weight > 0:
        dis_opt = opt
        if opt.dis_type == 'coco':
            discrimiantor_model_dir = '/mnt/workspace2018/nakamura/selfsequential/discriminator_log/coco/discriminator_150.pth'
            dis_opt.input_label_h5 = '/mnt/poplin/share/dataset/MSCOCO/cocotalk_coco_for_discriminator_label.h5'
            dis_opt.input_json = '/mnt/poplin/share/dataset/MSCOCO/cocotalk_coco_for_discriminator.json'
        elif opt.dis_type == 'iapr':
            discrimiantor_model_dir = '/mnt/workspace2018/nakamura/selfsequential/discriminator_log/iapr_dict/discriminator_125.pth'
            dis_opt.input_label_h5 = '/mnt/workspace2019/visual_genome_pretrain/iapr_talk_cocodict_label.h5'
            dis_opt.input_json = '/mnt/workspace2018/nakamura/IAPR/iapr_talk_cocodict.json'
        elif opt.dis_type == 'ss':
            discrimiantor_model_dir = '/mnt/workspace2019/nakamura/selfsequential/discriminator_log/shuttorstock_dict/discriminator_900.pth'
            dis_opt.input_label_h5 = '/mnt/workspace2019/nakamura/shutterstock/shuttorstock_talk_cocodict_label.h5'
            dis_opt.input_json = '/mnt/workspace2019/nakamura/shutterstock/shuttorstock_talk_cocodict.json'
        elif opt.dis_type == 'sew':
            discrimiantor_model_dir = '/mnt/workspace2019/nakamura/selfsequential/discriminator_log/sew/discriminator_900.pth'
            dis_opt.input_label_h5 = '/mnt/poplin/share/dataset/simple_english_wikipedia/sew_talk_label.h5'
            dis_opt.input_json = '/mnt/poplin/share/dataset/simple_english_wikipedia/sew_talk.json'
        elif opt.dis_type == 'sew_cut5':
            discrimiantor_model_dir = '/mnt/workspace2019/nakamura/selfsequential/discriminator_log/sew_cut5/discriminator_90.pth'
            dis_opt.input_label_h5 = '/mnt/poplin/share/dataset/simple_english_wikipedia/sew_talk_label.h5'
            dis_opt.input_json = '/mnt/poplin/share/dataset/simple_english_wikipedia/sew_talk.json'
            opt.cut_length = 5
        elif opt.dis_type == 'vg_cut5':
            opt.cut_length = 5
            discrimiantor_model_dir = '/mnt/workspace2019/nakamura/selfsequential/discriminator_log/vg_cut5/discriminator_200.pth'
            dis_opt.input_label_h5 = '/mnt/poplin/share/dataset/MSCOCO/cocotalk_subset_vg_larger_label.h5'
            dis_opt.input_json = '/mnt/poplin/share/dataset/MSCOCO/cocotalk_subset_vg_larger_addregions.json'
        else:
            print('select existing discriminative model!')
            exit()

        discriminator_path_learned = os.path.join(result_path, 'discriminator_{}_{}.pth'.format(epoch, iteration))
        Discriminator = dis_utils.Discriminator(opt)
        if os.path.isfile(discriminator_path_learned):
            Discriminator.load_state_dict(torch.load(discriminator_path_learned, map_location='cuda:' + str(opt.gpu)))
        else:
            Discriminator.load_state_dict(torch.load(discrimiantor_model_dir, map_location='cuda:' + str(opt.gpu)))
        Discriminator = Discriminator.cuda()
        # change discriminator learning rate
        # opt.learning_rate = opt.learning_rate/10
        dis_optimizer = utils.build_optimizer(Discriminator.parameters(), opt)
        # for group in dis_optimizer.param_groups:
        #     group['lr'] = opt.learning_rate/100
        Discriminator.eval()
        dis_loss_func = nn.BCELoss().cuda()
        dis_loader = dis_dataloader.DataLoader(dis_opt)
    else:
        Discriminator = None
        dis_loader = None
        dis_optimizer = None

    # set Acter Critic network
    if opt.actor_critic_flg == 1:
        Q_net = models.Actor_Critic_Net_upper(opt)
        target_Q_net = models.Actor_Critic_Net_upper(opt)
        Q_net.load_state_dict(target_Q_net.state_dict())
        target_model = models.setup(opt).cuda()
        target_model.load_state_dict(model.state_dict())
        target_model.eval()
        Q_net.cuda()
        target_Q_net.cuda()
        Q_net_optimizer = utils.build_optimizer(Q_net.parameters(), opt)
    elif opt.actor_critic_flg == 2:
        Q_net = models.Actor_Critic_Net_seq(opt)
        target_Q_net = models.Actor_Critic_Net_seq(opt)
        Q_net.load_state_dict(target_Q_net.state_dict())
        target_model = models.setup(opt).cuda()
        target_model.load_state_dict(model.state_dict())
        target_model.eval()
        Q_net.cuda()
        target_Q_net.cuda()
        Q_net_optimizer = utils.build_optimizer(Q_net.parameters(), opt)

        seq_mask = torch.zeros((opt.batch_size * opt.seq_per_img, opt.seq_length, opt.seq_length)).cuda().type(torch.cuda.LongTensor)
        for i in range(opt.seq_length):
            seq_mask[:, i, :i] += 1
    elif opt.t_model_flg == 1:
        target_model = models.setup(opt).cuda()
        target_model.load_state_dict(model.state_dict())
        target_model.eval()
    else:
        target_model = None

    baseline = None
    new_model = None
    # set functions calculating loss
    if opt.caption_model == 'hcatt_hard' or opt.caption_model == 'basicxt_hard' or opt.caption_model == 'hcatt_hard_nregion' or opt.caption_model == 'basicxt_hard_nregion' :
        if opt.ppo == 1:
            new_model = models.setup(opt).cuda()
            new_model.load_state_dict(model.state_dict())
            # new_optimizer = utils.build_optimizer(new_model.parameters(), opt)
            # new_model.eval()

        # If you use hard attention, use this setting (but is is not implemented completely)
        crit = utils.LanguageModelCriterion_hard()
    else:
        crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()
    rl_crit_hard = utils.RewardCriterion_hard()
    rl_crit_conly = utils.RewardCriterion_conly()
    rl_crit_hard_base = utils.RewardCriterion_hard_baseline()
    att_crit = utils.AttentionCriterion()

    if opt.caption_model == 'hcatt_hard' and opt.ppo == 1:
        optimizer = utils.build_optimizer(new_model.parameters(), opt)
    else:
        # set optimizer
        optimizer = utils.build_optimizer(model.parameters(), opt)
    # Load the optimizer
    if vars(opt).get('start_from', None) is not None and os.path.isfile(os.path.join(opt.start_from,"optimizer.pth")):
        if opt.sf_epoch is None:
            optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))
        else:
            optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer_' + str(opt.sf_epoch) + '_' +str(opt.sf_itr) + '.pth')))

    critic_train_count = 0
    total_critic_reward = 0
    pre_para = None

    #------------------------------------------------------------------------------------------------------------#
    # training start
    while True:
        train_loss = 0
        if update_lr_flag:
            # cahnge lr
            opt, optimizer, model, internal_optimizer, dis_optimizer = utils.change_lr(opt, epoch, optimizer, model, internal_optimizer, dis_optimizer)

            # If start self critical training
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                sc_flag = True
                # internal_rl_flg == False
                init_scorer(opt.cached_tokens, len(info_json['ix_to_word']))
            else:
                sc_flag = False

            update_lr_flag = False

        # # !!!!!
        # internal_rl_flg = False
        # model.train()
        # internal.eval()
        # #!!!!!

        # Load data from train split (0)
        data = loader.get_batch('train')

        torch.cuda.synchronize()
        start = time.time()

        # get datch
        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks'],
               data['bbox'], data['sub_att'], data['fixed_region']]
        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        fc_feats, att_feats, labels, masks, att_masks, bbox, sub_att, fixed_region = tmp

        optimizer.zero_grad()
        # calculating loss...
        if not sc_flag:
            # use cross entropy
            if opt.weight_deterministic_flg > 0:
                weight_index = np.array(data['weight_index'])
                # fc_feats = fc_feats * 0.0
                output = dp_model(fc_feats, att_feats, labels, att_masks, internal, weight_index=weight_index)
                # output = dp_model(fc_feats, att_feats, labels, att_masks, internal, weight_index=None)
            else:
                output = dp_model(fc_feats, att_feats, labels, att_masks, internal)
            if opt.caption_model == 'hcatt_prob':
                print(torch.exp(output).mean(),  model.probs.mean())
                output = output + model.probs.view(output.size(0), output.size(1), 1)
                loss = crit(output, labels[:,1:], masks[:,1:])
            elif opt.caption_model != 'hcatt_hard' and opt.caption_model != 'hcatt_hard_nregion'and opt.caption_model != 'basicxt_hard_nregion' and opt.caption_model != 'basicxt_hard':
                loss = crit(output, labels[:,1:], masks[:,1:])
            else:
                if baseline is None:
                    baseline = torch.zeros((output.size()[0], output.size()[1]))/output.size()[1]
                    baseline = baseline.cuda()
                    # baseline = torch.log(baseline)
                # print('pre:', baseline.mean().item())
                loss, baseline = crit(output, labels[:,1:], masks[:,1:], baseline, dp_model.weights_p, dp_model.weights)
                # print('after:', baseline.mean().item())
        else:
            # use rl
            if opt.weight_deterministic_flg > 0:
                weight_index = np.array(data['weight_index'])
            else:
                weight_index = None

            if dp_model.training:
                sample_max_flg = 0
            else:
                sample_max_flg = 1

            # get predicted captions and logprops, similarity
            gen_result, sample_logprobs, word_exist_seq = dp_model(fc_feats, att_feats, att_masks,internal,
                                                   opt={'sample_max':sample_max_flg}, sim_model = sim_model, New_Critic=new_internal,
                                                   bbox=bbox, sub_att=sub_att, label_region = data['label_region'], weight_index=weight_index,mode='sample')
            train_similarity = dp_model.similarity

            # ---------- learning discriminator ----------------
            if Discriminator is not None and opt.dis_adv_flg == 1 and internal_rl_flg == False:
                correct = 0
                Discriminator.train()
                fake_data = gen_result.data.cpu()
                hokan = torch.zeros((len(fake_data), 1)).type(torch.LongTensor)
                fake_data = torch.cat((hokan, fake_data, hokan), 1).cuda()
                fake_data = fake_data[:, 1:]
                label = torch.ones((fake_data.size(0))).cuda()
                # pdb.set_trace()
                Discriminator, dis_optimizer, correct, neg_loss = \
                    dis_utils.learning_func(Discriminator, dis_optimizer, fake_data, label, correct, 0, opt.cut_length, opt.random_disc, opt.all_switch_end_dis, opt.all_switch_dis,
                                            loss_func=dis_loss_func, weight_index=weight_index, model_gate=model.gate.data.cpu().numpy())

                dis_data = dis_loader.get_batch('train', batch_size=fake_data.size(0))
                real_data = torch.from_numpy(dis_data['labels']).cuda()
                real_data = real_data[:, 1:]
                Discriminator, dis_optimizer, correct, pos_loss = \
                    dis_utils.learning_func(Discriminator, dis_optimizer, real_data, label, correct, 1, opt.cut_length, 0, 0, 0,
                                            loss_func=dis_loss_func, weight_index=weight_index)

                loss_mean = (pos_loss + neg_loss) / 2
                dis_accuracy = correct/(fake_data.size(0) * 2)
                print('Discriminator loss: {}, accuracy: {}'.format(loss_mean, dis_accuracy))
                Discriminator.eval()
            else:
                loss_mean = -1.0
                dis_accuracy = -1.0
            # --------------------------------------------------


            # ---------- calculate att loss -----------
            if opt.att_reward_flg == 1 and model.training:
            # if opt.att_reward_flg == 1 :
                att_loss = att_crit(model, gen_result.data.cpu().numpy())
                att_loss_num = att_loss.data.cpu().numpy()
            else:
                att_loss = 0.0
                att_loss_num = 0.0
            # ------------------------------------------

            # --- get states and actions xt and weights, ccs, seqs ---
            if opt.actor_critic_flg==1 and model.training:
                xts = model.all_xts
                weights_p = model.weights_p
                ccs = internal.output_action
            if opt.actor_critic_flg == 2 and model.training:
                all_logprops = model.all_logprops
                weight_state = model.state_weights
                # xts = model.all_xts
                gen_result_repeat = gen_result.repeat(1, opt.seq_length).view(all_logprops.size(0), opt.seq_length, opt.seq_length)
                # xts = seq_mask * gen_result_repeat
                xts = gen_result_repeat
                weights_p = model.weights_p
                # pdb.set_trace()
                if internal is not None:
                    ccs = internal.output_action
                else:
                    ccs = torch.zeros((len(xts), weights_p.size(1))).cuda()
            if opt.caption_model == 'hcatt_hard' and opt.ppo==1:
                xts = model.all_xts
                weights_p = model.weights_p
                weights = model.weights
            # ----------------------------------------------------------

            # ---------------- Calculate reward (CIDEr, Discriminator, Similarity...)---------------------
            if opt.actor_critic_flg == 2 and model.training:
                reward, pure_reward = get_self_critical_and_similarity_reward_for_actor_critic(dp_model,
                                                                                                   fc_feats,
                                                                                                   att_feats,
                                                                                                   att_masks, data,
                                                                                                   gen_result, opt,
                                                                                                   train_similarity,
                                                                                                   internal=internal,
                                                                                                   sim_model=sim_model,
                                                                                               label_region=data['label_region'],
                                                                                               D=Discriminator)
            else:
                reward, pure_reward, actor_critic_reward, target_update_flg = get_self_critical_and_similarity_reward(dp_model, fc_feats, att_feats,
                                                                          att_masks, data, gen_result, opt,
                                                                          train_similarity,
                                                                          internal=internal,
                                                                          sim_model=sim_model,
                                                                        label_region=data['label_region'],
                                                                          bbox=bbox,
                                                                        D=Discriminator,
                                                                        weight_index=weight_index,
                                                                        fixed_region=fixed_region,
                                                                        target_model=target_model)
                if target_update_flg and target_model is not None:
                    print('----- target model updated ! -----')
                    target_model.load_state_dict(model.state_dict())

                # print(train_similarity.mean(), model.similarity.mean())
            #----------------------------------------------------------


            #-------------------------------- calculate captioning model loss -----------------------------------------
            #------------ Calculate actor critic loss ----------------
            if opt.actor_critic_flg == 1 and model.training:
                # get q_value
                q_value = Q_net(fc_feats, att_feats, xts, weights_p, gen_result)
                # get target_sample
                with torch.no_grad():
                    gen_result_sample, __ = target_model(fc_feats, att_feats, att_masks,
                                                           seqs=gen_result, ccs=ccs, mode='sample')
                    target_q_value = target_Q_net(fc_feats, att_feats, target_model.all_xts, target_model.weights_p, gen_result)
                # calculate actor critic loss
                actor_critic_loss = Q_net.loss_func(actor_critic_reward, q_value, target_q_value)
                add_summary_value(tb_summary_writer, 'actor_critic_loss', actor_critic_loss.item(), iteration, opt.tag)
                Q_net_optimizer.zero_grad()
            elif opt.actor_critic_flg == 2 and model.training:
                # get q_value
                q_value = Q_net(fc_feats, att_feats, xts, weight_state.detach(), weights_p, all_logprops[:,:-1,:], gen_result)
                # get target_sample
                with torch.no_grad():
                    gen_result_sample, __ = target_model(fc_feats, att_feats, att_masks,
                                                         seqs=gen_result, ccs=ccs, mode='sample', state_weights=weight_state)
                    # pdb.set_trace()
                    target_q_value = target_Q_net(fc_feats, att_feats, xts, target_model.state_weights,
                                                  target_model.weights_p, target_model.all_logprops[:,:-1,:], gen_result)
                # calculate actor critic loss
                if reward is None:
                    pdb.set_trace()
                actor_critic_loss = Q_net.loss_func(reward, q_value, target_q_value, gen_result)
                print('actor_critic_loss', actor_critic_loss.item())
                add_summary_value(tb_summary_writer, 'actor_critic_loss', actor_critic_loss.item(), iteration,
                                  opt.tag)
                Q_net_optimizer.zero_grad()
            else:
                actor_critic_loss = 0

            model.att_score = att_loss_num

            # update ppo old policy
            if new_internal is not None and internal.iteration % 1 == 0:
                internal.load_state_dict(new_internal.state_dict())
            if opt.caption_model == 'hcatt_hard' and opt.ppo == 1:
                model.load_state_dict(new_model.state_dict())

            if not internal_rl_flg or opt.multi_learn_flg == 1:
                # if opt.ppo == 1 and opt.caption_model == 'hcatt_hard':
                # -------------- calculaete self critical loss ---------------
                if False:
                    # get coeffitient and calculate
                    new_gen_result, new_sample_logprobs = new_model(fc_feats, att_feats, att_masks,
                                                         seqs=gen_result,  mode='sample', decided_att=weights)
                    new_model.pre_weights_p = new_model.weights_p
                    new_model.pre_weights = new_model.weights
                    att_index = np.where(weights.data.cpu() > 0)
                    weights_p_ = weights_p[att_index].view(weights_p.size(0), weights_p.size(1))  # (batch, seq_length)
                    reward_coefficient = 1 / (torch.exp(sample_logprobs) * weights_p_).data.cpu()
                    # train caption network get reward and calculate loss
                    reward_loss, baseline = utils.calculate_loss(opt, rl_crit, rl_crit_hard, rl_crit_hard_base,
                                                                 new_sample_logprobs, gen_result, reward,
                                                                 baseline, new_model, reward_coefficient=reward_coefficient)
                elif (not internal_rl_flg or opt.multi_learn_flg == 1) and opt.actor_critic_flg == 0:
                    # train caption network get reward and calculate loss
                    if opt.weight_deterministic_flg == 7:
                        reward_loss, baseline = utils.calculate_loss(opt, rl_crit, rl_crit_hard, rl_crit_hard_base,
                                                                     sample_logprobs, word_exist_seq, reward,
                                                                     baseline, model)
                    else:
                        reward_loss, baseline = utils.calculate_loss(opt, rl_crit, rl_crit_hard, rl_crit_hard_base,
                                                                     sample_logprobs, gen_result, reward,
                                                                     baseline, model)
                else:
                    reward_loss = 0

                # -------------- calculaete self critical loss ---------------
                if (opt.caption_model == 'hcatt_simple' or  opt.caption_model == 'hcatt_simple_switch') and opt.xe_weight > 0.0:
                    output = dp_model(fc_feats, att_feats, labels, att_masks, internal, weight_index=weight_index)
                    xe_loss = crit(output, labels[:, 1:], masks[:, 1:])
                    print('r_loss: {}, xe_loss: {}'.format(reward_loss.item(), xe_loss.item()))
                    add_summary_value(tb_summary_writer, 'xe_loss', xe_loss.item(), iteration, opt.tag)
                    add_summary_value(tb_summary_writer, 'r_loss', reward_loss.item(), iteration, opt.tag)
                else:
                    xe_loss = 0.0

                loss = opt.rloss_weight * reward_loss + opt.att_lambda * att_loss + actor_critic_loss + opt.xe_weight * xe_loss
            # --------------------------------------------------------------------------------------------------------


        # ------------------------- calculate internal critic loss and update ---------------------------
        if internal_optimizer is not None and internal_rl_flg == True and sc_flag:

            internal_optimizer.zero_grad()
            if opt.region_bleu_flg == 1:
                sim_model_optimizer.zero_grad()
            if opt.only_critic_train == 0:
                internal_loss = rl_crit(internal.pre_output, gen_result.data, torch.from_numpy(reward).float().cuda(),
                                    reward_coefficient=internal.pre_reward_coefficient)
            else:
                internal_loss = rl_crit_conly(internal.pre_output, gen_result.data, torch.from_numpy(reward).float().cuda(),
                                        reward_coefficient=internal.pre_reward_coefficient, c_count=critic_train_count)
            q_value_prop = torch.exp(internal.pre_output)
            entropy = torch.mean(-1 * q_value_prop * torch.log2(q_value_prop + 1e-8) + -1 * (1 - q_value_prop) * torch.log2(
                    1 - q_value_prop + 1e-8))

            internal_loss = internal_loss
            internal_loss.backward()
            internal_optimizer.step()
            if opt.region_bleu_flg == 1:
                sim_model_optimizer.step()

            # ----- record loss and reward to tensorboard -----
            # q_value_prop = torch.exp(internal.pre_output)
            # entropy = torch.mean(-1 * q_value_prop * torch.log2(q_value_prop + 1e-8) + -1 * (1 - q_value_prop) * torch.log2(1 - q_value_prop + 1e-8))
            if opt.only_critic_train == 1:
                if internal is not None and sc_flag:
                    num_internal_switching = internal.same_action_flg.mean().item()
                else:
                    num_internal_switching = 0
                total_critic_reward += np.mean(pure_reward)
                total_critic_reward = utils.record_tb_about_critic(model, internal_loss.cpu().data, critic_train_count, opt.tag,
                                                                   tb_summary_writer, reward,
                                                                   pure_reward, entropy,
                                                                   opt.sim_sum_flg,num_internal_switching,
                                                                   total_critic_reward=total_critic_reward)
            else:
                if internal is not None and sc_flag:
                    num_internal_switching = internal.same_action_flg.mean().item()
                else:
                    num_internal_switching = 0
                total_critic_reward = utils.record_tb_about_critic(model, internal_loss.cpu().data, iteration, opt.tag,
                                         tb_summary_writer, reward, pure_reward, entropy, opt.sim_sum_flg, num_internal_switching)
            # -------------------------------------------------

            critic_train_count += 1

            internal.reset()
            internal.iteration+=1

            print('iter {} (epoch {}), internal_loss: {}, avg_reward: {}, entropy: {}'.format(iteration, epoch,internal_loss, reward.mean(), entropy))
        # --------------------------------------------------------------------------------------------------------
        else:
            #------------------------- updating captioning model ----------------------------
            loss.backward()
            utils.clip_gradient(optimizer, opt.grad_clip)
            optimizer.step()
            if opt.actor_critic_flg > 0 and model.training:
                utils.clip_gradient(Q_net_optimizer, opt.grad_clip)
                Q_net_optimizer.step()
                utils.soft_update(target_model, model, 0.001)
                utils.soft_update(target_Q_net, Q_net, 0.001)
                # if iteration % 1000 == 0:
                #     utils.hard_update(target_model, model)
                #     utils.hard_update(target_Q_net, Q_net)
                # else:
                #     utils.soft_update(target_model, model, 0.001)
                #     utils.soft_update(target_Q_net, Q_net, 0.001)

            train_loss = loss.item()
            torch.cuda.synchronize()
            del loss
            end = time.time()
            if internal is not None and sc_flag:
                num_internal_switching = internal.same_action_flg.mean().item()
            else:
                num_internal_switching = 0
            if not sc_flag:
                print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, train_loss, end - start))
            else:
                try:
                    print("iter {} (epoch {}), avg_reward = {:.3f}, att_loss = {:.3f}. time/batch = {:.3f}" \
                        .format(iteration, epoch, np.mean(reward[:,0]), model.att_score.item(), end - start))
                    utils.record_tb_about_model(model, pure_reward, tb_summary_writer, iteration, opt.tag,
                                                opt.sim_sum_flg, loss_mean, dis_accuracy, num_internal_switching)
                except AttributeError:
                    print("iter {} (epoch {}), avg_reward = {:.3f}, att_loss = {:.3f}. time/batch = {:.3f}" \
                          .format(iteration, epoch, np.mean(reward[:, 0]), model.att_score, end - start))
                    utils.record_tb_about_model(model, pure_reward, tb_summary_writer, iteration, opt.tag,
                                                opt.sim_sum_flg, loss_mean, dis_accuracy, num_internal_switching)
                RL_count += 1

            # --------------------------------------------------------------------------------



        # Update the iteration and epoch
        iteration += 1

        # -------------------- change train internal critic or caption network -----------------------------
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True
            if opt.cycle is None and internal is not None and opt.multi_learn_flg != 1:
                # and entropy < 1.0
                if internal_rl_flg == True and opt.only_critic_train == 0:
                    if opt.actor_critic_flg == 1:
                        utils.hard_update(target_model, model)
                        utils.hard_update(target_Q_net, Q_net)
                    internal_rl_flg = False
                    internal.eval()
                    dp_model.train()
                    if weight_index is not None and loader.weight_deterministic_flg == 4:
                        loader.weight_deterministic_flg = 5

                    if opt.region_bleu_flg == 1:
                        sim_model.eval()
                    train_loss = None
                # elif internal_optimizer is not None and internal_rl_flg == False:
                # elif internal_optimizer is not None and internal_rl_flg == False and (epoch + 1) % 3 == 0 and opt.internal_model != 'sim_dammy':
                # elif internal_optimizer is not None and internal_rl_flg == False and opt.internal_model != 'sim_dammy':
                else:
                    internal_rl_flg = True
                    # internal.load_state_dict(torch.load(result_path + '/internal_best.pth'))
                    if opt.ppo == 1:
                        internal_optimizer = optim.Adam(new_internal.parameters(), opt.c_learning_rate,
                                                        weight_decay=1e-5)
                    else:
                        internal_optimizer = optim.Adam(internal.parameters(), opt.c_learning_rate, weight_decay=1e-5)
                    internal.train()
                    if opt.region_bleu_flg == 1:
                        sim_model.train()
                    dp_model.eval()
                    if weight_index is not None and loader.weight_deterministic_flg == 5:
                        loader.weight_deterministic_flg = 4
                    internal.reset()
                    internal.max_r = 0
        # --------------------------------------------------------------------------------------------------

        # ------------------- Write the training loss summary ------------------------------
        if (iteration % opt.losses_log_every == 0) and internal_rl_flg == False and train_loss is not None:
            add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration, opt.tag)
            add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration, opt.tag)
            add_summary_value(tb_summary_writer, 'critic_learning_rate', opt.c_current_lr, iteration, opt.tag)
            add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration, opt.tag)
            if sc_flag:
                add_summary_value(tb_summary_writer, 'avg_reward', np.mean(reward[:,0]), iteration, opt.tag)

            loss_history[iteration] = train_loss if not sc_flag else np.mean(reward[:,0])
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob
        # ----------------------------------------------------------------------------------

        # ------------------------ make evaluation on validation set, and save model ------------------------------
        wdf7_eval_flg = (opt.weight_deterministic_flg != 7 or sc_flag)
        if ((iteration % opt.save_checkpoint_every == 0) or iteration == 39110 or iteration == 113280 or iteration == 151045 or iteration == 78225 or iteration == 31288 or iteration == 32850 or iteration == 46934) and train_loss is not None:
            if sc_flag and (opt.caption_model == 'hcatt_hard' or opt.caption_model == 'basicxt_hard' or opt.caption_model == 'hcatt_hard_nregion' or opt.caption_model == 'basicxt_hard_nregion'):
                if baseline is None:
                    baseline = torch.zeros((sample_logprobs.size()[0], sample_logprobs.size()[1] + 1)) / sample_logprobs.size()[1]
                    baseline = baseline.cuda()
                    # baseline = torch.log(baseline)
            # eval model
            varbose_loss = not sc_flag


            eval_kwargs = {'split': 'val',
                           'internal': internal,
                           'sim_model': sim_model,
                           'caption_model': opt.caption_model,
                           'baseline': baseline,
                           'gts': data['gts'],
                           'dataset': opt.dataset,
                           'verbose_loss': varbose_loss,
                           'weight_deterministic_flg': opt.weight_deterministic_flg
                           }
            eval_kwargs.update(vars(opt))

            # pdb.set_trace()
            if wdf7_eval_flg:
                # eval_utils.eval_writer(dp_model, iteration, loader, tb_summary_writer, eval_kwargs)
                val_loss, predictions, lang_stats = eval_utils.eval_split(dp_model, crit, loader, eval_kwargs)

                # Write validation result into summary
                add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration, opt.tag)
                if lang_stats is not None:
                    for k,v in lang_stats.items():
                        add_summary_value(tb_summary_writer, k, v, iteration, opt.tag)
                val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

                # Save model if is improving on validation result
                if opt.language_eval == 1:
                    current_score = lang_stats['CIDEr']
                else:
                    current_score = - val_loss
            else:
                val_result_history[iteration] = {'loss': None, 'lang_stats': None, 'predictions': None}
                current_score = 0

            best_flag = False
            if True: # if true
                if internal_rl_flg == False:
                    if best_val_score is None or current_score > best_val_score:
                        best_val_score = current_score
                        best_flag = True
                    checkpoint_path = os.path.join(result_path, 'model_{}_{}.pth'.format(epoch, iteration))
                    torch.save(model.state_dict(), checkpoint_path)

                    optimizer_path = os.path.join(result_path, 'optimizer_{}_{}.pth'.format(epoch, iteration))
                    torch.save(optimizer.state_dict(), optimizer_path)
                    print("model saved to {}".format(checkpoint_path))
                    if internal is not None:
                        internal.eval()
                        checkpoint_path = os.path.join(result_path, 'internal_{}_{}.pth'.format(epoch, iteration))
                        torch.save(internal.state_dict(), checkpoint_path)
                        optimizer_path = os.path.join(result_path,
                                                      'internal_optimizer_{}_{}.pth'.format(epoch, iteration))
                        torch.save(internal_optimizer.state_dict(), optimizer_path)
                        print("internal model saved to {}".format(checkpoint_path))
                        checkpoint_path = os.path.join(result_path, 'sim_model_{}_{}.pth'.format(epoch, iteration))
                        torch.save(sim_model.state_dict(), checkpoint_path)
                        print("sim_model saved to {}".format(checkpoint_path))

                else:
                    checkpoint_path = os.path.join(result_path, 'model_{}_{}.pth'.format(epoch, iteration))
                    torch.save(model.state_dict(), checkpoint_path)
                    optimizer_path = os.path.join(result_path, 'optimizer_{}_{}.pth'.format(epoch, iteration))
                    torch.save(optimizer.state_dict(), optimizer_path)
                    print("model saved to {}".format(checkpoint_path))
                    if best_val_score is None or current_score > best_val_score:
                        best_val_score = current_score
                        best_flag = True
                    checkpoint_path = os.path.join(result_path, 'internal_{}_{}.pth'.format(epoch, iteration))
                    torch.save(internal.state_dict(), checkpoint_path)

                    optimizer_path = os.path.join(result_path, 'internal_optimizer_{}_{}.pth'.format(epoch, iteration))
                    torch.save(internal_optimizer.state_dict(), optimizer_path)
                    print("internal model saved to {}".format(checkpoint_path))
                    checkpoint_path = os.path.join(result_path, 'sim_model_{}_{}.pth'.format(epoch, iteration))
                    torch.save(sim_model.state_dict(), checkpoint_path)
                    print("sim_model saved to {}".format(checkpoint_path))
                    dp_model.eval()

                if Discriminator is not None:
                    discriminator_path = os.path.join(result_path, 'discriminator_{}_{}.pth'.format(epoch, iteration))
                    torch.save(Discriminator.state_dict(), discriminator_path)
                    dis_optimizer_path = os.path.join(result_path, 'dis_optimizer_{}_{}.pth'.format(epoch, iteration))
                    torch.save(dis_optimizer.state_dict(), dis_optimizer_path)

                # Dump miscalleous informations
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['split_ix'] = loader.split_ix
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['vocab'] = loader.get_vocab()
                infos['internal_rl_flg'] = internal_rl_flg

                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history

                with open(os.path.join(result_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(result_path, 'histories_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(histories, f)
                if best_flag:
                    checkpoint_path = os.path.join(result_path, 'model-best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))

                    with open(os.path.join(result_path, 'infos_'+opt.id+'-best.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)
                # pdb.set_trace()
        # ---------------------------------------------------------------------------------------------------------

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

opt = opts.parse_opt()

param_dict = vars(opt)
pretty = lambda x: x.replace('_', ' ').capitalize()
text = '\n'.join('  {} = {}'.format(pretty(k), str(v)) for k, v in param_dict.items())
with open('/mnt/workspace2019/nakamura/selfsequential/log_python3/' + opt.checkpoint_path + '/config.txt', 'w') as f:
    f.write(text)

train(opt)

