import argparse
import os

def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_json', type=str, default= dir + 'data/cocotalk.json',
    #                 help='path to the json file containing additional info and vocab')
    parser.add_argument('--type', type=str, default='train', help='you can select train, eval or initial')
    parser.add_argument('--bu', type=int, default= 1)
    parser.add_argument('--id', type=str, default='sample')
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--itr', type=int, default=113280, help='if you use 20 epoch ver, set 75520. '
                                                                'use 40 epoch ver, set 151045')
    parser.add_argument('--internal_model', type=str, default='sim')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpu2', type=int, default=None)
    parser.add_argument('--data', type=str, default='coco')
    parser.add_argument('--part', type=str, default='')

    # not use critic
    parser.add_argument('--base_flg', type=int, default=0)

    # only train '--learning_rate_decay_start 0 ' \
    parser.add_argument('--caption_model', type=str, default='hcatt')
    parser.add_argument('--sim_reward_flg', type=int, default=1)
    parser.add_argument('--att_reward_flg', type=int, default=0)
    parser.add_argument('--att_lambda', type=float, default=0)
    parser.add_argument('--self_critical_after', type=int, default=20)
    parser.add_argument('--sf_internal', type=int, default=0)
    parser.add_argument('--internal_rl_flg', type=int, default=0)
    parser.add_argument('--val_images_use', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--lr_ds', type=int, default=-1)
    parser.add_argument('--c_lr', type=float, default=1e-5)
    parser.add_argument('--c_lr_ds', type=int, default=-1)
    parser.add_argument('--b_w', type=float, default=0.0)
    parser.add_argument('--c_w', type=float, default=0.0)
    parser.add_argument('--d_w', type=float, default=0.0)
    parser.add_argument('--r_w', type=float, default=0.0)
    parser.add_argument('--gs_w', type=float, default=0.0)
    parser.add_argument('--wb_w', type=float, default=0.0)
    parser.add_argument('--xe_w', type=float, default=0.0)
    parser.add_argument('--rloss_w', type=float, default=1.0)
    parser.add_argument('--used_area_w', type=float, default=0.0)
    parser.add_argument('--c_c_w', type=float, default=None)
    parser.add_argument('--l_score', type=float, default=None)
    parser.add_argument('--log_flg', type=int, default=None)
    parser.add_argument('--sig_flg', type=int, default=None)
    parser.add_argument('--sp_flg', type=int, default=None)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--h_flg', type=int, default=1)
    parser.add_argument('--cycle', type=int, default=None)
    parser.add_argument('--critic_encode', type=int, default=0)
    parser.add_argument('--multi_learn_flg', type=int, default=0)
    parser.add_argument('--only_critic_train', type=int, default=0)
    parser.add_argument('--critic_probabilistic', type=int, default=1)
    parser.add_argument('--ppo_flg', type=int, default=0)
    parser.add_argument('--ppo', type=int, default=1)
    parser.add_argument('--bag_flg', type=int, default=0)
    parser.add_argument('--penalty_type', type=str, default='nashi', help='select from cases or compare')
    parser.add_argument('--pfh', type=int, default=0)
    parser.add_argument('--sim_sum_flg', type=int, default=0)
    parser.add_argument('--actor_critic_flg', type=int, default=0)
    parser.add_argument('--region_bleu_flg', type=int, default=1)
    parser.add_argument('--bleu_option', type=str, default='closest')
    parser.add_argument('--cider_option', type=str, default=None)
    parser.add_argument('--dis_adv_flg', type=int, default=1)
    parser.add_argument('--dis_type', type=str, default='sew')
    parser.add_argument('--cr_w', type=float, default=0.0)
    parser.add_argument('--use_weight_probability', type=int, default=0)
    parser.add_argument('--no_local_reward', type=int, default=0)
    parser.add_argument('--max_att_len', type=int, default=36)
    parser.add_argument('--weight_deterministic_flg', type=int, default=0)
    parser.add_argument('--cut_length', type=int, default=-1)
    parser.add_argument('--area_feature_use', type=int, default=0)
    parser.add_argument('--softplus_flg', type=int, default=0)
    parser.add_argument('--max_epochs', type=int, default=300)
    parser.add_argument('--random_disc', type=int, default=0)
    parser.add_argument('--use_next_region', type=int, default=0)
    parser.add_argument('--seq_length', type=int, default=20)
    parser.add_argument('--t_model_flg', type=int, default=0, help='use target model to calclate reward')

    # only hard
    parser.add_argument('--sum_reward_rate', type=float, default=0.0)
    parser.add_argument('--p_switch', type=int, default=0)

    # only eval
    parser.add_argument('--output', type=int, default=1)
    parser.add_argument('--num_images', type=int, default=-1)
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--use_region_similarity', type=int, default=0)
    parser.add_argument('--discriminator_flg', type=int, default=0)
    parser.add_argument('--selected_region_file', type=str, default='/mnt/workspace2018/nakamura/selfsequential/data/allvg_selected_region_info_165.json')
    parser.add_argument('--whole_att_flg', type=int, default=0)
    parser.add_argument('--baseline_concat', type=int, default=0)

    args = parser.parse_args()

    return args

def image_features_info(text):
    if opt.bu == 1:
        # text += '--input_fc_dir /mnt/poplin/share/2018/nakamura_M1/self_sequential/data_old/cocotalk_fc ' \
        #         '--input_att_dir /mnt/workspace2019/nakamura/selfsequential/data/cocotalk_subset_att '
        text += '--input_fc_dir /mnt/workspace2019/nakamura/vg_feature/vg_resnet101_feature_fc ' \
                '--input_att_dir /mnt/workspace2019/nakamura/vg_feature/vg_resnet101_feature_att/_concat '

    return text

def image_features_info_test(text):
    if opt.bu == 1:
        # text += '--input_fc_dir /mnt/poplin/share/2018/nakamura_M1/self_sequential/data_old/cocotalk_fc ' \
        #         '--input_att_dir /mnt/workspace2019/nakamura/selfsequential/data/cocotalk_subset_att '
        text += '--input_fc_dir /mnt/poplin/share/2018/nakamura_M1/self_sequential/data_old/cocotalk_fc ' \
                '--input_att_dir /mnt/workspace2019/nakamura/selfsequential/data/cocotalk_subset_att_larger_test2 '

    return text

def make_train_script(opt):
    if opt.data == 'coco':
        data_ = 'mscoco'
    else:
        data_ = opt.data

    # '--input_json /mnt/poplin/share/dataset/MSCOCO/cocotalk_subset_mscoco_'+ opt.part +'.json ' \
    # '--input_label_h5 /mnt/poplin/share/dataset/MSCOCO/cocotalk_subset_' + data_ + '__label.h5 ' \
    # '--input_json /mnt/poplin/share/dataset/MSCOCO/cocotalk_subset_vg_larger.json ' \

    text = 'python train.py ' \
           '--id ' + opt.id + ' ' \
           '--caption_model ' + opt.caption_model + ' ' \
           '--input_json /mnt/poplin/share/dataset/MSCOCO/cocotalk_subset_vg_larger_addregions.json ' \
           '--input_label_h5 /mnt/poplin/share/dataset/MSCOCO/cocotalk_subset_' + data_ + '_larger_label_all.h5 '\
           '--dataset ' + opt.data + ' ' \
           '--batch_size 30 ' \
           '--learning_rate '+ str(opt.lr)  + ' ' \
           '--learning_rate_decay_start ' + str(opt.lr_ds) + ' ' \
           '--learning_rate_decay_every 6 ' \
           '--save_checkpoint_every 3000 ' \
           '--language_eval 1 ' \
           '--val_images_use ' + str(opt.val_images_use) + ' ' \
           '--max_epochs ' + str(opt.max_epochs) + ' ' \
           '--checkpoint_path log_' + opt.id + ' ' \
           '--start_from log_' + opt.id + ' ' \
           '--bleu_reward_weight ' + str(opt.b_w) + ' ' \
           '--cider_reward_weight ' + str(opt.c_w) + ' ' \
           '--recall_reward_weight ' + str(opt.r_w) + ' ' \
           '--discriminator_weight ' + str(opt.d_w) + ' ' \
           '--wbleu_reward_weight ' + str(opt.wb_w) + ' ' \
           '--rloss_weight ' + str(opt.rloss_w) + ' ' \
           '--xe_weight ' + str(opt.xe_w) + ' ' \
           '--gsim_weight ' + str(opt.gs_w) + ' ' \
           '--sf_epoch ' + str(opt.epoch) + ' ' \
           '--sf_itr ' + str(opt.itr) + ' ' \
           '--gpu ' + str(opt.gpu) + ' ' \
           '--att_reward_flg ' + str(opt.att_reward_flg) + ' ' \
           '--sim_reward_flg ' + str(opt.sim_reward_flg) + ' ' \
           '--prohibit_flg_hard ' + str(opt.pfh) + ' ' \
           '--att_lambda ' + str(opt.att_lambda) + ' ' \
           '--sim_sum_flg ' + str(opt.sim_sum_flg) + ' ' \
           '--ppo ' + str(opt.ppo) + ' ' \
           '--actor_critic_flg ' + str(opt.actor_critic_flg) + ' ' \
           '--sum_reward_rate ' + str(opt.sum_reward_rate) + ' ' \
           '--region_bleu_flg ' + str(opt.region_bleu_flg) + ' ' \
           '--bleu_option ' + str(opt.bleu_option) + ' ' \
           '--dis_adv_flg ' + str(opt.dis_adv_flg) + ' ' \
           '--dis_type ' + opt.dis_type + ' ' \
           '--use_weight_probability ' + str(opt.use_weight_probability) + ' ' \
           '--no_local_reward ' + str(opt.no_local_reward) + ' ' \
           '--max_att_len ' + str(opt.max_att_len) + ' ' \
           '--cut_length ' + str(opt.cut_length) + ' ' \
           '--random_disc ' + str(opt.random_disc) + ' ' \
           '--weight_deterministic_flg ' + str(opt.weight_deterministic_flg) + ' ' \
           '--p_switch ' + str(opt.p_switch) + ' ' \
           '--used_area_weight ' + str(opt.used_area_w) + ' ' \
           '--area_feature_use ' + str(opt.area_feature_use) + ' ' \
           '--softplus_flg ' + str(opt.softplus_flg) + ' ' \
           '--selected_region_file ' + str(opt.selected_region_file) + ' ' \
           '--critic_weight ' + str(opt.cr_w) + ' ' \
           '--use_next_region ' + str(opt.use_next_region) + ' ' \
           '--seq_length ' + str(opt.seq_length) + ' ' \
           '--t_model_flg ' + str(opt.t_model_flg) + ' '


    if opt.base_flg == 0:
        text += '--c_learning_rate ' + str(opt.c_lr) + ' ' \
                '--c_learning_rate_decay_start ' + str(opt.c_lr_ds) + ' ' \
                '--internal_rl_flg ' + str(opt.internal_rl_flg) + ' ' \
                '--input_h_flg ' + str(opt.h_flg) + ' ' \
                '--critic_encode ' + str(opt.critic_encode) + ' ' \
                '--multi_learn_flg ' + str(opt.multi_learn_flg) + ' ' \
                '--only_critic_train ' + str(opt.only_critic_train) + ' ' \
                '--critic_probabilistic ' + str(opt.critic_probabilistic) + ' ' \
                '--ppo_flg ' + str(opt.ppo_flg) + ' ' \
                '--bag_flg ' + str(opt.bag_flg) + ' ' \
                '--penalty_type ' + opt.penalty_type + ' '


        if opt.cycle is not None:
            text += '--cycle ' + str(opt.cycle) + ' '

        if opt.c_c_w is not None:
            text += '--critic_cider_reward_weight ' + str(opt.c_c_w) + ' '

        if opt.internal_model is not None:
            text += '--internal_model ' + str(opt.internal_model) + ' '

        if opt.sf_internal == 1:
            text += '--sf_internal_epoch ' + str(opt.epoch) + ' ' \
                    '--sf_internal_itr ' + str(opt.itr) + ' ' \
                    '--sim_pred_type 2 '

    if opt.bu == 1:
        text = image_features_info(text)
    if opt.itr == 75520:
        text += '--self_critical_after 20  '
    elif opt.itr == 113280:
        text += '--self_critical_after 30  '
    elif opt.itr == 46934:
        text += '--self_critical_after 30  '
    else:
        text += '--self_critical_after 20  '

    if opt.cider_option is not None:
        text += '--cider_option ' + opt.cider_option + ' '

    if opt.l_score is not None:
        text += '--l_score ' + str(opt.l_score) + ' '
    if opt.log_flg is not None:
        text += '--log_flg ' + str(opt.log_flg) + ' '
    if opt.sig_flg is not None:
        text += '--sig_flg ' + str(opt.sig_flg) + ' '
    if opt.sp_flg is not None:
        text += '--separate_reward ' + str(opt.sp_flg) + ' '
    if opt.tag is not None:
        text += '--tag ' + opt.tag + ' '

    return text

def make_eval_script(opt):
    if opt.data == 'coco':
        data_ = 'mscoco'
    else:
        data_ = opt.data
    text = 'python eval.py ' \
           '--id ' + opt.id + ' ' \
           '--model log_' + opt.id + '/model_' + str(opt.epoch) + '_' + str(opt.itr) + '.pth ' \
           '--infos_path log_' + opt.id + '/infos_' + opt.id + '.pkl ' \
           '--input_json /mnt/workspace2018/nakamura/selfsequential/data/cocotalk_allvg07_larger.json ' \
           '--input_label_h5 /mnt/workspace2018/nakamura/selfsequential/data/cocotalk_allvg07_larger_label.h5 ' \
           '--beam_size 1 ' \
           '--dump_images 0 ' \
           '--language_eval 1 ' \
           '--num_images ' + str(opt.num_images) + ' ' \
           '--output ' + str(opt.output) + ' ' \
           '--gpu ' + str(opt.gpu) + ' ' \
           '--prohibit_flg_hard ' + str(opt.pfh) + ' ' \
           '--beam_size ' + str(opt.beam_size) + ' ' \
           '--region_bleu_flg ' + str(opt.region_bleu_flg) + ' ' \
           '--dataset ' + opt.data + ' ' \
           '--max_att_len ' + str(opt.max_att_len) + ' ' \
           '--use_weight_probability ' + str(opt.use_weight_probability) + ' ' \
           '--weight_deterministic_flg ' + str(opt.weight_deterministic_flg) + ' ' \
           '--p_switch ' + str(opt.p_switch) + ' ' \
           '--selected_region_file ' + str(opt.selected_region_file) + ' ' \
           '--whole_att_flg ' + str(opt.whole_att_flg) + ' ' \
           '--baseline_concat ' + str(opt.baseline_concat) + ' ' \
           '--use_next_region ' + str(opt.use_next_region) + ' ' \
           '--seq_length ' + str(opt.seq_length) + ' '

    if opt.discriminator_flg == 1:
        text += '--discriminator log_' + opt.id + '/discriminator_' + str(opt.epoch) + '_' + str(opt.itr) + '.pth '


    if opt.bu == 1:
        if opt.use_region_similarity == 0:
            text = image_features_info_test(text)
        else:
            text = image_features_info(text)

    if opt.gpu2 is not None:
        text += '--gpu2 ' + str(opt.gpu2) + ' '

    if opt.internal_model == 'sim':
        text += '--internal_model ' + str(opt.internal_model) + ' ' \
                '--internal_dir log_' + opt.id + '/internal_' + str(opt.epoch) + '_' + str(opt.itr) + '.pth '
    elif opt.internal_model == 'sim_dammy':
        text += '--internal_model ' + str(opt.internal_model) + ' '

    return text

def make_initial_script(opt):
    if opt.data == 'coco':
        data_ = 'mscoco'
    else:
        data_ = opt.data
    text = 'python train.py ' \
           '--id ' + opt.id + ' ' \
            '--caption_model ' + opt.caption_model + ' ' \
            '--input_json /mnt/workspace2018/nakamura/selfsequential/data/cocotalk_allvg07_larger.json ' \
           '--input_label_h5 /mnt/workspace2018/nakamura/selfsequential/data/cocotalk_allvg07_larger_label.h5 '\
            '--batch_size 30 ' \
            '--learning_rate 5e-4 ' \
            '--learning_rate_decay_start 0 ' \
            '--scheduled_sampling_start 0 ' \
            '--save_checkpoint_every 3000 ' \
            '--language_eval 0 ' \
            '--val_images_use ' + str(opt.val_images_use) + ' ' \
            '--self_critical_after 30  ' \
            '--max_epochs 30 ' \
            '--checkpoint_path log_' + opt.id + ' ' \
            '--weight_deterministic_flg ' + str(opt.weight_deterministic_flg) + ' ' \
            '--gpu ' + str(opt.gpu) + ' ' \
            '--max_att_len ' + str(opt.max_att_len) + ' ' \
            '--area_feature_use ' + str(opt.area_feature_use) + ' ' \
            '--use_next_region ' + str(opt.use_next_region) + ' ' \
            '--selected_region_file ' + opt.selected_region_file + ' ' \
            '--seq_length ' + str(opt.seq_length) + ' ' \
            '--no_local_reward 1 '
    # area_Feature : axis of

    if opt.bu == 1:
        text = image_features_info(text)

    return text

if __name__ == '__main__':
    opt = parse_opt()

    if opt.epoch is None:
        opt.epoch = int(opt.itr / 1564.45)

    if opt.epoch % 2 == 0:
        opt.internal_rl_flg = 1

    if opt.type == 'train':
        text = make_train_script(opt)
    elif opt.type == 'initial':
        text = make_initial_script(opt)
    elif opt.type == 'eval':
        text = make_eval_script(opt)
    else:
        print('Select type from train,initial and eval')

    with open('run.sh', 'w') as f:
        f.write(text)

    if not os.path.isdir('/mnt/workspace2019/nakamura/selfsequential/log_python3/log_' + opt.id + '/'):
        os.mkdir('/mnt/workspace2019/nakamura/selfsequential/log_python3/log_' + opt.id + '/')
    with open('/mnt/workspace2019/nakamura/selfsequential/log_python3/log_' + opt.id + '/' + opt.type + '_run.sh', 'w') as f:
        f.write(text)
    with open('/mnt/workspace2019/nakamura/selfsequential/log_python3/log_' + opt.id + '/' + opt.type + '_run.txt', 'w') as f:
        f.write(text)





