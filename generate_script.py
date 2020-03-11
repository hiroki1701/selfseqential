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

    # not use critic
    parser.add_argument('--base_flg', type=int, default=0)

    # only train '--learning_rate_decay_start 0 ' \
    parser.add_argument('--caption_model', type=str, default='hcatt')
    parser.add_argument('--sim_reward_flg', type=int, default=1)
    parser.add_argument('--att_reward_flg', type=int, default=0)
    parser.add_argument('--att_lambda', type=float, default=0)
    parser.add_argument('--self_critical_after', type=int, default=30)
    parser.add_argument('--sf_internal', type=int, default=0)
    parser.add_argument('--internal_rl_flg', type=int, default=0)
    parser.add_argument('--val_images_use', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--lr_ds', type=int, default=-1)
    parser.add_argument('--c_lr', type=float, default=1e-5)
    parser.add_argument('--c_lr_ds', type=int, default=-1)
    parser.add_argument('--c_w', type=float, default=0.5)
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
    parser.add_argument('--penalty_type', type=str, default='cases', help='select from cases or compare')
    parser.add_argument('--pfh', type=int, default=0)
    parser.add_argument('--sim_sum_flg', type=int, default=0)
    parser.add_argument('--actor_critic_flg', type=int, default=0)
    parser.add_argument('--weight_deterministic_flg', type=int, default=0)
    parser.add_argument('--use_weight_probability', type=int, default=0)
    parser.add_argument('--cr_w', type=float, default=0.0)

    # only hard
    parser.add_argument('--sum_reward_rate', type=float, default=0.0)

    # only eval
    parser.add_argument('--output', type=int, default=1)
    parser.add_argument('--num_images', type=int, default=-1)
    parser.add_argument('--beam_size', type=int, default=1)

    args = parser.parse_args()

    return args

def image_features_info(text):
    if opt.bu == 1:
        text += '--input_fc_dir /mnt/poplin/share/dataset/MSCOCO/cocobu_fc ' \
                '--input_att_dir /mnt/poplin/share/dataset/MSCOCO/cocobu_att '
    else:
        text += '--input_fc_dir /mnt/poplin/share/2018/nakamura_M1/self_sequential/data/data/cocotalk_fc '\
                '--input_att_dir /mnt/poplin/share/2018/nakamura_M1/self_sequential/data/data/cocotalk_att'

    return text

def make_train_script(opt):
    text = 'python train.py ' \
           '--id ' + opt.id + ' ' \
           '--caption_model ' + opt.caption_model + ' ' \
           '--input_json /mnt/poplin/share/dataset/MSCOCO/cocotalk.json  ' \
           '--input_label_h5 data/cocotalk_label.h5 ' \
           '--batch_size 30 ' \
           '--learning_rate '+ str(opt.lr)  + ' ' \
           '--learning_rate_decay_start ' + str(opt.lr_ds) + ' ' \
           '--learning_rate_decay_every 6 ' \
           '--save_checkpoint_every 3000 ' \
           '--language_eval 1 ' \
           '--val_images_use ' + str(opt.val_images_use) + ' ' \
           '--max_epochs 300 ' \
           '--checkpoint_path log_' + opt.id + ' ' \
           '--start_from log_' + opt.id + ' ' \
           '--cider_reward_weight ' + str(opt.c_w) + ' ' \
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
           '--sum_reward_rate '  + str(opt.sum_reward_rate) + ' ' \
           '--use_weight_probability ' + str(opt.use_weight_probability) + ' '



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
                '--penalty_type ' + opt.penalty_type + ' ' \
                '--critic_weight ' + str(opt.cr_w) + ' '

        if opt.cycle is not None:
            text += '--cycle ' + str(opt.cycle) + ' '

        if opt.c_c_w is not None:
            text += '--critic_cider_reward_weight ' + str(opt.c_c_w) + ' '

        if opt.internal_model is not None:
            text += '--internal_model ' + str(opt.internal_model) + ' '

        if opt.sf_internal == 1:
            text += '--sf_internal_epoch ' + str(opt.epoch) + ' ' \
                    '--sf_internal_itr ' + str(opt.itr) + ' '

    if opt.bu == 1:
        text = image_features_info(text)
    if opt.itr == 75520:
        text += '--self_critical_after 20  '
    elif opt.itr == 113280:
        text += '--self_critical_after 30  '
    else:
        text += '--self_critical_after 20  '

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
    text = 'python eval.py ' \
           '--model log_' + opt.id + '/model_' + str(opt.epoch) + '_' + str(opt.itr) + '.pth ' \
           '--infos_path log_' + opt.id + '/infos_' + opt.id + '-best.pkl ' \
           '--input_json /mnt/poplin/share/dataset/MSCOCO/cocotalk.json  ' \
           '--input_label_h5 data/cocotalk_label.h5 ' \
           '--beam_size 1 ' \
           '--dump_images 0 ' \
           '--language_eval 1 ' \
           '--num_images ' + str(opt.num_images) + ' ' \
           '--output ' + str(opt.output) + ' ' \
           '--gpu ' + str(opt.gpu) + ' ' \
           '--prohibit_flg_hard ' + str(opt.pfh) + ' ' \
           '--beam_size ' + str(opt.beam_size) + ' ' \
           '--sim_pred_type 0 '


    if opt.bu == 1:
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
    text = 'python train.py ' \
           '--id ' + opt.id + ' ' \
            '--caption_model ' + opt.caption_model + ' ' \
            '--input_json /mnt/poplin/share/dataset/MSCOCO/cocotalk.json  ' \
            '--input_label_h5 data/cocotalk_label.h5 ' \
            '--batch_size 30 ' \
            '--learning_rate 5e-4 ' \
            '--learning_rate_decay_start 0 ' \
            '--scheduled_sampling_start 0 ' \
            '--save_checkpoint_every 6000 ' \
            '--language_eval 1 ' \
            '--val_images_use ' + str(opt.val_images_use) + ' ' \
            '--self_critical_after 30  ' \
            '--max_epochs 30 ' \
            '--checkpoint_path log_' + opt.id + ' ' \
            '--gpu ' + str(opt.gpu) + ' '

    if opt.bu == 1:
        text = image_features_info(text)

    return text

if __name__ == '__main__':
    opt = parse_opt()

    if opt.epoch is None:
        opt.epoch = int(opt.itr / 3776.2333)

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





