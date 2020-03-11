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
    parser.add_argument('--region_bleu_flg', type=int, default=1)
    parser.add_argument('--bleu_option', type=str, default='closest')
    parser.add_argument('--cider_option', type=str, default=None)

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
        text += '--input_fc_dir /mnt/workspace2019/visual_genome_pretrain/resnet_feature_fc ' \
                '--input_att_dir /mnt/workspace2019/visual_genome_pretrain/resnet_feature_att '

    return text

def make_initial_script(opt):

    text = 'python train.py ' \
           '--id ' + opt.id + ' ' \
            '--caption_model ' + opt.caption_model + ' ' \
            '--input_json /mnt/workspace2019/visual_genome_pretrain/cocotalk_pretrain_vg.json  ' \
            '--input_label_h5 /mnt/workspace2019/visual_genome_pretrain/cocotalk_pretrain_vg_label.h5 ' \
            '--batch_size 30 ' \
            '--learning_rate 5e-4 ' \
            '--learning_rate_decay_start 0 ' \
            '--scheduled_sampling_start 0 ' \
            '--save_checkpoint_every 3000 ' \
            '--language_eval 1 ' \
            '--val_images_use ' + str(opt.val_images_use) + ' ' \
            '--self_critical_after 20  ' \
            '--max_epochs 20 ' \
            '--checkpoint_path log_' + opt.id + ' ' \
            '--gpu ' + str(opt.gpu) + ' '

    if opt.bu == 1:
        text = image_features_info(text)

    return text

if __name__ == '__main__':
    opt = parse_opt()

    if opt.epoch is None:
        opt.epoch = int(opt.itr / 1564.45)

    if opt.epoch % 2 == 0:
        opt.internal_rl_flg = 1

    text = make_initial_script(opt)

    with open('run.sh', 'w') as f:
        f.write(text)

    if not os.path.isdir('/mnt/workspace2018/nakamura/selfsequential/log_python3/log_' + opt.id + '/'):
        os.mkdir('/mnt/workspace2018/nakamura/selfsequential/log_python3/log_' + opt.id + '/')
    with open('/mnt/workspace2018/nakamura/selfsequential/log_python3/log_' + opt.id + '/' + opt.type + '_run.sh', 'w') as f:
        f.write(text)
    with open('/mnt/workspace2018/nakamura/selfsequential/log_python3/log_' + opt.id + '/' + opt.type + '_run.txt', 'w') as f:
        f.write(text)





