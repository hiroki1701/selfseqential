import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    dir = '/mnt/poplin/share/2018/nakamura_M1/self_sequential/data/'
    # parser.add_argument('--input_json', type=str, default= dir + 'data/cocotalk.json',
    #                 help='path to the json file containing additional info and vocab')
    parser.add_argument('--input_json', type=str, default= '/mnt/poplin/share/dataset/MSCOCO/cocotalk.json',
                    help='path to the json file containing additional info and vocab')
    parser.add_argument('--input_fc_dir', type=str, default=dir + 'data/cocotalk_fc',
                    help='path to the directory containing the preprocessed fc feats')
    parser.add_argument('--input_att_dir', type=str, default=dir + 'data/cocotalk_att',
                    help='path to the directory containing the preprocessed att feats')
    parser.add_argument('--input_subatt_dir', type=str, default=dir + 'data/cocotalk_att',
                        help='path to the directory containing the preprocessed att feats')
    parser.add_argument('--input_box_dir', type=str, default=dir + 'data/cocotalk_box',
                    help='path to the directory containing the boxes of att feats')
    parser.add_argument('--input_label_h5', type=str, default=dir + 'data/coco_label.h5',
                    help='path to the h5file containing the preprocessed dataset')
    # parser.add_argument('--input_bu_feature', type=str, default='/mnt/poplin/share/dataset/MSCOCO/cocobu_box/',
    #                     help='path to the h5file containing the preprocessed dataset')
    # /mnt/workspace2018/nakamura/bottom-up-attention/cocobu_weight
    parser.add_argument('--input_bu_feature', type=str, default='/mnt/poplin/share/2018/nakamura_M1/self_sequential/data/data/bbox_info.json')
    parser.add_argument('--start_from', type=str, default=None,
                    help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                        'infos.pkl'         : configuration;
                        'checkpoint'        : paths to model file(s) (created by tf).
                                              Note: this file contains absolute paths, be careful when moving files around;
                        'model.ckpt-*'      : file(s) with model definition (created by tf)
                    """)
    parser.add_argument('--sf_epoch', type=int, default=None)
    parser.add_argument('--sf_itr', type=int, default=None)
    parser.add_argument('--sf_internal_epoch', type=int, default=None)
    parser.add_argument('--sf_internal_itr', type=int, default=None)
    parser.add_argument('--cached_tokens', type=str, default='coco-train-idxs',
                    help='Cached token file for calculating cider score during self critical training.')
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--selected_region_file', type=str, default=None)

    # Model settings
    parser.add_argument('--caption_model', type=str, default="show_tell",
                    help='show_tell, show_attend_tell, all_img, fc, att2in, att2in2, att2all2, adaatt, adaattmo, topdown, stackatt, denseatt, hie_att, hcatt_hard')
    parser.add_argument('--manager_model', type=str, default=None,
                    help = 'manager,manager_lstm')
    parser.add_argument('--rnn_size', type=int, default=512,
                    help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers in the RNN')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                    help='rnn, gru, or lstm')
    parser.add_argument('--input_encoding_size', type=int, default=512,
                    help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--att_hid_size', type=int, default=512,
                    help='the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using hidden layer')
    parser.add_argument('--fc_feat_size', type=int, default=2048,
                    help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--att_feat_size', type=int, default=2048,
                    help='2048 for resnet, 512 for vgg')
    parser.add_argument('--logit_layers', type=int, default=1,
                    help='number of layers in the RNN')
    parser.add_argument('--baseline_regressor', type=int, default=0,
                        help='reward_baseline regressor')
    parser.add_argument('--internal_model', type=str, default='',
                        help='P or R or D or sim or sim_newr')
    parser.add_argument('--actor_critic_flg', type=int, default=0)
    parser.add_argument('--use_similarity', type=int, default=0,
                        help='using similarity as a reward')
    parser.add_argument('--att_num', type=int, default=1,
                        help='att_num 1~')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='threshold for internal critic r')
    parser.add_argument('--internal_alpha', type=float, default=0.7)
    parser.add_argument('--internal_rl_flg', type=int, default=0)
    parser.add_argument('--att_reward_flg', type=int, default=0, help= 'when calclate reward, use attention normalization')
    parser.add_argument('--sim_reward_flg', type=int, default=0, help = 'use similarity as a reward')
    parser.add_argument('--sim_newr_reward_flg', type=int, default=0, help='use similarity as a reward and consider process of similarity')
    parser.add_argument('--sim_reward_flg_minus', type=int, default=0, help='use similarity as a reward')
    parser.add_argument('--sim_att_norm_reward_flg', type=int, default=0, help='use similarity and att as a reward')
    parser.add_argument('--sim_pred_type', type=int, default=0, help='select similarity predictor')
    parser.add_argument('--sim_newr_only_prob_reward_flg', type=int, default=0, help='use similarity as a reward and consider process of similarity and only use prob reward')
    parser.add_argument('--prohibit_flg', type=int, default=0, help='use prohibition of return saliency')
    parser.add_argument('--prohibit_flg_hard', type=int, default=0, help='use prohibition of return saliency for hard attention')
    parser.add_argument('--sub_seq_flg', type=int, default=0,
                        help='use prohibition of return saliency for hard attention')
    parser.add_argument('--ppo_flg', type=int, default=0,
                        help='use PPO like update')
    parser.add_argument('--ppo', type=int, default=0,
                        help='use PPO like update')
    parser.add_argument('--min_seq_length', type=int, default=-1,
                        help='minimun number of sequrnse length')
    parser.add_argument('--whole_att_flg', type=int, default=0, help='To evaluate baseline method, whole image feature is used as a attention')

    parser.add_argument('--use_bn', type=int, default=0,
                    help='If 1, then do batch_normalization first in att_embed, if 2 then do bn both in the beginning and the end of att_embed')

    # feature manipulation
    parser.add_argument('--norm_att_feat', type=int, default=0,
                    help='If normalize attention features')
    parser.add_argument('--use_box', type=int, default=0,
                    help='If use box features')
    parser.add_argument('--norm_box_feat', type=int, default=0,
                    help='If use box, do we normalize box feature')

    # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=-1,
                    help='number of epochs')
    parser.add_argument('--seq_length', type=int, default=20,
                        help='number of sequense')
    parser.add_argument('--batch_size', type=int, default=16,
                    help='minibatch size')
    parser.add_argument('--grad_clip', type=float, default=0.1, #5.,
                    help='clip gradients at this value')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--self_critical_after', type=int, default=-1,
                    help='After what epoch do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
    parser.add_argument('--seq_per_img', type=int, default=5,
                    help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
    parser.add_argument('--beam_size', type=int, default=1,
                    help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')

    #Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam',
                    help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                    help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=-1, 
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=3, 
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8, 
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--optim_alpha', type=float, default=0.9,
                    help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                    help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                    help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight_decay')

    # Optimization: for the internal critic
    parser.add_argument('--c_optim', type=str, default='adam',
                        help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--c_learning_rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--c_learning_rate_decay_start', type=int, default=-1,
                        help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--c_learning_rate_decay_every', type=int, default=6,
                        help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--c_learning_rate_decay_rate', type=float, default=0.8,
                        help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--c_optim_alpha', type=float, default=0.9,
                        help='alpha for adam')
    parser.add_argument('--c_optim_beta', type=float, default=0.999,
                        help='beta used for adam')
    parser.add_argument('--c_optim_epsilon', type=float, default=1e-8,
                        help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--c_weight_decay', type=float, default=1e-8,
                        help='weight_decay')
    parser.add_argument('--input_h_flg', type=int, default=1,
                        help='input hidden state')
    parser.add_argument('--multi_learn_flg', type=int, default=0,
                        help='lenarning both captioning and critic')
    parser.add_argument('--only_critic_train', type=int, default=0)
    parser.add_argument('--critic_probabilistic', type=int, default=0)
    parser.add_argument('--bag_flg', type=int, default=0)
    parser.add_argument('--region_bleu_flg', type=int, default=0)
    parser.add_argument('--softplus_flg', type=int, default=0)

    parser.add_argument('--scheduled_sampling_start', type=int, default=-1, 
                    help='at what iteration to start decay gt probability')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5, 
                    help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05, 
                    help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25, 
                    help='Maximum scheduled sampling prob.')


    # Evaluation/Checkpointing
    parser.add_argument('--val_images_use', type=int, default=3200,
                    help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--save_checkpoint_every', type=int, default=2500,
                    help='how often to save a model checkpoint (in iterations)?')
    parser.add_argument('--checkpoint_path', type=str, default='save',
                    help='directory to store checkpointed models')
    parser.add_argument('--language_eval', type=int, default=0,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add_argument('--losses_log_every', type=int, default=25,
                    help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')       
    parser.add_argument('--load_best_score', type=int, default=1,
                    help='Do we load previous best score when resuming training.')       

    # misc
    parser.add_argument('--id', type=str, default='',
                    help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--train_only', type=int, default=0,
                    help='if true then use 80k, else use 110k')
    parser.add_argument('--cycle', type=int, default=None)


    # Reward
    parser.add_argument('--cider_reward_weight', type=float, default=0.0,
                    help='The reward weight from cider')
    parser.add_argument('--wbleu_reward_weight', type=float, default=0.0,
                        help='The reward weight from weighted bleu')
    parser.add_argument('--discriminator_weight', type=float, default=0.0,
                    help='The reward weight from Disctriminator')
    parser.add_argument('--critic_cider_reward_weight', type=float, default=None,
                        help='The reward weight from cider for critic')
    parser.add_argument('--bleu_reward_weight', type=float, default=0,
                    help='The reward weight from bleu4')
    parser.add_argument('--gsim_weight', type=float, default=0.0,
                        help='The reward weight from similarity')
    parser.add_argument('--recall_reward_weight', type=float, default=0,
                        help='The reward weight from Recall')
    parser.add_argument('--att_reward_weight', type=float, default=0.0,
                        help='The reward weight from att score')
    parser.add_argument('--critic_weight', type=float, default=0.0,
                        help='The reward weight from critic')
    parser.add_argument('--used_area_weight', type=float, default=0.0,
                        help='the reward weight of used_area_weight')
    parser.add_argument('--xe_weight', type=float, default=0.0,
                        help='the loss weight of used_xe_weight')
    parser.add_argument('--rloss_weight', type=float, default=1.0,
                        help='the loss weight of used_xe_weight')
    parser.add_argument('--l_score', type=float, default=None,
                        help='The lowest score to learn similarity')
    parser.add_argument('--l_sim', type=float, default=None,
                        help='The lowest score to learn cider')
    parser.add_argument('--log_flg', type=int, default=0,
                        help='if it is 1, cider score will be log')
    parser.add_argument('--sig_flg', type=int, default=0,
                        help='if it is 1, cider score will be sigmoid')
    parser.add_argument('--separate_reward', type=int, default=0)
    parser.add_argument('--att_lambda', type=float, default=0.0)
    parser.add_argument('--penalty_type', type=str, default='nashi', help='select from cases or compare')
    parser.add_argument('--sim_sum_flg', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu_num')
    parser.add_argument('--gpu2', type=int, default=None,
                        help='gpu_num')
    parser.add_argument('--r_baseline', type=int, default=0)
    parser.add_argument('--critic_encode', type=int, default=0)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--sum_reward_rate', type=float, default=0.0)
    parser.add_argument('--bleu_option', type=str, default='closest', help='closest or times')
    parser.add_argument('--cider_option', type=str, default=None)
    parser.add_argument('--dis_adv_flg', type=int, default=0, help='do adversarial training to discriminator and captioning model')
    parser.add_argument('--dis_type', type=str, default='coco', help='selsect discriminator model type')
    parser.add_argument('--no_local_reward', type=int, default=0, help='if you use internal critic with no phrase reward')
    parser.add_argument('--weight_deterministic_flg', type=int, default=0, help='if you want to use initial learning')
    parser.add_argument('--use_weight_probability', type=int, default=0)
    parser.add_argument('--max_att_len', type=int, default=36)
    parser.add_argument('--p_switch', type=int, default=0)
    parser.add_argument('--area_feature_use', type=int, default=0)

    parser.add_argument('--cut_length', type=int, default=-1)
    parser.add_argument('--random_disc', type=int, default=0, help='use switch time sequence randomly')
    parser.add_argument('--all_switch_end_dis', type=int, default=0, help='use all switch time sequences and end time')
    parser.add_argument('--all_switch_dis', type=int, default=0)
    parser.add_argument('--use_next_region', type=int, default=0, help='use next region feature to predict next word')
    parser.add_argument('--t_model_flg', type=int, default=0, help='use target model to calclate reward')

    args = parser.parse_args()
    # args = parser.parse_args(args=[])

    # Check if args are valid
    assert args.rnn_size > 0, "rnn_size should be greater than 0"
    assert args.num_layers > 0, "num_layers should be greater than 0"
    assert args.input_encoding_size > 0, "input_encoding_size should be greater than 0"
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert args.drop_prob_lm >= 0 and args.drop_prob_lm < 1, "drop_prob_lm should be between 0 and 1"
    assert args.seq_per_img > 0, "seq_per_img should be greater than 0"
    assert args.beam_size > 0, "beam_size should be greater than 0"
    assert args.save_checkpoint_every > 0, "save_checkpoint_every should be greater than 0"
    assert args.losses_log_every > 0, "losses_log_every should be greater than 0"
    # assert args.language_eval == 0 or args.language_eval == 1, "language_eval should be 0 or 1"
    assert args.load_best_score == 0 or args.load_best_score == 1, "language_eval should be 0 or 1"
    assert args.train_only == 0 or args.train_only == 1, "language_eval should be 0 or 1"

    return args