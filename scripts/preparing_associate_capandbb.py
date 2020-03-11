import sys
sys.path.append("/home/nakamura/project/python3_selfsequential/vg/")
import visual_genome.local as vg
dir = '/mnt/poplin/share/dataset/visualgenome'
# all_images = vg.get_all_image_data(dir)
# all_discriptions = vg.get_all_region_descriptions(dir)
import numpy as np
import json
from nltk.corpus import wordnet
import argparse

match_table = json.load(open('/mnt/poplin/share/dataset/visualgenome/match_table.json'))


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

def main(opt):
    print('Associating...')
    if opt.vb_flg == 1:
        all_noun_vb_sequenses_coco = json.load(open('/mnt/poplin/share/dataset/visualgenome/all_noun_vb_sequenses_coco.json'))
        all_noun_vb_sequenses_vg = json.load(open('/mnt/poplin/share/dataset/visualgenome/all_noun_vb_sequenses_vg.json'))
    else:
        all_noun_vb_sequenses_coco = json.load(open('/mnt/poplin/share/dataset/visualgenome/all_noun_sequenses_coco.json'))
        all_noun_vb_sequenses_vg= json.load(open('/mnt/poplin/share/dataset/visualgenome/all_noun_sequenses_vg.json'))

    all_set_regions = json.load(open('/mnt/workspace2018/nakamura/selfsequential/data/marged_set_larger07.json'))

    all_set_regions_inverse = []
    if opt.nms_flg == 1:
        for i in range(len(all_set_regions)):
            return_info = {}
            set_info = all_set_regions[i]
            for key in set_info:
                return_info[int(key)] = int(key)
                if len(set_info[key]) > 0:
                    for id in set_info[key]:
                        return_info[id] = int(key)
            all_set_regions_inverse.append(return_info)
    else:
        return_info = {}
        for i in range(1000):
            return_info[i] = i
        for i in range(len(all_set_regions)):
            all_set_regions_inverse.append(return_info)

    associate_nounseq_to_bb_list = []
    for i in range(len(match_table)):
        if all_noun_vb_sequenses_vg[match_table[i][1]]['coco_id'] == all_noun_vb_sequenses_coco[match_table[i][0]]['coco_id']:
            seq_vgs = all_noun_vb_sequenses_vg[match_table[i][1]]['sequences']
            seq_areas = np.array(all_noun_vb_sequenses_vg[match_table[i][1]]['S_rate'])
            seq_areas_mask = seq_areas > opt.area_th
            seq_areas = seq_areas * seq_areas_mask
            seq_cocos = all_noun_vb_sequenses_coco[match_table[i][0]]['sequences']
            set_region_info = all_set_regions_inverse[i]
            associate_info = {}
            associate_info['coco_id'] = all_noun_vb_sequenses_vg[match_table[i][1]]['coco_id']
            associate_info['order'] = []
            associate_info['rate'] = []
            for j in range(len(seq_cocos)):
                seq_coco = seq_cocos[j]
                bb_sequence, not_found_num = select_boundingbox(seq_coco, seq_vgs, seq_areas, opt.sim_flg, set_region_info)
                associate_info['order'].append(bb_sequence)
                if len(bb_sequence) + not_found_num > 0:
                    associate_info['rate'].append(len(bb_sequence) / (len(bb_sequence) + not_found_num))
                else:
                    associate_info['rate'].append(0)
            associate_nounseq_to_bb_list.append(associate_info)
        else:
            pdb.set_trace()
        if i % 1000 == 0:
            print('{}/{} ({:.2f}%) completed!'.format(i, len(match_table), i*100/len(match_table)))
    print('Complete association!')

    print('Aline regions ...')
    # aline to regions
    for i in range(len(associate_nounseq_to_bb_list)):
        orders = associate_nounseq_to_bb_list[i]['order']
        #     orders = associate_nounvbseq_to_bb_list[i]['order']
        dets = list(all_set_regions[i].keys())
        dets = [int(i) for i in dets]
        alined_orders = []
        for order in orders:
            alined_order = []
            for j in range(len(order)):
                if order[j] in dets:
                    dets = np.array(dets)
                    index = np.where(dets == order[j])[0][0]
                else:
                    index = pickup_hidden_id(order[j], all_set_regions[i].keys(), all_set_regions[i])
                alined_order.append(index)
            alined_orders.append(alined_order)
        associate_nounseq_to_bb_list[i]['alined_order'] = alined_orders
    print('Allinment completed !')

    if opt.vb_flg == 1:
        name_vb = 'vb'
    else:
        name_vb = ''
    if opt.sim_flg == 1:
        name_sim = 'sim_'
    else:
        name_sim = ''
    if opt.nms_flg == 1:
        name_nms = 'nms_'
    else:
        name_nms = ''
    if opt.area_th > 0.0:
        name_area = str(opt.area_th) + '_'
    else:
        name_area = ''

    base_dir_name = '/mnt/poplin/share/dataset/visualgenome/associate_noun'
    save_dir = base_dir_name + name_vb + 'seq_to_bb_' + name_sim + name_nms + name_area + 'nooverlap_list_mod.json'
    json.dump(associate_nounseq_to_bb_list, open(save_dir, 'w'), cls=MyEncoder)
    print('save to ' + save_dir)


def pickup_hidden_id(region_id, keys, all_id):
    for i, key in enumerate(keys):
        for j in all_id[key]:
            if j == region_id:
                return i
    return -1

def search_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return synonyms

def select_boundingbox(seq_coco, seq_vgs, seq_areas, sim_flg, set_region_info):
    # seq_coco: 1 sequence
    # seq_vgs : all of vgs_sentences:

    bb_sequence = []
    flg = True
    not_found_num = 0
    for i in range(len(seq_coco)):
        if flg:
            if i + 1 < len(seq_coco):
                part_seq = seq_coco[i:i + 2]
                seq_vg_id = cal_iou(part_seq, seq_vgs, seq_areas, bb_sequence, set_region_info)  # 一致するものがあればidを，なければ-1を渡す．
            else:
                seq_vg_id = -1

            if seq_vg_id < 0:
                part_seq = seq_coco[i:i + 1]
                seq_vg_id = cal_iou(part_seq, seq_vgs, seq_areas, bb_sequence, set_region_info)
            else:
                flg = False

            if seq_vg_id < 0 and sim_flg > 0:
                w1 = wordnet.synsets(seq_coco[i])
                if len(w1) > 0:
                    part_seq[0] = w1[0]
                    seq_vg_id = cal_sim(part_seq, seq_vgs, seq_areas, bb_sequence, set_region_info)
                else:
                    seq_vg_id = -1

            if seq_vg_id > len(seq_vgs):
                pdb.set_trace()

            if seq_vg_id >= 0:
                bb_sequence.append(seq_vg_id)
            else:
                not_found_num += 1
        else:
            flg = True

    return bb_sequence, not_found_num

import pdb

def cal_sim(part_seq, seq_vgs, seq_areas, bb_sequence, set_region_info):
    seq_len = len(part_seq)
    scores = np.zeros(len(seq_vgs))
    score_max = 0
    for num, seq_vg in enumerate(seq_vgs):
        r_num = set_region_info[num]
        score = 0
        for i in range(len(seq_vg) - seq_len + 1):
            part_seq_vg = seq_vg[i:i + seq_len]
            w2 = wordnet.synsets(part_seq_vg[0])
            if len(w2) > 0:
                w2 = w2[0]
                score = part_seq[0].wup_similarity(w2)
            else:
                score = 0

        if score is None:
            score = 0


        if score >= score_max and score > 0:
            A = np.where(scores == score)
            if len(A[0]) == 0:
                scores[r_num] = score
            elif seq_areas[r_num] > np.max(seq_areas[A]):
                scores[r_num] = score

            score_max = score

    if len(bb_sequence) > 0:
        scores[np.array(bb_sequence)] = 0.0

    if np.sum(scores) == 0:
        return -1
    else:
        return np.argmax(scores)


def cal_iou(part_seq, seq_vgs, seq_areas, bb_sequence, set_region_info):
    seq_len = len(part_seq)
    scores = np.zeros(len(seq_vgs))

    for num, seq_vg in enumerate(seq_vgs):
        r_num = set_region_info[num]
        score = 0
        for i in range(len(seq_vg) - seq_len + 1):
            part_seq_vg = seq_vg[i:i + seq_len]
            if np.all(part_seq_vg == part_seq):
                score = seq_areas[r_num]
            else:
                score = 0

        scores[r_num] = score

    if len(bb_sequence) > 0:
        scores[np.array(bb_sequence)] = 0.0
    if np.sum(scores) == 0:
        return -1
    else:
        return np.argmax(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vb_flg', default=0, type=int)
    parser.add_argument('--sim_flg', default=0, type=int)
    parser.add_argument('--nms_flg', default=0, type=int)
    parser.add_argument('--area_th', default=0.0, type=float)
    params = parser.parse_args()

    main(params)