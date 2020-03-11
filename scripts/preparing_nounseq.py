import sys
import os
sys.path.append("/home/nakamura/project/python3_selfsequential/vg/")
import visual_genome.local as vg
dir = '/mnt/poplin/share/dataset/visualgenome'

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
import json
ps = PorterStemmer()


import argparse

def vg(opt):
    all_images = vg.get_all_image_data(dir)
    all_discriptions = vg.get_all_region_descriptions(dir)

    all_noun_sequenses_vg = []
    for i, discriptions in enumerate(all_discriptions):
        noun_sequenses_vg = {}
        noun_sequenses_vg['id'] = all_images[i].id
        noun_sequenses_vg['coco_id'] = all_images[i].coco_id
        sequenses = []
        for region in discriptions:
            splits = region.phrase.lower().split()
            e_words_stem_tag = make_stemwords(splits) # add tag
            nouns = pickup_classnoun(e_words_stem_tag, None) # pickup noun and make seq
            #         print('----------------------------------------')
            #         print(splits)
            #         print(e_words_stem_tag)
            #         print(nouns)
            sequenses.append(nouns)
        noun_sequenses_vg['sequences'] = sequenses
        #     all_noun_sequenses_vg.append(noun_sequenses_vg)
        all_noun_sequenses_vg.append(noun_sequenses_vg)
        if i % 1000 == 0:
            print('{}/{} ({:.2f}%) completed!'.format(i, len(all_images), i * 100 / len(all_images)))

    for i in range(len(all_noun_sequenses_vg)):
        all_noun_sequenses_vg[i]['S'] = []
        discriptions = all_discriptions[i]
        for j in range(len(discriptions)):
            S = discriptions[j].width * discriptions[j].height
            all_noun_sequenses_vg[i]['S'].append(S)
            all_noun_sequenses_vg[i]['rate'].append(S)

    save_dir = '/mnt/poplin/share/dataset/visualgenome/all_noun_sequenses_vg_mod.json'
    json.dump(all_noun_sequenses_vg, open(save_dir, 'w'))

def coco(opt):
    coco_dataset = json.load(open('/mnt/poplin/share/dataset/MSCOCO/dataset_coco.json', 'r'))
    all_noun_sequenses_coco = []
    for i in range(len(coco_dataset['images'])):
        noun_sequenses_coco = {}
        noun_sequenses_coco['coco_id'] = coco_dataset['images'][i].coco_id
        sequenses_ = []
        sentences = coco_dataset['images'][i]['sentences']
        for sent in sentences:
            splits = sent['tokens']
            e_words_stem_tag = make_stemwords(splits)
            nouns = pickup_classnoun(e_words_stem_tag, None)
            # print(splits)
            # print(nouns)
            sequenses_.append(nouns)
        noun_sequenses_coco['sequences'] = sequenses_
        all_noun_sequenses_coco.append(noun_sequenses_coco)

        if i % 1000 == 0:
            print('{}/{} ({:.2f}%) completed!'.format(i, len(coco_dataset['images']), i * 100 / len(coco_dataset['images'])))

    save_dir = '/mnt/poplin/share/dataset/visualgenome/all_noun_sequenses_vg_mod.json'
    json.dump(all_noun_sequenses_coco, open(save_dir, 'w'))

import copy
def make_stemwords(splits):
    e_words= splits
    e_words_stem = copy.copy(e_words)
    # e_words_tag = nltk.pos_tag(e_words)
    e_words_stem_tag = nltk.pos_tag(e_words_stem)

    return e_words_stem_tag

def pickup_classnoun(e_words_stem_tag, class_list):
    nouns = []
    flg = True
    for num, w in enumerate(e_words_stem_tag):
        tag = w[1]
        word = w[0]
        if (tag == 'NN' or tag == 'NNP' or tag == 'NNPS' or tag == 'NNS') and flg:
            # or tag == 'VB' or tag == 'VBD' or tag == 'VBG' or tag == 'VBN' or tag == 'VBP' or tag == 'VBZ'

            # 動仕入れるときは足す
            # if tag == 'VB' or tag == 'VBD' or tag == 'VBG' or tag == 'VBN' or tag == 'VBP' or tag == 'VBZ':
            #     word = ps.stem(word)

            if flg:
                nouns, flg = check_in_class(word, nouns, class_list)
        else:
            flg = True

    return nouns


def check_in_class(word, nouns, class_list):
    flg = True
    nouns.append(word)

    return nouns, flg

def search_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return synonyms

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='vg', type=str)
    opt = parser.parse_args()

    if opt.data == 'coco':
        coco(opt)
    else:
        vg(opt)