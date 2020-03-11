import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image as PIL_Image
import requests
import pdb
import numpy as np
import json
import h5py
parser = argparse.ArgumentParser()

info = json.load(open('/mnt/poplin/share/dataset/MSCOCO/cocotalk_subset_vg_larger_addregions.json'))
h5_label_file = h5py.File('/mnt/poplin/share/dataset/MSCOCO/cocotalk_subset_vg_larger_label_all.h5', 'r', driver='core')

# '--input_json /mnt/poplin/share/dataset/MSCOCO/cocotalk_subset_vg_larger_addregions.json '
# '--input_label_h5 /mnt/poplin/share/dataset/MSCOCO/cocotalk_subset_vg_larger_label_all.h5 '

label_start_ix = h5_label_file['label_start_ix'][:]
label_end_ix = h5_label_file['label_end_ix'][:]
label_start_ix_er = h5_label_file['label_start_ix_er'][:]
label_end_ix_er = h5_label_file['label_end_ix_er'][:]
labels = h5_label_file['labels']

label_start_ix_new = np.array(label_start_ix)
label_end_ix_new = np.array(label_end_ix)
label_start_ix_er_new = np.array(label_start_ix_er)
label_end_ix_er_new = np.array(label_end_ix_er)
labels_new = np.array(labels)

region_info = json.load(open('/mnt/poplin/share/dataset/MSCOCO/selected_region_info_88_08.json'))
region_file = region_info['acrions']
images = info['images']

# load new label_data
new_label_1 = json.load(open('/mnt/workspace2019/nakamura/selfsequential/vg_similar_gt/all_info_dict_18000_0.json'))
new_label_2 = json.load(open('/mnt/workspace2019/nakamura/selfsequential/vg_similar_gt/all_info_dict_33000_17000.json'))
new_label_3 = json.load(open('/mnt/workspace2019/nakamura/selfsequential/vg_similar_gt/all_info_dict_34000_33000.json'))
new_label_4 = json.load(open('/mnt/workspace2019/nakamura/selfsequential/vg_similar_gt/all_info_dict_48000_34000.json'))
new_label_5 = json.load(open('/mnt/workspace2019/nakamura/selfsequential/vg_similar_gt/all_info_dict_51196_48000.json'))

all_label = {}
all_label.update(new_label_1)
all_label.update(new_label_2)
all_label.update(new_label_3)
all_label.update(new_label_4)
all_label.update(new_label_5)

def update_labels(old_seq, new_seq, start_id):
    upper_seq = old_seq[:start_id - 1]
    later_seq = old_seq[start_id - 1:]
    middle_seq = np.zeros([3, 30], dtype='int')
    middle_seq[:, :20] = np.array(new_seq)

    if len(upper_seq) > 0 and len(later_seq) > 0:
        output_seq = np.concatenate((upper_seq, middle_seq, later_seq), axis=0)
    elif len(upper_seq) == 0:
        output_seq = np.concatenate((middle_seq, later_seq), axis=0)
    else:
        output_seq = np.concatenate((upper_seq, middle_seq), axis=0)

    return output_seq


def update_id(label_start_ix_new, label_end_ix_new, label_start_ix_er_new, label_end_ix_er_new, image_id, region_id):
    # region image
    label_end_ix_er_new[image_id, region_id:] += 3
    if region_id + 1 < len(label_start_ix_er_new[image_id]):
        label_start_ix_er_new[image_id, region_id + 1:] += 3

    # other image
    label_end_ix_new[image_id:] += 3
    if image_id + 1 < len(label_start_ix_new):
        label_start_ix_new[image_id + 1:] += 3
        label_end_ix_er_new[image_id + 1:] += 3
        label_start_ix_er_new[image_id + 1:] += 3

    return label_start_ix_new, label_end_ix_new, label_start_ix_er_new, label_end_ix_er_new

for i in range(len(images)):
    new_labels = all_label[str(images[i]['id'])]
    regions = region_file[i]
    if len(new_labels) > 0:
        for j in range(len(regions)):
            if regions[j] < 0:
                break
            new_seq = new_labels[j]['seq']
            start_id = label_start_ix_er_new[i, j]
            labels_new = update_labels(labels_new, new_seq, start_id)
            label_start_ix_new, label_end_ix_new, label_start_ix_er_new, label_end_ix_er_new =\
                update_id(label_start_ix_new, label_end_ix_new, label_start_ix_er_new, label_end_ix_er_new, i, j)
    if i % 100 == 0:
        print('{}/{}'.format(i,len(images)))

f_lb = h5py.File('/mnt/poplin/share/dataset/MSCOCO/cocotalk_subset_vg_larger_label_all_newlabels.h5', "w")
f_lb.create_dataset("labels", dtype='uint32', data=labels_new)
f_lb.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix_new)
f_lb.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix_new)
# f_lb.create_dataset("label_length", dtype='uint32', data=label_length_vg)
f_lb.create_dataset("label_start_ix_er", dtype='uint32', data=label_start_ix_er_new)
f_lb.create_dataset("label_end_ix_er", dtype='uint32', data=label_end_ix_er_new)
f_lb.close()