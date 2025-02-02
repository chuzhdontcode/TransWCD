# -*- coding: utf-8 -*#
# pixel-level  ——> image-level labels

import os
import numpy as np
import cv2
import natsort
import argparse
from pathlib import Path
import shutil 


parser = argparse.ArgumentParser()
parser.add_argument("--file_path", default="/data/zhenghui.zhao/Dataset/Change Detection/WHU-CD-256/label", type=str, help="file_path")
parser.add_argument("--save_path", default="/data/zhenghui.zhao/Dataset/Change Detection/WHU-CD-256", type=str, help="save_path")


if __name__ == "__main__":
    args = parser.parse_args()
    file_path = args.file_path
    save_path = args.save_path
    file_list = os.listdir(file_path)
    inList = natsort.natsorted(file_list)


    num = 0
    n = 0
    m = 0
    image_dict = {}
    #change = np.array([0., 0.], dtype=np.uint8)
    for name in inList:
        img = cv2.imread(file_path+'/'+ name)
        if img.any():
            change = np.array([1], dtype=np.float32)
            m = m+1
        else:
            change =  np.array([0], dtype=np.float32)
            n = n+1
        image_dict[name] = change

        num = num+1
    print('label_num:', num)
    print('nonchange_num:', n)
    print('change_num:', m)
    #print(image_dict)

    np.save(save_path+'/imagelevel_labels.npy', image_dict)
    #np.save('/data/zhenghui.zhao/Code/Affinity-from-attention/Affinity-from-attention-transformer/dual_stream/datasets/AICD_128/imagelevel_labels.npy', image_dict)
    #np.save('/data/zhenghui.zhao/Code/Affinity-from-attention/Affinity-from-attention-transformer_Single/dual_stream/datasets/AICD_128/imagelevel_labels.npy',image_dict)

    label = np.load(save_path+'/imagelevel_labels.npy',allow_pickle=True)
    #test = np.load('./dual_stream/datasets/voc/cls_labels_onehot.npy', allow_pickle=True)
    print(label)
    print('save_image_path:', save_path+'/imagelevel_labels.npy')

    # Create datasets/ with split.txt and .npy
    datasets_path = "../transwcd/datasets/" + save_path.split("/")[-1]
    Path(datasets_path).mkdir(parents=True, exist_ok=True)

    shutil.copytree(save_path+"/list", datasets_path, dirs_exist_ok=True)
    shutil.copy(save_path+"/imagelevel_labels.npy", datasets_path+"/imagelevel_labels.npy")
