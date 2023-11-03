'''
[ 필요한 값들 ]
- 모든 영상에서의 logits 값(LOGIT_PATH)
- image명들(IMAGE_NAMES_PATH)
- 해당 image의 gt들(GT_PATH)
'''

import os
import math
import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from copy import deepcopy
import itertools

##################################################################

# global variables
ROOT = '/activate_logits과 image_names를 저장할 경로'

# logit 과 images 명 설정 파일 경로
LOGIT_PATH = os.path.join(ROOT, "activate_logits.npy")
IMAGE_NAMES_PATH = os.path.join(ROOT, "image_names.txt")

# save path
SAVE_THREE_CHANNELS = '/채널별로 합친 activation map을 저장할 경로'

UPPER_PALETTE = [0,0,0,   
                0,0,0,    
                0,128,0]
BOTTOM_PALETTE = [0,0,0,   
                128,0,0,    
                0,0,0]

##################################################################

'''
function1. logits 전처리 함수
- logits: need to be preprocessed
- remove padding -> integrate channels based on max value -> resize
'''
def preprocess_logits(logits):
    part_am = logits[:,:,70:-70]
    # part_am = logits[:,:,50:-50]
    part_am = np.max(part_am, axis=-1)
    bs, h, w = part_am.shape
    
    resized_images = np.empty((bs, 128, 64))
    for idx in range(bs):
        resized_images[idx] = cv2.resize(part_am[idx], (64, 128))

    return resized_images

'''
function2. point값을 하나하나 매칭시키기 위한 function
'''
def make_dicmap(dataset):
    dic2clt = {'upper':2, 'bottom':1, 'dress':3, 'bkg': 0}

    dic_parset = {
        'ppss': {'upper':[3], 'bottom':[4]},  
        'LIP': {'upper':[5,7], 'bottom':[9,12], 'dress':[6,10], 'bkg':[0]},
        'MHP_v2': {'upper':[10,11,12,13,14,15,34,57],  
                'bottom':[17,18,19,58],  
                'dress':[35,36,37,38]},  
        'Duke': {'upper':[3], 'bottom':[5], 'dress': [200]},
        'HPD': {'upper':[4,5], 'bottom':[6], 'dress':[7]}
    }

    dic_map = {v: dic2clt[key] for key, value in dic_parset[dataset].items() for v in value}
    return dic_map, dic2clt, dic_parset[dataset]

def preprocess():
    PRE_NORM_UPPER_ONE_LOGIT = preprocess_logits(MINMAX_UPPER_ONE_LOGIT)
    PRE_NORM_BOTTOM_ONE_LOGIT = preprocess_logits(MINMAX_BOTTOM_ONE_LOGIT)
    PRE_NORM_BKG_ONE_LOGIT = preprocess_logits(MINMAX_BKG_ONE_LOGIT)

    return PRE_NORM_UPPER_ONE_LOGIT, PRE_NORM_BOTTOM_ONE_LOGIT, PRE_NORM_BKG_ONE_LOGIT

def convert_to_l(am):
    heatmapshow = Image.fromarray(np.asarray(am, dtype=np.uint8))
    heatmapshow = heatmapshow.convert('L') #굳이 안 해도 됨
    return heatmapshow

##################################################################

#Step1. Set images & logits
#Step1-1. load image names
image_names = []
with open(IMAGE_NAMES_PATH, "r") as file:
    for i in file:
        image_names.append(i.strip())
print(image_names)

#Step1-2. load logits
one_logit = np.load(LOGIT_PATH)
one_logit_ = one_logit[1:, ]

#Step2. 상/하의 부분에 관해 추출한 mean, std 값을 통해 normalize 진행
UPPER_MEAN, UPPER_STD, UPPER_MAX, UPPER_MIN = np.mean(one_logit_[:,:,:,[5,7]]), np.std(one_logit_[:,:,:,[5,7]]), \
                                                np.max(one_logit_[:,:,:,[5,7]]), np.min(one_logit_[:,:,:,[5,7]])
BOTTOM_MEAN, BOTTOM_STD, BOTTOM_MAX, BOTTOM_MIN = np.mean(one_logit_[:,:,:,[9, 12]]), np.std(one_logit_[:,:,:,[9, 12]]), \
                                                np.max(one_logit_[:,:,:,[9, 12]]), np.min(one_logit_[:,:,:,[9, 12]])
BKG_MEAN, BKG_STD, BKG_MAX, BKG_MIN = np.mean(one_logit_[:,:,:,[0]]), np.std(one_logit_[:,:,:,[0]]), \
                                                np.max(one_logit_[:,:,:,[0]]), np.min(one_logit_[:,:,:,[0]])

#minmax 진행
MINMAX_UPPER_ONE_LOGIT = (deepcopy(one_logit_[:,:,:,[5,7]]) - UPPER_MIN) / (UPPER_MAX - UPPER_MIN)*255         #N*H*W*CHANNEL
MINMAX_UPPER_MEAN, MINMAX_UPPER_STD, MINMAX_UPPER_MAX, MINMAX_UPPER_MIN = np.mean(MINMAX_UPPER_ONE_LOGIT), np.std(MINMAX_UPPER_ONE_LOGIT), \
                                                                    np.max(MINMAX_UPPER_ONE_LOGIT), np.min(MINMAX_UPPER_ONE_LOGIT)
MINMAX_BOTTOM_ONE_LOGIT = (deepcopy(one_logit_[:,:,:,[9, 12]]) - BOTTOM_MIN) / (BOTTOM_MAX - BOTTOM_MIN)*255    #N*H*W*CHANNEL
MINMAX_BOTTOM_MEAN, MINMAX_BOTTOM_STD, MINMAX_BOTTOM_MAX, MINMAX_BOTTOM_MIN = np.mean(MINMAX_BOTTOM_ONE_LOGIT), np.std(MINMAX_BOTTOM_ONE_LOGIT), \
                                                                    np.max(MINMAX_BOTTOM_ONE_LOGIT), np.min(MINMAX_BOTTOM_ONE_LOGIT)
MINMAX_BKG_ONE_LOGIT = (deepcopy(one_logit_[:,:,:,[0]]) - BKG_MIN) / (BKG_MAX - BKG_MIN)*255    #N*H*W*CHANNEL
MINMAX_BKG_MEAN, MINMAX_BKG_STD, MINMAX_BKG_MAX, MINMAX_BKG_MIN = np.mean(MINMAX_BKG_ONE_LOGIT), np.std(MINMAX_BKG_ONE_LOGIT), \
                                                                    np.max(MINMAX_BKG_ONE_LOGIT), np.min(MINMAX_BKG_ONE_LOGIT)

PRE_NORM_UPPER_ONE_LOGIT, PRE_NORM_BOTTOM_ONE_LOGIT, PRE_NORM_BKG_ONE_LOGIT = preprocess()
_, dic2clt, LIP = make_dicmap('LIP')

for idx, (image_name, upper_norm_log, bottom_norm_log, bkg_norm_log) in enumerate(zip(image_names, PRE_NORM_UPPER_ONE_LOGIT, PRE_NORM_BOTTOM_ONE_LOGIT, PRE_NORM_BKG_ONE_LOGIT)):
    print(f'{idx+1}번 !!{image_name} 입니다!!')
    
    '''
    dic2clt = {'upper':2, 'bottom':1, 'dress':3, 'bkg': 0}
    LIP = {'upper':[5,7], 'bottom':[9,12], 'dress':[6,10], 'bkg':[0]}
    '''
    output_im1 = convert_to_l(upper_norm_log)
    output_im2 = convert_to_l(bottom_norm_log)
    output_im3 = convert_to_l(bkg_norm_log)

    stacked_image = np.stack((output_im1, output_im2, output_im3), axis=-1)
    plt.imshow(stacked_image)

    stacked_image = Image.fromarray(np.asarray(stacked_image, dtype=np.uint8))
    stacked_image.save(os.path.join(SAVE_THREE_CHANNELS, image_name+'.png'))