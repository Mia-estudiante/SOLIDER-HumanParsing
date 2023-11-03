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

# images/gt/model결과 디렉토리 경로
RAW_IMAGES_PATH = "/Market1501/val_images"
GT_PATH = '/Market1501/val_segmentations'
MODEL_PATH = '/lip_solider_swin_base_market1501' #parsing result 저장 경로

# 저장경로
# nonorm
# SAVE_ROOT_NONORM = '/lip_solider_swin_base_market1501_activation_map/nonorm'
# SAVE_ROOT_NONORM_UPPER, SAVE_ROOT_NONORM_BOTTOM = os.path.join(SAVE_ROOT_NONORM, 'upper'), os.path.join(SAVE_ROOT_NONORM, 'bottom')

# norm
SAVE_ROOT_NORM = '/lip_solider_swin_base_market1501_activation_map/norm'
SAVE_ROOT_NORM_UPPER, SAVE_ROOT_NORM_BOTTOM = os.path.join(SAVE_ROOT_NORM, 'upper'), os.path.join(SAVE_ROOT_NORM, 'bottom')
# save pseudo activation maps adjusted by threshold values
SAVE_ROOT_NORM_PSEUDO_UPPER, SAVE_ROOT_NORM_PSEUDO_BOTTOM = os.path.join(SAVE_ROOT_NORM, 'pseudo_upper'), os.path.join(SAVE_ROOT_NORM, 'pseudo_bottom')

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
    part_am = logits[:,:,50:-50]
    part_am = np.max(part_am, axis=-1)
    bs, h, w = part_am.shape
    
    resized_images = np.empty((bs, 128, 64))
    for idx in range(bs):
        resized_images[idx] = cv2.resize(part_am[idx], (64, 128))

    return resized_images

'''
function2. 분포도 확인
- logits:정규화 및 전처리 받음 
'''
def all_dist(logits):
    #대략 1단위로 bin 값을 정하기 위해 binwidth를 max-min 값으로 구한다.
    min_value = np.min(logits.reshape(-1))
    max_value = np.max(logits.reshape(-1))
    binwidth = math.ceil(max_value) - math.floor(min_value)

    # hist, bin_edges, _ = plt.hist(data, bins=5, alpha=0.6, color='b', edgecolor='k')
    hist, bin_edges, _ = plt.hist(logits.reshape(-1), bins=binwidth, density=True, alpha=0.4, histtype='step', color='b', edgecolor='k')
    plt.xlim([min_value, max_value]) 
    plt.show()
    
    ###################################################

    # Now you can access the bin edges and bin counts
    for i in range(len(bin_edges) - 1):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        bin_count = hist[i]
        print(f"Bin {i + 1}: Range = ({bin_start}, {bin_end}), Count = {bin_count}")

    ###################################################

    return min_value, max_value, binwidth
    '''
    UPPER
    * (min_value, max_value) = (-2.2293992042541504, 5.574288368225098)
    * binwidth = 9

    BOTTOM
    * (min_value, max_value) = 
    * binwidth =
    '''

'''
function3. Load a gt image
'''
def load_gt_image(root, image_name):
    gt_image_path = os.path.join(root, image_name+'.png')
    gt_image = Image.open(gt_image_path)
    gt_image = np.asarray(gt_image, dtype=np.uint8)

    return gt_image

'''
function4. Print histogram
'''
def show_hist(part_logits, max_value, min_value, binwidth):
    #Step1. 0인 값들은 gt 영역에 해당하지 않는 값들이므로 이들은 담지않음
    blank_list = []
    for logit in part_logits:
        blank_list.append([ i for i in list(itertools.chain(*logit)) if i!=0 ])

    gt_part_logits = list(itertools.chain(*blank_list))
    # print(gt_part_logits)

    #Step2. 동일 x범위 내에서 분포도를 구함
    hist, bin_edges, _ = plt.hist(gt_part_logits, bins=binwidth, density=True, alpha=0.4, histtype='step', range=(min_value, max_value))
    plt.xlim([min_value, max_value]) 
    plt.show()

    return [np.min(gt_part_logits), np.max(gt_part_logits)]

'''
function5. 분포도 확인함으로써 threshold range 를 대략 정하기 위함
- 정규화한 upper&bottom log들은 gt와 크기가 동일해야 함
'''
def extract_thresholds(upper_logits, bottom_logits, upper_max, upper_min, upper_binwidth, bottom_max, bottom_min, bottom_binwidth, image_names): 
    gt_upper_logits = []
    gt_bottom_logits = []

    for _, (image_name, upper_norm_log, bottom_norm_log) in enumerate(zip(image_names, upper_logits, bottom_logits)):
        #load gt image
        gt_image = load_gt_image(GT_PATH, image_name) #128*64

        #Step1. 상/하의로 구분
        '''
        값이 아닌 위치를 매칭시켜 해당 위치에 있는 logit 값을 구한다
        '''
        upper_gt_image = np.where(gt_image==2, upper_norm_log, 0) #{upper에 해당하는 logits}
        bottom_gt_image = np.where(gt_image==1, bottom_norm_log, 0) #{bottom에 해당하는 logits}

        #Step2. gt(mask) image를 기준으로 해당 바운더리에 속하는 logit값들을 모음
        '''
        upper_gt_image 
        bottom_gt_image 에 해당하는 위치에 해당하는 logit 값들을 구함
        - 해당 바운더리에 속하지 않는 값들은 전부 0으로 맞춰줌
        '''
        gt_upper_logits.append(upper_gt_image) 
        gt_bottom_logits.append(bottom_gt_image)

    upper_range = show_hist(gt_upper_logits, upper_max, upper_min, upper_binwidth)
    bottom_range = show_hist(gt_bottom_logits, bottom_max, bottom_min, bottom_binwidth)

    return upper_range, bottom_range

'''
function6.
이 모든 작업은 상/하의 영역에 대해 더 좁은 영역에서의 올바른 pseudo image를 추출하기 위함
상/하의 영역에 해당하는 threshold를 정할 수 있는 range를 구하기 위함
    1. {상/하의에 해당하는 activation map} 에서의 normalize를 진행
    2. 전체 영상에 대해서 {상/하의에 해당하는 activation map 의 분포도} 를 구함
    3. 전체 영상에서 {"상/하의" 영역에 대한 logit 분포도} 를 구함
        - 이때, 2번과 비교하기 위해, 2번에서 구한 것들과 함께 보기 위해,
            x축의 범위를 동일하게 함
'''
def preprocess():
    #Step1. Resize normalized logits using std and mean value
    PRE_NORM_UPPER_ONE_LOGIT = preprocess_logits(NORM_UPPER_ONE_LOGIT)
    PRE_NORM_BOTTOM_ONE_LOGIT = preprocess_logits(NORM_BOTTOM_ONE_LOGIT)

    #Step2. "전체 영상"에 대한 상/하의 logit 분포도
    upper_min_value, upper_max_value, upper_binwidth = all_dist(PRE_NORM_UPPER_ONE_LOGIT)
    bottom_min_value, bottom_max_value, bottom_binwidth = all_dist(PRE_NORM_BOTTOM_ONE_LOGIT)

    #Step3. 전체 영상에서 "상/하의" 영역에 대한 logit 분포도 -> 이를 통해 threshold range를 대략 잡음                   
    UPPER_RANGE, BOTTOM_RANGE = extract_thresholds(PRE_NORM_UPPER_ONE_LOGIT, PRE_NORM_BOTTOM_ONE_LOGIT, upper_max_value, upper_min_value, upper_binwidth, bottom_max_value, bottom_min_value, bottom_binwidth, image_names)
    
    return UPPER_RANGE, BOTTOM_RANGE, PRE_NORM_UPPER_ONE_LOGIT, PRE_NORM_BOTTOM_ONE_LOGIT

'''
function7. point값을 하나하나 매칭시키기 위한 function
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

'''
function8.
raw image, gt image, model image 반환
- 이때, model image는 이미 모델에 돌려서 나온 결과 pseudo image를 의미
- 또한, 이 이미지는 gt와 클래스 값이 맞지 않을 수 있기에 맞춰주기 위한 작업
'''
def load_images(image_name):
    #Step1. raw image
    raw_image_path = os.path.join(RAW_IMAGES_PATH, image_name+'.jpg')
    raw_image = Image.open(raw_image_path)
    raw_image = np.asarray(raw_image, dtype=np.uint8)
    raw_image = cv2.resize(raw_image, (64,128))

    #Step2. gt image
    gt_image = load_gt_image(GT_PATH, image_name)
    
    #Step3. model image and do mapping
    model_image_path = os.path.join(MODEL_PATH, image_name+'.png')
    model_image = Image.open(model_image_path)
    model_mapping = model_image.point(lambda p: map_dict.get(p, 0)) #gt와 동일한 값으로 맞춰주기 위한 mapping 작업
    model_mapping = np.asarray(model_mapping, dtype=np.uint8)   

    return raw_image, gt_image, model_mapping

'''
function9.
- target image와 상/하의 activation map과의 common part 구하는 함수
'''
def make_OL(target_image, upper_AM, bottom_AM):

    #Step1. 상의 공통부분
    upper_boolean = ((target_image==dic2clt['upper'])&(upper_AM==dic2clt['upper']))
    upper_OL = np.where(upper_boolean, dic2clt['upper'], 0)

    #Step2. 하의 공통부분
    bottom_boolean = ((target_image==dic2clt['bottom'])&(bottom_AM==dic2clt['bottom']))
    bottom_OL = np.where(bottom_boolean, dic2clt['bottom'], 0)
    
    return upper_OL, bottom_OL

def save_pseudo_labels(pseudo_image, root, image_name, palette):
    parsing_result_path = os.path.join(root, image_name + '.png')
    output_im = Image.fromarray(np.asarray(pseudo_image, dtype=np.uint8))
    output_im.putpalette(palette)
    output_im.save(parsing_result_path)

def save_am(am, root, image_name):
    heatmap_result_path = os.path.join(root, image_name + '.png')
    heatmapshow = cv2.normalize(am, None, alpha=0, beta=155, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_HSV)
    heatmapshow = Image.fromarray(np.asarray(heatmapshow, dtype=np.uint8))
    heatmapshow.save(heatmap_result_path)

#3. threshold 값 임의로 설정 - 곧 정할 것
# THRESHOLD_VALUES = {'upper': 7, 'bottom':7, 'dress':0}

##################################################################

#Step1. images& logit setting
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

# Adjust resizing to raw logits
# UPPER_ONE_LOGIT = preprocess_logits(deepcopy(one_logit_[:,:,:,[5,7]]))
# BOTTOM_ONE_LOGIT = preprocess_logits(deepcopy(one_logit_[:,:,:,[9, 12]]))

#전체 영상에 대해서 상의 logit 정규화(mean&std) 진행
NORM_UPPER_ONE_LOGIT = (deepcopy(one_logit_[:,:,:,[5,7]]) - UPPER_MEAN) / UPPER_STD         #N*H*W*CHANNEL
NORM_UPPER_MEAN, NORM_UPPER_STD, NORM_UPPER_MAX, NORM_UPPER_MIN = np.mean(NORM_UPPER_ONE_LOGIT), np.std(NORM_UPPER_ONE_LOGIT), \
                                                                    np.max(NORM_UPPER_ONE_LOGIT), np.min(NORM_UPPER_ONE_LOGIT)
#전체 영상에 대해서 하의 logit 정규화 진행
NORM_BOTTOM_ONE_LOGIT = (deepcopy(one_logit_[:,:,:,[9, 12]]) - BOTTOM_MEAN) / BOTTOM_STD    #N*H*W*CHANNEL
NORM_BOTTOM_MEAN, NORM_BOTTOM_STD, NORM_BOTTOM_MAX, NORM_BOTTOM_MIN = np.mean(NORM_BOTTOM_ONE_LOGIT), np.std(NORM_BOTTOM_ONE_LOGIT), \
                                                                    np.max(NORM_BOTTOM_ONE_LOGIT), np.min(NORM_BOTTOM_ONE_LOGIT)

_, _, PRE_NORM_UPPER_ONE_LOGIT, PRE_NORM_BOTTOM_ONE_LOGIT = preprocess()
#위 과정을 통해 대략적으로 설정해본 threshold range
THR_UPPER_RANGE, THR_BOTTOM_RANGE = [2.11, 3.84], [1.91, 3.82] 
map_dict, dic2clt, LIP = make_dicmap('LIP')

model_not_confidence = []
gt_not_confidence = []

'''
CASE1: 수도 레이블과 thresholded Activation Map
CASE2: GT 레이블과 thresholded Activation Map
'''
CASE1 = deepcopy({image: dict.fromkeys(deepcopy(list(LIP.keys()))) for image in image_names})
CASE2 = deepcopy(CASE1)

##################################################################

for idx, (image_name, log, upper_norm_log, bottom_norm_log) in enumerate(zip(image_names, one_logit_, PRE_NORM_UPPER_ONE_LOGIT, PRE_NORM_BOTTOM_ONE_LOGIT)):
    print(f'{idx+1}번 !!{image_name} 입니다!!')
    
    #Step1. gt 형태와 동일한 매핑 dictionary 설정
    '''
    dic2clt = {'upper':2, 'bottom':1, 'dress':3}
    map_dict =
    LIP = {'upper':[5,7], 'bottom':[9,12], 'dress':[6,10]}
    '''

    #Step2. raw_image, gt_image, model_image load
    raw_image, gt_image, model_image = load_images(image_name)

    #Step3. part 별 activation map 추출
    #Step3-1~3-2 상/하의 - activation map(adjusted by theshold), activation map
    upper_thresholded_AM = np.where((upper_norm_log >= THR_UPPER_RANGE[0])  \
                            &(upper_norm_log <= THR_UPPER_RANGE[1]), dic2clt['upper'], 0)
    bottom_thresholded_AM = np.where((bottom_norm_log >= THR_BOTTOM_RANGE[0])  \
                            &(bottom_norm_log <= THR_BOTTOM_RANGE[1]), dic2clt['bottom'], 0)
    
    #Step4. 결과 저장
    #pseudo label 저장
    save_pseudo_labels(upper_thresholded_AM, SAVE_ROOT_NORM_PSEUDO_UPPER, image_name, UPPER_PALETTE)
    save_pseudo_labels(bottom_thresholded_AM, SAVE_ROOT_NORM_PSEUDO_BOTTOM, image_name, BOTTOM_PALETTE)

    #nonorm activation
    # save_am(upper_norm_log, SAVE_ROOT_NORM_UPPER, image_name)
    # save_am(bottom_norm_log, SAVE_ROOT_NORM_BOTTOM, image_name)

    #norm activation
    save_am(upper_norm_log, SAVE_ROOT_NORM_UPPER, image_name)
    save_am(bottom_norm_log, SAVE_ROOT_NORM_BOTTOM, image_name)

    #pseudo&activation map 간 라벨링 개수 확인
    upper_bottom_AM_labels = np.unique(np.append(np.unique(upper_thresholded_AM), np.unique(bottom_thresholded_AM)))
    model_labels = np.unique(model_image)
    gt_labels = np.unique(gt_image)

    if len(upper_bottom_AM_labels)!=len(model_labels) or not((upper_bottom_AM_labels==model_labels).all()):
        model_not_confidence.append(image_name)
        continue
    if len(upper_bottom_AM_labels)!=len(gt_labels) or not((upper_bottom_AM_labels==gt_labels).all()):        #일단 gt가 안 맞아도 넘기는 걸로 -> 그리고 이 배열들을 비교해보자
        gt_not_confidence.append(image_name)
        continue

    #gt&activation map 간 라벨링 개수 확인
    #Step5. (pseudo label & GT) & activation map의 공통부분
    '''
    #CASE1: 수도 레이블과 thresholded Activation Map
    #CASE2: GT 레이블과 thresholded Activation Map
    '''
    #Step5-1. Case1 상/하의 겹치는 부분
    Case1_Upper_OL, Case1_Bottom_OL = make_OL(model_image, upper_thresholded_AM, bottom_thresholded_AM) #예외2. overlapped의 경우, np.unique = [0, 2] - 하지만 0만 있을 경우도 존재
    #Step5-2. Case2 상/하의 겹치는 부분
    Case2_Upper_OL, Case2_Bottom_OL = make_OL(gt_image, upper_thresholded_AM, bottom_thresholded_AM)    #예외2. overlapped의 경우, np.unique = [0, 1] - 하지만 0만 있을 경우도 존재 

    #Step5. part 별 pseudo label과 GT의 빈도수
    #Step5-1. pseudo label & GT & am의 빈도수
    cnt_model = np.bincount(model_image.reshape(-1)) #[bkg,bottom,upper]  array([5219, 1177, 1796])       #예외3. 본 model or gt_image의 경우, np.unique = [0, 1, 2] - 하지만 부분만 있을 경우도 존재
    cnt_gt = np.bincount(gt_image.reshape(-1)) #[bkg,bottom,upper]  array([5364, 1150, 1678])             #예외3. 본 model or gt_image의 경우, np.unique = [0, 1, 2] - 하지만 부분만 있을 경우도 존재

    #Step5-2. am의 빈도수
    cnt_am_upper = np.bincount(upper_thresholded_AM.reshape(-1))[2] #[bkg,bottom,upper]  array([5364, 1150, 1678])
    cnt_am_bottom = np.bincount(bottom_thresholded_AM.reshape(-1))[1] #[bkg,bottom,upper]  array([5364, 1150, 1678])
    
    #Step5-3-1. Case1 상/하의 빈도수
    cnt_Case1_Upper, cnt_Case1_Bottom = np.bincount(Case1_Upper_OL.reshape(-1))[2], np.bincount(Case1_Bottom_OL.reshape(-1))[1]
    #Step5-3-2. Case2 상/하의 빈도수
    cnt_Case2_Upper, cnt_Case2_Bottom = np.bincount(Case2_Upper_OL.reshape(-1))[2], np.bincount(Case2_Bottom_OL.reshape(-1))[1]

    CASE1[image_name]['upper'] = ((cnt_Case1_Upper/cnt_model[2] * 100), (cnt_Case1_Upper/cnt_am_upper * 100))      #(pseudo, activation map)
    CASE1[image_name]['bottom'] = ((cnt_Case1_Bottom/cnt_model[1] * 100), (cnt_Case1_Bottom/cnt_am_bottom * 100))  #(pseudo, activation map)

    # CASE1[image_name]['upper'] = (cnt_Case1_Upper/cnt_am_upper * 100)       #activation map
    # CASE1[image_name]['bottom'] = (cnt_Case1_Bottom/cnt_am_bottom * 100)     #activation map

    #Step5-2. Case2 - gt, activation map
                                #(gt에서의 비율, activation map(고정)에서의 비율)
    CASE2[image_name]['upper'] = ((cnt_Case2_Upper/cnt_gt[2] * 100), (cnt_Case2_Upper/cnt_am_upper * 100))      #(pseudo, activation map)
    CASE2[image_name]['bottom'] = ((cnt_Case2_Bottom/cnt_gt[1] * 100), (cnt_Case2_Bottom/cnt_am_bottom * 100))  #(pseudo, activation map)

    # except:
    # errors.append(image_name)
    # print(f"에러 존재!! {image_name}")
print("case1, case2 출력")
