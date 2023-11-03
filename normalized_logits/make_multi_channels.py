'''
원하는 class(upper/bottom/bkg)에 해당하는 activation map을 따로 저장한 경우
'''
import os
import cv2
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt

SAVE_ROOT_NORM = '/normalized activation maps을 저장한 경로'
SAVE_ROOT_NORM_UPPER, SAVE_ROOT_NORM_BOTTOM, SAVE_ROOT_NORM_BKG = os.path.join(SAVE_ROOT_NORM, 'upper'), os.path.join(SAVE_ROOT_NORM, 'bottom'), os.path.join(SAVE_ROOT_NORM, 'bkg')
images_names = os.listdir(SAVE_ROOT_NORM_UPPER)

#채널별로 합친 activation map을 저장할 경로
SAVE_THREE_CHANNELS = os.path.join(SAVE_ROOT_NORM, 'stacked')

def convert_to_l(image_path):
    raw_image = Image.open(image_path)
    output_im = np.asarray(raw_image, dtype=np.uint8) #np.asarray: 동일한 객체 반환 / np.array: 다른 객체 생성
    output_im = Image.fromarray(output_im)
    output_im = output_im.convert('L') #굳이 안 해도 됨

    return output_im

for image_name in images_names:
    print(f'!!{image_name} 입니다!!')

    upper = os.path.join(SAVE_ROOT_NORM_UPPER, image_name)
    bottom = os.path.join(SAVE_ROOT_NORM_BOTTOM, image_name)
    bkg = os.path.join(SAVE_ROOT_NORM_BKG, image_name)

    output_im1 = convert_to_l(upper)
    output_im2 = convert_to_l(bottom)
    output_im3 = convert_to_l(bkg)

    stacked_image = np.stack((output_im1, output_im2, output_im3), axis=-1)
    plt.imshow(stacked_image)

    stacked_image = Image.fromarray(np.asarray(stacked_image, dtype=np.uint8))
    stacked_image.save(os.path.join(SAVE_THREE_CHANNELS, image_name))