#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   evaluate.py
@Time    :   8/4/19 3:36 PM
@Desc    :
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import os
import argparse
import numpy as np
import torch

from torch.utils import data
from tqdm import tqdm
from PIL import Image as PILImage
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import networks
from datasets.datasets import LIPDataValSet
from utils.miou import compute_mean_ioU_4classes
from utils.transforms import BGR2RGB_transform
from utils.transforms import transform_parsing

ARCH = 'swin_tiny'
DATA_DIR = '/data/SOLIDER-HumanParsing/data/LIP'
BS = 1
INPUT_SIZE = '473,473'
NUM_CLASSES = 20
IGNORE = 255
RESTORE = './SOLIDER/log/swin_tiny.pth'

RESULTS_DIR = '/data/SOLIDER-HumanParsing/results'
MODEL_NAME = 'lip_solider_swin_tiny'

LABELS = ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat', \
          'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm', 'Left-leg',
          'Right-leg', 'Left-shoe', 'Right-shoe']

LIP = ['Background', 'bottom', 'upper', 'dress']

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    # Network Structure
    parser.add_argument("--arch", type=str, default=ARCH)
    # Data Preference
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=BS)
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE)
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--ignore-label", type=int, default=IGNORE)
    parser.add_argument("--random-mirror", action="store_true")
    parser.add_argument("--random-scale", action="store_true")
    # Evaluation Preference
    parser.add_argument("--log-dir", type=str, default='./log')
    parser.add_argument("--model-restore", type=str, default=RESTORE)
    
    parser.add_argument("--gpu", type=str, default='0', help="choose gpu device.")
    
    parser.add_argument("--save-results", action="store_true", help="whether to save the results.")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--data-name", type=str)
    
    parser.add_argument("--flip", action="store_true", help="random flip during the test.")
    parser.add_argument("--multi-scales", type=str, default='1', help="multiple scales during the test")
    return parser.parse_args()

def get_palette_4classes():
    palette = [0,0,0,   #Background
            0,0,0,      #Hat
            0,0,0,      #Hair
            0,0,0,      #Glove
            0,0,0,      #Sunglasses
            0,128,0,    #Upper-clothes
            0,0,85,     #Dress
            0,128,0,    #Coat
            0,0,0,      #Socks
            128,0,0,    #Pants
            0,0,85,     #Jumpsuits
            0,0,0,      #Scarf
            128,0,0,    #Skirt
            0,0,0,      #Face
            0,0,0,      #Left-arm
            0,0,0,      #Right-arm
            0,0,0,      #Left-leg
            0,0,0,      #Right-leg
            0,0,0,      #Left-shoe
            0,0,0]      #Right-shoe
    return palette  

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

                                #resize되어서 들어오는 input image
def multi_scale_testing(model, batch_input_im, crop_size=[473, 473], flip=True, multi_scales=[1]):
    flipped_idx = (15, 14, 17, 16, 19, 18)
    if len(batch_input_im.shape) > 4:
        batch_input_im = batch_input_im.squeeze()
    if len(batch_input_im.shape) == 3:
        batch_input_im = batch_input_im.unsqueeze(0)

    interp = torch.nn.Upsample(size=crop_size, mode='bilinear', align_corners=True)
    ms_outputs = []
    for s in multi_scales:
        interp_im = torch.nn.Upsample(scale_factor=s, mode='bilinear', align_corners=True)
        scaled_im = interp_im(batch_input_im)
        parsing_output = model(scaled_im)
        parsing_output = parsing_output[0][-1] # torch.Size([1, 20, 72, 48]) - [[parsing_result, fusion_result], [edge_result]] -> fusion result
        output = parsing_output[0]             # torch.Size([20, 72, 48])
        if flip:
            flipped_output = parsing_output[1]
            flipped_output[14:20, :, :] = flipped_output[flipped_idx, :, :]
            output += flipped_output.flip(dims=[-1])
            output *= 0.5
        output = interp(output.unsqueeze(0))
        ms_outputs.append(output[0])                            # torch.Size([20, 72, 48])
    ms_fused_parsing_output = torch.stack(ms_outputs)           #torch.Size([5, 20, 572, 384])
    ms_fused_parsing_output = ms_fused_parsing_output.mean(0)   #torch.Size([20, 572, 384])
    ms_fused_parsing_output = ms_fused_parsing_output.permute(1, 2, 0)  # HWC - permute 후, torch.Size([572, 384, 20])
    parsing = torch.argmax(ms_fused_parsing_output, dim=2)      #torch.Size([572, 384])

    parsing = parsing.data.cpu().numpy()
    ms_fused_parsing_output = ms_fused_parsing_output.data.cpu().numpy()

    return parsing, ms_fused_parsing_output #, list(model.parameters())[-2].data.cpu().numpy()


def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    multi_scales = [float(i) for i in args.multi_scales.split(',')]
    gpus = [int(i) for i in args.gpu.split(',')]
    assert len(gpus) == 1
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cudnn.benchmark = True
    cudnn.enabled = True

    h, w = map(int, args.input_size.split(','))
    input_size = [h, w]

    model = networks.init_model(args.arch, num_classes=args.num_classes, pretrained=None)

    IMAGE_MEAN = model.mean
    IMAGE_STD = model.std
    INPUT_SPACE = model.input_space
    print('image mean: {}'.format(IMAGE_MEAN))
    print('image std: {}'.format(IMAGE_STD))
    print('input space:{}'.format(INPUT_SPACE))
    if INPUT_SPACE == 'BGR':
        print('BGR Transformation')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN,
                                 std=IMAGE_STD),
        ])
    if INPUT_SPACE == 'RGB':
        print('RGB Transformation')
        transform = transforms.Compose([
            transforms.ToTensor(),
            BGR2RGB_transform(),
            transforms.Normalize(mean=IMAGE_MEAN,
                                 std=IMAGE_STD),
        ])

    # Data loader
    lip_test_dataset = LIPDataValSet(args.data_dir, 'val', crop_size=input_size, transform=transform, flip=args.flip)
    num_samples = len(lip_test_dataset)
    print('Totoal testing sample numbers: {}'.format(num_samples))
    testloader = data.DataLoader(lip_test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # Load model weight
    state_dict = torch.load(args.model_restore)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    sp_results_dir = os.path.join(args.results_dir, args.model_name)
    if not os.path.exists(sp_results_dir):
        os.makedirs(sp_results_dir)

    # palette = get_palette(20)
    palette = get_palette_4classes() #integrate classes
    parsing_preds = []
    scales = np.zeros((num_samples, 2), dtype=np.float32)
    centers = np.zeros((num_samples, 2), dtype=np.int32)

    #Set variables for making activation maps
    ##############################################################################
    # images_name = []
    # npy_logits = np.empty((572, 384, 20))
    ##############################################################################

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(testloader)):
            image, meta = batch
            if (len(image.shape) > 4):
                image = image.squeeze()
            im_name = meta['name'][0]
            c = meta['center'].numpy()[0]
            s = meta['scale'].numpy()[0]
            w = meta['width'].numpy()[0]
            h = meta['height'].numpy()[0]
            scales[idx, :] = s
            centers[idx, :] = c

            parsing, _ = multi_scale_testing(model, image.cuda(), crop_size=input_size, flip=args.flip,
                                                  multi_scales=multi_scales)
            
            #Stack logits
            # if (npy_logits==0).all():
            #     npy_logits = _.copy()
            # else:
            #     npy_logits= np.stack([npy_logits, _], axis=0)

            if args.save_results:
                parsing_result = transform_parsing(parsing, c, s, w, h, input_size)
                parsing_result_path = os.path.join(sp_results_dir, im_name + '.png')
                output_im = PILImage.fromarray(np.asarray(parsing_result, dtype=np.uint8))
                output_im.putpalette(palette)
                output_im.save(parsing_result_path)
            
            parsing_preds.append(parsing)

    assert len(parsing_preds) == num_samples
    mIoU = compute_mean_ioU_4classes(parsing_preds, scales, centers, len(LIP), args.data_dir, input_size)
    print(mIoU)
    return

if __name__ == '__main__':
    main()