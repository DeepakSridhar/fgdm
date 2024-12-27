import os, sys
from PIL import Image
import numpy as np
import csv
import shutil
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader, Dataset


class BatchColorize(object):
    def __init__(self, n=150, custom_palette=None):
        if custom_palette is not None:
            self.cmap = custom_palette
        else:
            self.cmap = color_map(n)
        # print(self.cmap[170:172])
        # self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((size[0], 3, size[1], size[2]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[:,0][mask] = self.cmap[label][0]
            color_image[:,1][mask] = self.cmap[label][1]
            color_image[:,2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[:,0][mask] = color_image[:,1][mask] = color_image[:,2][mask] = 255

        return color_image

def get_max_coordinates(image_array):

    # Find the maximum value and its coordinates
    max_value = np.max(image_array)
    # print(image_array)
    # print(np.sum(image_array), max_value)
    max_indices = np.where(image_array == max_value)

    # print(max_indices)

    # Extract the row and column coordinates
    row_coordinates = max_indices[0]
    column_coordinates = max_indices[1]

    # Return the coordinates as a tuple
    return [row_coordinates[0], column_coordinates[0]]


class BatchDeColorize(object):
    def __init__(self, n=40, custom_palette=None):
        if custom_palette is not None:
            self.cmap = custom_palette
        else:
            self.cmap = color_map(n)
        # print(self.cmap)
        # self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, rgb_image, return_points=False, dst=''):
        size = rgb_image.shape
        gray_image = np.zeros((size[0], size[1], size[2]), dtype=np.float32) - 1
        eps1 = 28
        eps = eps1
        pts = []
        for label in range(0, len(self.cmap)):
            tmp = np.zeros_like(rgb_image, dtype=int)
            tmp[...,0] = self.cmap[label][0]
            tmp[...,1] = self.cmap[label][1]
            tmp[...,2] = self.cmap[label][2]
            mask = (tmp == rgb_image)
            # m = np.prod(mask, -1).astype(bool)   
                
            m11 = np.maximum(tmp[...,0] - eps1, 0) <= rgb_image[...,0]
            m12 = rgb_image[...,0] <= np.minimum(tmp[...,0] + eps, 255)
            m21 = np.maximum(tmp[...,1] - eps1, 0) <= rgb_image[...,1] 
            m22 = rgb_image[...,1] <= np.minimum(tmp[...,1] + eps, 255)
            m31 = np.maximum(tmp[...,2] - eps1, 0) <= rgb_image[...,2]
            m32 = rgb_image[...,2] <= np.minimum(tmp[...,2] + eps, 255)
            m1 = np.logical_and(m11, m12)
            m2 = np.logical_and(m21, m22)
            m3 = np.logical_and(m31, m32) 
            m = np.logical_and(m1, m2)
            m = np.logical_and(m, m3) 
          
            gray_image[m] = label            
            if return_points:
                m11 = np.maximum(tmp[...,0] - eps1, 0) <= rgb_image[...,0]
                m12 = rgb_image[...,0] <= np.minimum(tmp[...,0] + eps, 255)
                m21 = np.maximum(tmp[...,1] - eps1, 0) <= rgb_image[...,1] 
                m22 = rgb_image[...,1] <= np.minimum(tmp[...,1] + eps, 255)
                m31 = np.maximum(tmp[...,2] - eps1, 0) <= rgb_image[...,2]
                m32 = rgb_image[...,2] <= np.minimum(tmp[...,2] + eps, 255)
                m1 = np.logical_and(m11, m12)
                m2 = np.logical_and(m21, m22)
                m3 = np.logical_and(m31, m32) 
                m = np.logical_and(m1, m2)
                m = np.logical_and(m, m3) 
                pts.append(get_max_coordinates(m*1))

        # handle void
        mask = (-1 == gray_image)
        gray_image[mask] = 182
        if return_points:
            return pts
        return gray_image[0]

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

num_classes = 183
decolorize = BatchDeColorize(num_classes)
colorize = BatchColorize(num_classes)

def ade_palette():
    """ADE20K palette for external use."""
    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255], [255, 255, 255]]

colorize_ade = BatchColorize(151, custom_palette=ade_palette())
decolorize_ade = BatchDeColorize(151, custom_palette=ade_palette())
# The given dictionary
lab_mapping = {
    '0': 12, '1': 127, '2': 20, '4': 90, '5': 80, '7': 83, '8': 76, '9': 136,
    '14': 69, '38': 150, '39': 150, '43': 98, '44': 142, '53': 150, '56': 150,
    '61': 19, '64': 7, '65': 27, '67': 8, '68': 33, '69': 65, '70': 14, '77': 124,
    '78': 118, '80': 47, '81': 50, '83': 67, '84': 148, '85': 135, '92': 131,
    '94': 61, '96': 150, '97': 10, '99': 150, '100': 150, '106': 45, '108': 18,
    '112': 32, '118': 66, '123': 9, '126': 68, '127': 25, '129': 82, '134': 16,
    '140': 57, '145': 38, '147': 60, '148': 6, '149': 34, '151': 28, '153': 46,
    '154': 26, '155': 24, '157': 48, '160': 53, '164': 15, '165': 114, '167': 81,
    '168': 4, '169': 150, '170': 0, '171': 0, '172': 0, '173': 0, '174': 0, '175': 0,
    '176': 0, '95': 1, '156': 2, '113': 3, '114': 3, '115': 3, '116': 3, '117': 3,
    '11': 4, '101': 5, '102': 5, '146': 6, '179': 8, '180': 8, '111': 14, '66': 15,
    '63': 17, '141': 17, '177': 21, '178': 21, '132': 27, '144': 29, '109': 33,
    '32': 55, '30': 115, '36': 119, '120': 120, '29': 147, '45': 147, '182': 150,
    '3': 150, '6': 150, '10': 150, '12': 150, '13': 150, '15': 150, '16': 150,
    '17': 150, '18': 150, '19': 150, '20': 150, '21': 150, '22': 150, '23': 150,
    '24': 150, '25': 150, '26': 150, '27': 150, '28': 150, '31': 150, '33': 150,
    '34': 150, '35': 150, '37': 150, '40': 150, '41': 150, '42': 150, '46': 150,
    '47': 150, '48': 150, '49': 150, '50': 150, '51': 150, '52': 150, '54': 150,
    '55': 150, '57': 150, '58': 150, '59': 150, '60': 150, '62': 150, '71': 150,
    '72': 150, '73': 150, '74': 150, '75': 150, '76': 150, '79': 150, '82': 150,
    '86': 150, '87': 150, '88': 150, '89': 150, '90': 150, '91': 150, '93': 150,
    '98': 150, '103': 150, '104': 150, '105': 150, '107': 150, '110': 150,
    '119': 150, '121': 150, '122': 150, '124': 150, '125': 150, '128': 150,
    '130': 150, '131': 150, '133': 150, '135': 150, '136': 150, '137': 150,
    '138': 150, '139': 150, '142': 150, '143': 150, '150': 150, '152': 150,
    '158': 150, '159': 150, '161': 150, '162': 150, '163': 150, '166': 150,
    '181': 150
}
label_mapping = {}
for k, v in lab_mapping.items():
    label_mapping[int(k)] = v

ade_label_mapping = {}
for k, v in label_mapping.items():
    if v == 150:
        k= 181
    ade_label_mapping[v] = int(k)

for i in range(150):
    if i == 106:
        ade_label_mapping[i] = 4
    if i not in ade_label_mapping:
        ade_label_mapping[i] = 181