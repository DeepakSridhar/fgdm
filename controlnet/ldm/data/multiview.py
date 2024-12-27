import os
import math
import random

from PIL import Image
import blobfile as bf
# from mpi4py import MPI
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

class_dict = {
    0: 'bag',
    1: 'bottle',
    2: 'washer',
    3: 'vessel',
    4: 'train',
    5: 'telephone',
    6: 'table',
    7: 'stove',
    8: 'sofa',
    9: 'skateboard',
    10: 'rifle',
    11: 'pistol',
    12: 'remote control',
    13: 'printer',
    14: 'flowerpot',
    15: 'pillow',
    16: 'piano',
    17: 'mug',
    18: 'motorcycle',
    19: 'microwave',
    20: 'microphone',
    21: 'mailbox',
    22: 'loudspeaker',
    23: 'laptop',
    24: 'lamp',
    25: 'knife',
    26: 'pot',
    27: 'helmet',
    28: 'guitar',
    29: 'bookshelf',
    30: 'faucet',
    31: 'earphone',
    32: 'display',
    33: 'dishwasher',
    34: 'computer keyboard',
    35: 'clock',
    36: 'chair',
    37: 'car',
    38: 'cap',
    39: 'can',
    40: 'camera',
    41: 'cabinet',
    42: 'bus',
    43: 'bowl',
    44: 'bicycle',
    45: 'bench',
    46: 'bed',
    47: 'bathtub',
    48: 'basket',
    49: 'ashcan',
    50: 'airplane',
    51: 'umbrella',
    52: 'plush toy',
    53: 'toy figure',
    54: 'towel',
    55: 'toothbrush',
    56: 'toy bear',
    57: 'toy cat',
    58: 'toy bird',
    59: 'toy insect',
    60: 'toy cow',
    61: 'toy dog',
    62: 'toy monkey',
    63: 'toy elephant',
    64: 'toy fish',
    65: 'toy horse',
    66: 'toy sheep',
    67: 'toy mouse',
    68: 'toy tiger',
    69: 'toy rabbit',
    70: 'toy dragon',
    71: 'toy snake',
    72: 'toy chook',
    73: 'toy pig',
    74: 'rice cooker',
    75: 'pressure cooker',
    76: 'toaster',
    77: 'dryer',
    78: 'battery',
    79: 'curtain',
    82: 'blackboard eraser',
    83: 'bucket',
    85: 'calculator',
    86: 'candle',
    87: 'cassette',
    88: 'cup sleeve',
    90: 'computer mouse',
    93: 'easel',
    94: 'fan',
    96: 'cookie',
    97: 'fries',
    98: 'donut',
    99: 'coat rack',
    100: 'guitar stand',
    101: 'can opener',
    102: 'flashlight',
    103: 'hammer',
    104: 'scissors',
    105: 'screw driver',
    106: 'spanner',
    107: 'hanger',
    108: 'jug',
    109: 'fork',
    110: 'chopsticks',
    111: 'spoon',
    112: 'ladder',
    113: 'ceiling lamp',
    114: 'wall lamp',
    115: 'lamp post',
    116: 'light switch',
    118: 'mirror',
    119: 'paper box',
    120: 'wheelchair',
    121: 'walking stick',
    122: 'picture frame',
    124: 'shower',
    125: 'toilet',
    126: 'sink',
    127: 'power socket',
    129: 'Bagged snacks',
    130: 'Tripod',
    131: 'Selfie stick',
    132: 'Hair dryer',
    133: 'Lipstick',
    134: 'Glasses',
    135: 'Sanitary napkin',
    136: 'Toilet paper',
    137: 'Rockery',
    138: 'Chinese hot dishes',
    139: 'Root carving',
    141: 'Flower',
    144: 'Book',
    145: 'Pipe PVC Metal pipe',
    146: 'Projector',
    147: 'Cabinet Air Conditioner',
    148: 'Desk Air Conditioner',
    149: 'Refrigerator',
    150: 'Percussion',
    152: 'Strings',
    153: 'Wind instruments',
    154: 'Balloons',
    155: 'Scarf',
    156: 'Shoe',
    157: 'Skirt',
    158: 'Pants',
    159: 'Clothing',
    160: 'Box',
    161: 'Soccer',
    162: 'Roast Duck',
    163: 'Pizza',
    164: 'Ginger',
    165: 'Cauliflower',
    166: 'Broccoli',
    167: 'Cabbage',
    168: 'Eggplant',
    169: 'Pumpkin',
    170: 'winter melon',
    171: 'Tomato',
    172: 'Corn',
    173: 'Sunflower',
    174: 'Potato',
    175: 'Sweet potato',
    176: 'Chinese cabbage',
    177: 'Onion',
    178: 'Momordica charantia',
    179: 'Chili',
    180: 'Cucumber',
    181: 'Grapefruit',
    182: 'Jackfruit',
    183: 'Star fruit',
    184: 'Avocado',
    185: 'Shakyamuni',
    186: 'Coconut',
    187: 'Pineapple',
    188: 'Kiwi',
    189: 'Pomegranate',
    190: 'Pawpaw',
    191: 'Watermelon',
    192: 'Apple',
    193: 'Banana',
    194: 'Pear',
    195: 'Cantaloupe',
    196: 'Durian',
    197: 'Persimmon',
    198: 'Grape',
    199: 'Peach',
    200: 'power strip',
    202: 'Racket',
    203: 'Toy butterfly',
    204: 'Toy duck',
    205: 'Toy turtle',
    206: 'Bath sponge',
    207: 'Glove',
    208: 'Badminton',
    209: 'Lantern',
    211: 'Chestnut',
    212: 'Accessory',
    214: 'Shovel',
    215: 'Cigarette',
    216: 'Stapler',
    217: 'Lighter',
    218: 'Bread',
    219: 'Key',
    220: 'Toothpaste',
    221: 'Swin ring',
    222: 'Watch',
    223: 'Telescope',
    224: 'Eggs',
    225: 'Bun',
    226: 'Guava',
    227: 'Okra',
    228: 'Tangerine',
    229: 'Lotus root',
    230: 'Taro',
    231: 'Lemon',
    232: 'Garlic',
    233: 'Mango',
    234: 'Sausage',
    235: 'Besom',
    237: 'Lock',
    238: 'Ashtray',
    240: 'Conch',
    241: 'Seafood',
    243: 'Hairbrush',
    244: 'Ice cream',
    245: 'Razor',
    246: 'Adhesive hook',
    247: 'Hand Warmer',
    250: 'Thermometer',
    251: 'Bell',
    252: 'Sugarcane',
    253: 'Adapter(Water pipe)',
    254: 'Calendar',
    261: 'Insecticide',
    263: 'Electric saw',
    265: 'Inflator',
    266: 'Ironmongery',
    267: 'Bulb'
}



def load_data(
    *,
    data_dir,
    image_size,
    random_crop=False,
    random_flip=False,
    is_train=True, 
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param image_size: the size to which images are resized.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    dataset = MultiViewDataset(
        image_size,
        data_folder=data_dir,
        random_crop=random_crop,
        random_flip=random_flip,
        is_train=is_train
    )

    return dataset


class MultiViewDataset(Dataset):
    def __init__(self, 
        resolution=256,
        data_folder=None,
        random_crop=False,
        random_flip=False,
        select_idxs=[81,160,1],
        is_train=True):
        self.data_folder = data_folder
        self.is_train = is_train
        self.resolution = resolution

        self.random_crop = random_crop
        self.random_flip = random_flip


        self.samples = []
        for class_folder in os.listdir(data_folder):
            class_path = os.path.join(data_folder, class_folder)
            if os.path.isdir(class_path):
                if 'MV' in data_folder:
                    select_idxs=[29,15,1]
                    
                    for dir in os.listdir(class_path):
                        li = os.listdir(os.path.join(class_path, dir, 'images'))
                        if len(li) > 31:
                            select_idxs = [li[-1], li[len(li)//2], li[0]]
                        else:
                            select_idxs = [li[-1], li[len(li)//2], li[0]]
                        # sample_paths = [os.path.join(class_path, dir, 'images', f"{j:03d}.jpg") for j in select_idxs]
                        sample_paths = [os.path.join(class_path, dir, 'images', f"{item}") for item in select_idxs]
                        self.samples.append(sample_paths)
                else:
                    for i in range(1, 11):  # Loop through all 10 types
                        sample_paths = [os.path.join(class_path, f"{class_folder}_{i:01d}_{j:03d}.png") for j in select_idxs]
                        self.samples.append(sample_paths)
        # print("Len of Dataset:", len(self.samples))
        # Split the dataset into train and validation sets
        train_samples, val_samples = train_test_split(self.samples, test_size=0.1, random_state=42)
        self.samples = train_samples if self.is_train else val_samples
        print("Len of Dataset:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        out_dict = {}
        sample_paths = self.samples[index]
        images = [Image.open(path).convert('RGB') for path in sample_paths]
        cls = sample_paths[0].split('/')[3]

        if self.is_train:
            if self.random_crop:
                arr_images = random_crop_arr(images, self.resolution)
            else:
                arr_images = center_crop_arr(images, self.resolution)
        else:
            arr_images = resize_arr(images, self.resolution, keep_aspect=False)

        if self.random_flip and random.random() < 0.5:
            arry_images = []
            for arr_image in arr_images:
                arr_image = arr_image[:, ::-1].copy()
                arry_images.append(arr_image)
        arry_images = []
        for arr_image in arr_images:
                arr_image = arr_image.astype(np.float32) / 127.5 - 1
                arry_images.append(arr_image)
        
        out_dict["image"] = np.concatenate(arry_images, 2)
        out_dict["caption"] = class_dict[cls]
        return out_dict

if '__name__' == '__main__':
    # Example usage
    data_folder = "/data8/deepak/MIRO/MIRO/"

    dataset = MultiViewDataset(data_folder)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Iterate over batches
    for batch in dataloader:
        # batch will contain 3 images from the same type
        image_type_1, image_type_2, image_type_3 = batch[0]  # Unpack the list of images
        print(image_type_1.shape, image_type_2.shape, image_type_3.shape)


def resize_arr(pil_list, image_size, keep_aspect=True):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    pil_image, pil_class, pil_instance = pil_list

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    if keep_aspect:
        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
    else:
        pil_image = pil_image.resize((image_size, image_size), resample=Image.BICUBIC)

    pil_class = pil_class.resize(pil_image.size, resample=Image.NEAREST)
    if pil_instance is not None:
        pil_instance = pil_instance.resize(pil_image.size, resample=Image.NEAREST)
    
    # if pil_pose is not None:
    #     pil_pose = pil_pose.resize(pil_image.size, resample=Image.NEAREST)

    arr_image = np.array(pil_image)
    arr_class = np.array(pil_class)
    arr_instance = np.array(pil_instance) if pil_instance is not None else None
    # arr_pose = np.array(pil_pose) if pil_pose is not None else None
    return arr_image, arr_class, arr_instance


def center_crop_arr(pil_list, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    pil_image, pil_class, pil_instance = pil_list

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    pil_class = pil_class.resize(pil_image.size, resample=Image.NEAREST)
    if pil_instance is not None:
        pil_instance = pil_instance.resize(pil_image.size, resample=Image.NEAREST)
    
    # if pil_pose is not None:
    #     pil_pose = pil_pose.resize(pil_image.size, resample=Image.NEAREST)

    arr_image = np.array(pil_image)
    arr_class = np.array(pil_class)
    arr_instance = np.array(pil_instance) if pil_instance is not None else None
    # arr_pose = np.array(pil_pose) if pil_pose is not None else None
    crop_y = (arr_image.shape[0] - image_size) // 2
    crop_x = (arr_image.shape[1] - image_size) // 2
    return arr_image[crop_y : crop_y + image_size, crop_x : crop_x + image_size],\
           arr_class[crop_y: crop_y + image_size, crop_x: crop_x + image_size],\
           arr_instance[crop_y : crop_y + image_size, crop_x : crop_x + image_size] if arr_instance is not None else None#,\
        #    arr_pose[crop_y : crop_y + image_size, crop_x : crop_x + image_size] if arr_pose is not None else None


def random_crop_arr(pil_list, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    pil_image, pil_class, pil_instance = pil_list

    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    pil_class = pil_class.resize(pil_image.size, resample=Image.NEAREST)
    if pil_instance is not None:
        pil_instance = pil_instance.resize(pil_image.size, resample=Image.NEAREST)
    # if pil_pose is not None:
    #     pil_pose = pil_pose.resize(pil_image.size, resample=Image.NEAREST)

    arr_image = np.array(pil_image)
    arr_class = np.array(pil_class)
    arr_instance = np.array(pil_instance) if pil_instance is not None else None
    # arr_pose = np.array(pil_pose) if pil_pose is not None else None
    crop_y = random.randrange(arr_image.shape[0] - image_size + 1)
    crop_x = random.randrange(arr_image.shape[1] - image_size + 1)
    return arr_image[crop_y : crop_y + image_size, crop_x : crop_x + image_size],\
           arr_class[crop_y: crop_y + image_size, crop_x: crop_x + image_size],\
           arr_instance[crop_y : crop_y + image_size, crop_x : crop_x + image_size] if arr_instance is not None else None#,\
        #    arr_pose[crop_y : crop_y + image_size, crop_x : crop_x + image_size] if arr_pose is not None else None
