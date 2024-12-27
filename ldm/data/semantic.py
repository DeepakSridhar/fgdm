import os
import math
import random

from PIL import Image
import blobfile as bf

import pandas as pd
import numpy as np
import json
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset

from ldm.data.color_mapping import colorize_ade, decolorize_ade, label_mapping

import cv2
cv2.setNumThreads(1)


class BatchColorize(object):
    def __init__(self, n=150):
        self.cmap = color_map(n)

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((size[0], 3, size[1], size[2]), dtype=np.float32)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[:,0][mask] = self.cmap[label][0]
            color_image[:,1][mask] = self.cmap[label][1]
            color_image[:,2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[:,0][mask] = color_image[:,1][mask] = color_image[:,2][mask] = 255

        return color_image
    
class BatchDeColorize(object):
    def __init__(self, n=40):
        self.cmap = color_map(n)

    def __call__(self, rgb_image):
        size = rgb_image.shape
        gray_image = np.zeros((size[0], size[2], size[3]), dtype=np.float32) - 1
        

        for label in range(0, len(self.cmap)):
            tmp = np.zeros_like(rgb_image)
            tmp[:,0] = self.cmap[label][0]
            tmp[:,1] = self.cmap[label][1]
            tmp[:,2] = self.cmap[label][2]
            mask = (tmp == rgb_image)
            m = np.prod(mask, 1).astype(bool)
            gray_image[m] = label            

        # handle void
        mask = (-1 == gray_image)
        gray_image[mask] = 255

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


def load_data(
    *,
    dataset_mode,
    data_dir,
    image_size,
    random_crop=True,
    random_flip=True,
    is_train=True,
    use_pose=False,
    class_cond=False,
    deterministic=False,
    batch_size=1,       
    use_canny=False,
    use_depth=False,
    use_rgb=False,
    joint_data=False,
    img_factor_train=False,
    max_class_allowed=-1,
    use_ade_colormap=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    if dataset_mode == 'cityscapes':
        all_files = _list_image_files_recursively(os.path.join(data_dir, 'leftImg8bit', 'train' if is_train else 'val'))
        labels_file = _list_image_files_recursively(os.path.join(data_dir, 'gtFine', 'train' if is_train else 'val'))
        classes = [x for x in labels_file if x.endswith('_labelIds.png')]
        instances = [x for x in labels_file if x.endswith('_instanceIds.png')]
        poses=None
    elif dataset_mode == 'ade20k':
        all_files = _list_image_files_recursively(os.path.join(data_dir, 'images', 'training' if is_train else 'validation'))
        classes = _list_image_files_recursively(os.path.join(data_dir, 'annotations', 'training' if is_train else 'validation'))
        instances = None
        poses=None
    elif dataset_mode == 'celeba':
        all_files = _list_image_files_recursively(os.path.join(data_dir, 'images', 'training' if is_train else 'validation'))
        classes = _list_image_files_recursively(os.path.join(data_dir, 'annotations', 'training' if is_train else 'validation'))
        poses = _list_image_files_recursively(os.path.join(data_dir, 'lmannimages', 'training' if is_train else 'validation'))
        instances = _list_image_files_recursively(os.path.join(data_dir, 'annotations', 'training' if is_train else 'validation'))
    elif dataset_mode == 'coco':
        all_files = _list_image_files_recursively(os.path.join(data_dir, 'images', 'train2017' if is_train else 'val2017'))
        classes = _list_image_files_recursively(os.path.join(data_dir, 'annotations', 'train2017' if is_train else 'val2017'))
        poses = None
        instances = _list_image_files_recursively(os.path.join(data_dir, 'annotations', 'traininstance' if is_train else 'valinstance'))
    elif dataset_mode == 'sample':
        # For sampling during inference mode
        all_files = _list_image_files_recursively(os.path.join(data_dir, 'sample1'))
        classes = _list_image_files_recursively(os.path.join(data_dir, 'sample2'))
        poses = None
        instances = None
    else:
        raise NotImplementedError('{} not implemented'.format(dataset_mode))

    print("Len of Dataset:", len(all_files))

    dataset = ImageDataset(
        dataset_mode,
        image_size,
        all_files,
        classes=classes,
        instances=instances,
        poses=poses,
        shard=0,
        num_shards=1,
        data_dir=data_dir,
        random_crop=random_crop,
        random_flip=random_flip,
        use_pose=use_pose,
        is_train=is_train,
        use_canny=use_canny,
        use_depth=use_depth,
        use_rgb=use_rgb,
        joint_data=joint_data,
        img_factor_train=img_factor_train,
        max_class_allowed=max_class_allowed,
        use_ade_colormap=use_ade_colormap,
    )

    return dataset

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


class ImageDataset(Dataset):
    def __init__(
        self,
        dataset_mode,
        resolution,
        image_paths,
        classes=None,
        instances=None,
        poses=None,
        shard=0,
        num_shards=1,
        data_dir=None,
        random_crop=False,
        random_flip=True,
        use_pose=False,
        is_train=True,
        use_canny=False,
        use_depth=False,
        use_rgb=False,
        joint_data=False,
        img_factor_train=False,
        use_ade_colormap=False,
        max_class_allowed=-1,
    ):
        super().__init__()
        self.is_train = is_train
        self.dataset_mode = dataset_mode
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.local_instances = None if instances is None else instances[shard:][::num_shards]
        self.local_poses = None if poses is None else poses[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.use_pose = use_pose
        self.data_dir = data_dir
        self.use_canny = use_canny
        self.use_depth = use_depth
        self.use_rgb = use_rgb
        self.joint_data = joint_data
        self.img_factor_train = img_factor_train
        self.max_class_allowed = max_class_allowed
        self.use_ade_colormap = use_ade_colormap
        if poses is not None:
            self.colorize_pose = BatchColorize(68)
            csv_name = os.path.join(data_dir, 'annotations', 'training.csv' if is_train else 'validation.csv')
            # load annotations
            self.landmarks_frame = pd.read_csv(csv_name)
            self.landmarks_frame.set_index('image_name', inplace=True)

        if dataset_mode == 'ade20k':
            result_file = 'data/ade20k/ade20k_train.json' if is_train else 'data/ade20k/ade20k_validation.json'
            captions_list = json.load(open(result_file,'r'))
            self.captions = {}
            for cap in captions_list:
                if 'data8' in data_dir:
                    self.captions[cap["image_id"][0]] = cap["caption"]
                else:
                    self.captions[cap["image_id"][0].replace('data8', 'data')] = cap["caption"]
            object_list = pd.read_csv(f'{data_dir}/objectInfo150_mod.txt', delimiter='\t', delim_whitespace=False, header=None)
            scene_list = pd.read_csv(f'{data_dir}/sceneCategories.txt', delim_whitespace=True, header=None)
            self.class_dict = {}
            for idx, x in zip(object_list.loc[1:, 0], object_list.loc[1:, 4]):
                idx = int(idx)
                idx -= 1
                self.class_dict[idx] = x        
            self.class_dict[150] = 'background'
            self.scene_dict = {}
            img_dir = f'{data_dir}images/'
            for p, x in zip(scene_list.loc[:, 0],scene_list.loc[:, 1]):
                if 'train' in p:
                    idx = img_dir + 'training/' + p + '.jpg'
                else:
                    idx = img_dir + 'validation/' + p + '.jpg'
                self.scene_dict[idx] = x 
        elif dataset_mode == 'celeba':
            self.class_dict = ['background','skin', 'nose', 'eye_glasses', 'left_eye', 'right_eye', 'left_brow', 'right_brow',
                 'left_ear', 'right_ear', 'mouth', 'upper_lip', 'lower_lip', 'hair', 'hat', 'earring', 'necklace',
                  'neck', 'cloth', 'misc']
        elif dataset_mode == 'coco':
            self.class_dict = {
                            183: "unlabeled",
                            1: "person",
                            2: "bicycle",
                            3: "car",
                            4: "motorcycle",
                            5: "airplane",
                            6: "bus",
                            7: "train",
                            8: "truck",
                            9: "boat",
                            10: "traffic light",
                            11: "fire hydrant",
                            12: "street sign",
                            13: "stop sign",
                            14: "parking meter",
                            15: "bench",
                            16: "bird",
                            17: "cat",
                            18: "dog",
                            19: "horse",
                            20: "sheep",
                            21: "cow",
                            22: "elephant",
                            23: "bear",
                            24: "zebra",
                            25: "giraffe",
                            26: "hat",
                            27: "backpack",
                            28: "umbrella",
                            29: "shoe",
                            30: "eye glasses",
                            31: "handbag",
                            32: "tie",
                            33: "suitcase",
                            34: "frisbee",
                            35: "skis",
                            36: "snowboard",
                            37: "sports ball",
                            38: "kite",
                            39: "baseball bat",
                            40: "baseball glove",
                            41: "skateboard",
                            42: "surfboard",
                            43: "tennis racket",
                            44: "bottle",
                            45: "plate",
                            46: "wine glass",
                            47: "cup",
                            48: "fork",
                            49: "knife",
                            50: "spoon",
                            51: "bowl",
                            52: "banana",
                            53: "apple",
                            54: "sandwich",
                            55: "orange",
                            56: "broccoli",
                            57: "carrot",
                            58: "hot dog",
                            59: "pizza",
                            60: "donut",
                            61: "cake",
                            62: "chair",
                            63: "couch",
                            64: "potted plant",
                            65: "bed",
                            66: "mirror",
                            67: "dining table",
                            68: "window",
                            69: "desk",
                            70: "toilet",
                            71: "door",
                            72: "tv",
                            73: "laptop",
                            74: "mouse",
                            75: "remote",
                            76: "keyboard",
                            77: "cell phone",
                            78: "microwave",
                            79: "oven",
                            80: "toaster",
                            81: "sink",
                            82: "refrigerator",
                            83: "blender",
                            84: "book",
                            85: "clock",
                            86: "vase",
                            87: "scissors",
                            88: "teddy bear",
                            89: "hair drier",
                            90: "toothbrush",
                            91: "hair brush",
                            92: "banner",
                            93: "blanket",
                            94: "branch",
                            95: "bridge",
                            96: "building-other",
                            97: "bush",
                            98: "cabinet",
                            99: "cage",
                            100: "cardboard",
                            101: "carpet",
                            102: "ceiling-other",
                            103: "ceiling-tile",
                            104: "cloth",
                            105: "clothes",
                            106: "clouds",
                            107: "counter",
                            108: "cupboard",
                            109: "curtain",
                            110: "desk-stuff",
                            111: "dirt",
                            112: "door-stuff",
                            113: "fence",
                            114: "floor-marble",
                            115: "floor-other",
                            116: "floor-stone",
                            117: "floor-tile",
                            118: "floor-wood",
                            119: "flower",
                            120: "fog",
                            121: "food-other",
                            122: "fruit",
                            123: "furniture-other",
                            124: "grass",
                            125: "gravel",
                            126: "ground-other",
                            127: "hill",
                            128: "house",
                            129: "leaves",
                            130: "light",
                            131: "mat",
                            132: "metal",
                            133: "mirror-stuff",
                            134: "moss",
                            135: "mountain",
                            136: "mud",
                            137: "napkin",
                            138: "net",
                            139: "paper",
                            140: "pavement",
                            141: "pillow",
                            142: "plant-other",
                            143: "plastic",
                            144: "platform",
                            145: "playingfield",
                            146: "railing",
                            147: "railroad",
                            148: "river",
                            149: "road",
                            150: "rock",
                            151: "roof",
                            152: "rug",
                            153: "salad",
                            154: "sand",
                            155: "sea",
                            156: "shelf",
                            157: "sky-other",
                            158: "skyscraper",
                            159: "snow",
                            160: "solid-other",
                            161: "stairs",
                            162: "stone",
                            163: "straw",
                            164: "structural-other",
                            165: "table",
                            166: "tent",
                            167: "textile-other",
                            168: "towel",
                            169: "tree",
                            170: "vegetable",
                            171: "wall-brick",
                            172: "wall-concrete",
                            173: "wall-other",
                            174: "wall-panel",
                            175: "wall-stone",
                            176: "wall-tile",
                            177: "wall-wood",
                            178: "water-other",
                            179: "waterdrops",
                            180: "window-blind",
                            181: "window-other",
                            182: "wood"
                        }
            annFile = os.path.join(data_dir, 'annotations', 'captions_train2017.json' if is_train else 'captions_val2017.json')
            # create coco object and cocoRes object
            coco_ann = COCO(annFile)
            imgIds = coco_ann.getImgIds()
            annIds = coco_ann.getAnnIds(imgIds=imgIds)
            anns = coco_ann.loadAnns(annIds)
            # Initialize an empty dictionary to store the results
            self.captions = {}

            # Iterate through the list of annotations and populate the dictionary
            for annotation in anns:
                image_id = annotation['image_id']
                caption = annotation['caption']
                image_id = os.path.join(data_dir, 'images', 'train2017' if is_train else 'val2017', f'{image_id:012d}.jpg')
                if not os.path.exists(image_id):
                    image_id = os.path.join(data_dir, 'images', 'train2017' if is_train else 'val2017', f'{image_id:012d}.png')
                if image_id in self.captions:
                    self.captions[image_id].append(caption)
                else:
                    self.captions[image_id] = [caption]
            ff = open('data/coco_val_captions.txt', 'w')
            for k,v in sorted(self.captions.items()):
                ff.write(f"{v[0]}\n")
            ff.close()
            
            
    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        out_dict = {}
        class_path = self.local_classes[idx]
        with bf.BlobFile(class_path, "rb") as f:
            pil_class = Image.open(f)
            pil_class.load()
        if self.dataset_mode == 'sample':
            pil_class = pil_class.convert("RGB")
        else:
            pil_class = pil_class.convert("L")

        if self.local_instances is not None:
            instance_path = self.local_instances[idx] 
            with bf.BlobFile(instance_path, "rb") as f:
                pil_instance = Image.open(f)
                pil_instance.load()
            pil_instance = pil_instance.convert("L")
        else:
            pil_instance = None

        if self.local_poses is not None:
            
            try:
                pts = self.landmarks_frame.loc[path][3:].values
                pts = pts.astype('float').reshape(-1, 2)
            except:
                pts = np.zeros((68, 2))
            pts /= 1024.
            out_dict["lm"] = pts

            pose_path = self.local_poses[idx] 
            with bf.BlobFile(pose_path, "rb") as f:
                pil_pose = Image.open(f)
                pil_pose.load()
            pil_pose = pil_pose.convert("L")
        else:
            pil_pose = None
            out_dict["lm"] = np.zeros((68, 2))


        if self.dataset_mode == 'cityscapes':
            arr_image, arr_class, arr_instance, arr_pose = resize_arr([pil_image, pil_class, pil_instance, pil_pose], self.resolution)
        else:
            if self.is_train:
                if self.random_crop:
                    arr_image, arr_class, arr_instance, arr_pose = random_crop_arr([pil_image, pil_class, pil_instance, pil_pose], self.resolution)
                else:
                    arr_image, arr_class, arr_instance, arr_pose = resize_arr([pil_image, pil_class, pil_instance, pil_pose], self.resolution, keep_aspect=False)
            else:
                arr_image, arr_class, arr_instance, arr_pose = resize_arr([pil_image, pil_class, pil_instance, pil_pose], self.resolution, keep_aspect=False)

        if self.random_flip and random.random() < 0.5:
            arr_image = arr_image[:, ::-1].copy()
            arr_class = arr_class[:, ::-1].copy()
            arr_instance = arr_instance[:, ::-1].copy() if arr_instance is not None else None
            arr_pose = arr_pose[:, ::-1].copy() if arr_pose is not None else None
            try:
                tmpt = self.landmarks_frame.loc[path][3:].values
                pts[:,0] = 1 - pts[:,0]
            except:
                pts = np.zeros((68, 2))
            
            out_dict["lm"] = pts
        arr_image = arr_image.astype(np.float32) / 127.5 - 1
        num_classes = np.amax(arr_class)
        if self.max_class_allowed != -1:
            if num_classes > self.max_class_allowed:
                idx = np.random.randint(0, len(self)-1)
                sample = self[idx]
                return sample
        colorize = BatchColorize(num_classes)
        decolorize = BatchDeColorize(num_classes)

        out_dict['path'] = path
        out_dict['label_ori'] = arr_class.copy()

        if self.dataset_mode == 'ade20k':
            arr_class = arr_class - 1
       
        tmp = arr_class
        if self.dataset_mode == 'ade20k':
            tmp[tmp == 255] = 150
        if self.dataset_mode == 'coco':
            tmp[tmp == 255] = 182
        out_dict["parts"] = tmp[None, ]
        if self.dataset_mode == 'sample':
            pass
        else:
            if self.use_ade_colormap:
                arr_class = np.vectorize(label_mapping.get)(arr_class)
                arr_class = colorize_ade(arr_class[None, ])
            else:    
                arr_class = colorize(arr_class[None, ])

            arr_class = arr_class.squeeze(0).transpose([1, 2, 0])

        if arr_pose is not None:
            arr_pose = colorize(arr_pose[None, ])
            arr_pose = arr_pose.squeeze(0).transpose([1, 2, 0])

            arr_pose = arr_pose.astype(np.float32) / 127.5 - 1
            
            out_dict['pose'] = arr_pose 

        arr_class = arr_class.astype(np.float32) / 127.5 - 1
        
        out_dict['label'] = arr_class 

        if self.use_canny:
            arr_canny = cv2.Canny(((arr_image+1)*127.5).astype(np.uint8), 100, 200)
            arr_canny = np.stack([arr_canny, arr_canny, arr_canny], -1)
            arr_canny = arr_canny.astype(np.float32) / 127.5 - 1
            out_dict['label'] = arr_canny
        
        if self.use_depth:
            out_dict['label'] = arr_image

        if arr_instance is not None:
            out_dict['instance'] = arr_instance[None, ]
        if arr_pose is not None and self.use_pose:
            out_dict["image"] = np.concatenate([arr_image, arr_class, arr_pose],-1)
        else:
            if self.joint_data:
                if self.use_canny:
                    out_dict["image"] = np.concatenate([arr_image, arr_canny],-1)
                elif self.use_depth:
                    out_dict["image"] = np.concatenate([arr_image, arr_depth],-1)
                else:
                    out_dict["image"] = np.concatenate([arr_image, arr_class],-1)
            else:
                 #np.concatenate([arr_image, arr_class],-1)
                if self.use_canny:
                    out_dict["image"] = arr_canny
                elif self.use_depth:
                    out_dict["image"] = arr_depth
                elif self.use_rgb:
                    out_dict["image"] = arr_image
                else:
                    out_dict["image"] = arr_class
        if self.img_factor_train:
            out_dict["rgb"] = arr_image       
        

        classes_str = ""
        if self.dataset_mode == 'ade20k':
            scene = self.scene_dict[path]
            num = 1
            for i in range(num_classes):
                if i in tmp:
                    classes_str += self.class_dict[i].split(',')[0] + ";"
                    if num >= 76:
                        break
                    num += 1
            classes_str += scene #+ "."
        elif self.dataset_mode == 'celeba' or self.dataset_mode == 'coco':
            num = 1
            for i in range(num_classes):
                if i in tmp and i != 255:
                    if self.dataset_mode == 'coco':
                        classes_str += self.class_dict[i+1] + ";"
                        if len(classes_str) >= 77:
                            break
                        num += 1
                    else:
                        classes_str += self.class_dict[i] + ";"
        out_dict["class_caption"] = classes_str
        
        caps = self.captions[path]
        if isinstance(caps, list):
            if self.is_train:
                caps = random.choice(caps)
            else:
                caps = caps[0]
        out_dict["caption"] = caps
        
        return out_dict


def resize_arr(pil_list, image_size, keep_aspect=True):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    pil_image, pil_class, pil_instance, pil_pose = pil_list

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
    
    if pil_pose is not None:
        pil_pose = pil_pose.resize(pil_image.size, resample=Image.NEAREST)

    arr_image = np.array(pil_image)
    arr_class = np.array(pil_class)
    arr_instance = np.array(pil_instance) if pil_instance is not None else None
    arr_pose = np.array(pil_pose) if pil_pose is not None else None
    return arr_image, arr_class, arr_instance, arr_pose


def center_crop_arr(pil_list, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    pil_image, pil_class, pil_instance, pil_pose = pil_list

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
    
    if pil_pose is not None:
        pil_pose = pil_pose.resize(pil_image.size, resample=Image.NEAREST)

    arr_image = np.array(pil_image)
    arr_class = np.array(pil_class)
    arr_instance = np.array(pil_instance) if pil_instance is not None else None
    arr_pose = np.array(pil_pose) if pil_pose is not None else None
    crop_y = (arr_image.shape[0] - image_size) // 2
    crop_x = (arr_image.shape[1] - image_size) // 2
    return arr_image[crop_y : crop_y + image_size, crop_x : crop_x + image_size],\
           arr_class[crop_y: crop_y + image_size, crop_x: crop_x + image_size],\
           arr_instance[crop_y : crop_y + image_size, crop_x : crop_x + image_size] if arr_instance is not None else None,\
           arr_pose[crop_y : crop_y + image_size, crop_x : crop_x + image_size] if arr_pose is not None else None


def random_crop_arr(pil_list, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    pil_image, pil_class, pil_instance, pil_pose = pil_list

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
    if pil_pose is not None:
        pil_pose = pil_pose.resize(pil_image.size, resample=Image.NEAREST)

    arr_image = np.array(pil_image)
    arr_class = np.array(pil_class)
    arr_instance = np.array(pil_instance) if pil_instance is not None else None
    arr_pose = np.array(pil_pose) if pil_pose is not None else None
    crop_y = random.randrange(arr_image.shape[0] - image_size + 1)
    crop_x = random.randrange(arr_image.shape[1] - image_size + 1)
    return arr_image[crop_y : crop_y + image_size, crop_x : crop_x + image_size],\
           arr_class[crop_y: crop_y + image_size, crop_x: crop_x + image_size],\
           arr_instance[crop_y : crop_y + image_size, crop_x : crop_x + image_size] if arr_instance is not None else None,\
           arr_pose[crop_y : crop_y + image_size, crop_x : crop_x + image_size] if arr_pose is not None else None
