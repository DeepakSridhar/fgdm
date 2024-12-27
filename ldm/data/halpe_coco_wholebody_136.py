# -----------------------------------------------------
# Modified by Deepak Sridhar
# -----------------------------------------------------

"""Halpe Full-Body plus coco wholebody (136 points) Human keypoint dataset. Shanghai Jiao Tong University"""
import os

import numpy as np

from pycocotools.coco import COCO
import pickle as pk
import cv2
cv2.setNumThreads(1)
import copy
import random
import math
import blobfile as bf
from PIL import Image

from ldm.data.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy

from ldm.data.custom import CustomDataset
from torch.utils.data import DataLoader, Dataset

class BatchColorize(object):
    def __init__(self, n=150):
        self.cmap = color_map(n)
        # self.cmap = torch.from_numpy(self.cmap[:n])

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
        # self.cmap = torch.from_numpy(self.cmap[:n])

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


class Halpe_coco_wholebody_136(CustomDataset):
    """ Halpe Full-Body plus coco wholebody (136 points) Person dataset.

    Parameters
    ----------
    train: bool, default is True
        If true, will set as training mode.
    skip_empty: bool, default is False
        Whether skip entire image if no valid label is found. Use `False` if this dataset is
        for validation to avoid COCO metric error.
    dpg: bool, default is False
        If true, will activate `dpg` for data augmentation.
    """
    CLASSES = ['person']
    EVAL_JOINTS = list(range(136))
    num_joints = 136
    joint_colors = color_map(num_joints+1).tolist()[1:]
    CustomDataset.lower_body_ids = (11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25)
    """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
    joint_pairs =  [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], #17 body keypoints
        [20, 21], [22, 23], [24, 25], [26, 42], [27, 41], [28, 40], [29, 39], [30, 38], 
        [31, 37], [32, 36], [33, 35], [43, 52], [44, 51], [45, 50],[46, 49], [47, 48], 
        [62, 71], [63, 70], [64, 69], [65, 68], [66, 73], [67, 72], [57, 61], [58, 60],
        [74, 80], [75, 79], [76, 78], [87, 89], [93, 91], [86, 90], [85, 81], [84, 82],
        [94, 115], [95, 116], [96, 117], [97, 118], [98, 119], [99, 120], [100, 121],
        [101, 122], [102, 123], [103, 124], [104, 125], [105, 126], [106, 127], [107, 128],
        [108, 129], [109, 130], [110, 131], [111, 132], [112, 133], [113, 134], [114, 135]]
                
    vis_pairs = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),# Body
        (17, 18), (18, 19), (19, 11), (19, 12),
        (11, 13), (12, 14), (13, 15), (14, 16),
        # (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25),# Halpe Foot
        (20, 22), (25, 23), (21, 22), (24, 25), (15, 22), (16, 25),# Coco Foot
        (26, 27),(27, 28),(28, 29),(29, 30),(30, 31),(31, 32),(32, 33),(33, 34),(34, 35),(35, 36),(36, 37),(37, 38),#Face
        (38, 39),(39, 40),(40, 41),(41, 42),(43, 44),(44, 45),(45, 46),(46, 47),(48, 49),(49, 50),(50, 51),(51, 52),#Face
        (53, 54),(54, 55),(55, 56),(57, 58),(58, 59),(59, 60),(60, 61),(62, 63),(63, 64),(64, 65),(65, 66),(66, 67),#Face
        (68, 69),(69, 70),(70, 71),(71, 72),(72, 73),(74, 75),(75, 76),(76, 77),(77, 78),(78, 79),(79, 80),(80, 81),#Face
        (81, 82),(82, 83),(83, 84),(84, 85),(85, 86),(86, 87),(87, 88),(88, 89),(89, 90),(90, 91),(91, 92),(92, 93),#Face
        (94,95),(95,96),(96,97),(97,98),(94,99),(99,100),(100,101),(101,102),(94,103),(103,104),(104,105),#LeftHand
        (105,106),(94,107),(107,108),(108,109),(109,110),(94,111),(111,112),(112,113),(113,114),#LeftHand
        (115,116),(116,117),(117,118),(118,119),(115,120),(120,121),(121,122),(122,123),(115,124),(124,125),#RightHand
        (125,126),(126,127),(115,128),(128,129),(129,130),(130,131),(115,132),(132,133),(133,134),(134,135)#RightHand
    ]

    def _lazy_load_ann_file_2(self):
        if os.path.exists(self._ann_file_2 + '.pkl') and self._lazy_import:
            print('Lazy load json...')
            with open(self._ann_file_2 + '.pkl', 'rb') as fid:
                return pk.load(fid)
        else:
            _database = COCO(self._ann_file_2)
            if os.access(self._ann_file_2 + '.pkl', os.W_OK):
                with open(self._ann_file_2 + '.pkl', 'wb') as fid:
                    pk.dump(_database, fid, pk.HIGHEST_PROTOCOL)
            return _database

    def _load_jsons(self):
        """Load all image paths and labels from JSON annotation files into buffer."""
        items = []
        labels = []

        _coco = self._lazy_load_ann_file_2()

        self.json_id_to_contiguous = {
            v: k for k, v in enumerate(_coco.getCatIds())}

        # iterate through the annotations
        image_ids = sorted(_coco.getImgIds())  
        
        for entry in _coco.loadImgs(image_ids):
            dirname, filename = entry['coco_url'].split('/')[-2:]
            abs_path = os.path.join(self._root_2, 'images', dirname, filename)
            if not os.path.exists(abs_path):
                raise IOError('Image: {} not exists.'.format(abs_path))
            if self._use_pose:
                label = self._check_load_keypoints_2(_coco, entry)
                if not label:
                    continue
                
                items.append(abs_path)
                labels.append(label)
            else:
                items.append(abs_path)
                labels.append(abs_path)
        
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
        annFile = os.path.join(self._root_2, 'annotations', 'captions_train2017.json' if self._train else 'captions_val2017.json')
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
            image_id = os.path.join(self._root_2, 'images', 'train2017' if self._train else 'val2017', f'{image_id:012d}.jpg')
            # print(image_id)
            if not os.path.exists(image_id):
                image_id = os.path.join(self._root_2, 'images', 'train2017' if self._train else 'val2017', f'{image_id:012d}.png')
            if image_id in self.captions:
                self.captions[image_id].append(caption)
            else:
                self.captions[image_id] = [caption]           


        return items, labels

    def _check_load_keypoints_2(self, coco, entry):
        """Check and load ground-truth keypoints for coco wholebody"""
        ann_ids = coco.getAnnIds(imgIds=entry['id'], iscrowd=False)
        objs = coco.loadAnns(ann_ids)
        # check valid bboxes
        valid_objs = []
        width = entry['width']
        height = entry['height']

        for obj in objs:
            if 'foot_kpts' in obj and 'face_kpts' in obj and 'lefthand_kpts' in obj and 'righthand_kpts' in obj:
                obj['keypoints'].extend([0] * 9)    # coco wholebody has only 133 kpts
                obj['keypoints'].extend(obj['foot_kpts'])
                obj['keypoints'].extend(obj['face_kpts'])
                obj['keypoints'].extend(obj['lefthand_kpts'])
                obj['keypoints'].extend(obj['righthand_kpts'])
            contiguous_cid = self.json_id_to_contiguous[obj['category_id']]
            if contiguous_cid >= self.num_class:
                # not class of interest
                continue
            if max(obj['keypoints']) == 0:
                continue
            # convert from (x, y, w, h) to (xmin, ymin, xmax, ymax) and clip bound
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(obj['bbox']), width, height)
            # require non-zero box area
            if (xmax - xmin) * (ymax - ymin) <= 0 or xmax <= xmin or ymax <= ymin:
                continue
            if 'num_keypoints' in obj and obj['num_keypoints'] == 0:
                continue
            # joints 3d: (num_joints, 3, 2); 3 is for x, y, z; 2 is for position, visibility
            joints_3d = np.zeros((self.num_joints, 3, 2), dtype=np.float32)
            for i in range(self.num_joints):
                joints_3d[i, 0, 0] = obj['keypoints'][i * 3 + 0]
                joints_3d[i, 1, 0] = obj['keypoints'][i * 3 + 1]
                if i == 18:
                    if np.sum(joints_3d[5, :, 0]) != 0 and np.sum(joints_3d[6, :, 0]):
                        joints_3d[i, 0, 0] = (joints_3d[5, 0, 0] + joints_3d[6, 0, 0]) / 2.
                        joints_3d[i, 1, 0] = (joints_3d[5, 1, 0] + joints_3d[6, 1, 0]) / 2.
                if i == 19:
                    if np.sum(joints_3d[11, :, 0]) != 0 and np.sum(joints_3d[12, :, 0]):
                        joints_3d[i, 0, 0] = (joints_3d[11, 0, 0] + joints_3d[12, 0, 0]) / 2.
                        joints_3d[i, 1, 0] = (joints_3d[11, 1, 0] + joints_3d[12, 1, 0]) / 2.

                if obj['keypoints'][i * 3 + 2] >= 0.35:
                    visible = 1
                else:
                    visible = 0
                joints_3d[i, :2, 1] = visible

            if np.sum(joints_3d[:, 0, 1]) < 1:
                # no visible keypoint
                continue

            if self._check_centers and self._train:
                bbox_center, bbox_area = self._get_box_center_area((xmin, ymin, xmax, ymax))
                kp_center, num_vis = self._get_keypoints_center_count(joints_3d)
                ks = np.exp(-2 * np.sum(np.square(bbox_center - kp_center)) / bbox_area)
                if (num_vis / 80.0 + 47 / 80.0) > ks:
                    continue

            valid_objs.append({
                'bbox': (xmin, ymin, xmax, ymax),
                'width': width,
                'height': height,
                'joints_3d': joints_3d
            })

        if not valid_objs:
            if not self._skip_empty:
                # dummy invalid labels if no valid objects are found
                valid_objs.append({
                    'bbox': np.array([-1, -1, 0, 0]),
                    'width': width,
                    'height': height,
                    'joints_3d': np.zeros((self.num_joints, 2, 2), dtype=np.float32)
                })
        return valid_objs

    def _get_box_center_area(self, bbox):
        """Get bbox center"""
        c = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        return c, area

    def _get_keypoints_center_count(self, keypoints):
        """Get geometric center of all keypoints"""
        keypoint_x = np.sum(keypoints[:, 0, 0] * (keypoints[:, 0, 1] > 0))
        keypoint_y = np.sum(keypoints[:, 1, 0] * (keypoints[:, 1, 1] > 0))
        num = float(np.sum(keypoints[:, 0, 1]))
        return np.array([keypoint_x / num, keypoint_y / num]), num
    
    def __getitem__(self, idx):
        # get image id
        if type(self._items[idx]) == dict:
            img_path = self._items[idx]['path']
            # img_id = self._items[idx]['id']
        else:
            img_path = self._items[idx]
            
        out_dict = {}
        # load ground truth, including bbox, keypoints, image size
        
        with bf.BlobFile(img_path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        img = np.array(pil_image)
        class_path = img_path.replace('/images/', '/annotations/').replace('.jpg','.png')
        with bf.BlobFile(class_path, "rb") as f:
            pil_class = Image.open(f)
            pil_class.load()
        if self.dataset_mode == 'sample':
            pil_class = pil_class.convert("RGB")
        else:
            pil_class = pil_class.convert("L")
        arr_class = np.array(pil_class)
        image_width = img.shape[1] #640  # Set your desired image width
        image_height = img.shape[0] #480  # Set your desired image height

        if self._use_pose:
            label = copy.deepcopy(self._labels[idx])
            arr_pose = np.zeros((image_height, image_width, 3), dtype=np.uint8) #copy.deepcopy(img) #
            # print(image_height, image_width)
            for lab in label:
                #     print(lab.shape)
                joints = lab['joints_3d'][:,:,0] #
                for pair, color in zip(self.vis_pairs, self.joint_colors):
                    joint1 = (int(joints[pair[0]][0]*1), int(joints[pair[0]][1]*1))
                    joint2 = (int(joints[pair[1]][0]*1), int(joints[pair[1]][1]*1))
                    # print(joint1, joint2)
                    if max(joint1) == 0 or max(joint2) == 0:
                        continue
                    
                    cv2.line(arr_pose, joint1, joint2, color, 3)
                    cv2.circle(arr_pose, joint1, 3, color, -1)
                    cv2.circle(arr_pose, joint2, 3, color, -1)
            
        else:
            arr_pose = None
        
        arr_instance = None
        if self._train:
            if self.random_crop:
                arr_image, arr_class, arr_instance, arr_pose = random_crop_arr([img, arr_class, arr_instance, arr_pose], self.resolution)
            else:
                arr_image, arr_class, arr_instance, arr_pose = center_crop_arr([img, arr_class, arr_instance, arr_pose], self.resolution)
        else:
            arr_image, arr_class, arr_instance, arr_pose = resize_arr([img, arr_class, arr_instance, arr_pose], self.resolution, keep_aspect=False)

        if self.random_flip and random.random() < 0.5:
            arr_image = arr_image[:, ::-1].copy()
            arr_class = arr_class[:, ::-1].copy()
            arr_instance = arr_instance[:, ::-1].copy() if arr_instance is not None else None
            arr_pose = arr_pose[:, ::-1].copy() if arr_pose is not None else None
            try:
                pts[:,0] = 1 - pts[:,0]
            except:
                pts = np.zeros((68, 2))
            
        arr_image = arr_image.astype(np.float32) / 127.5 - 1
        num_classes = np.amax(arr_class)
        colorize = BatchColorize(num_classes)
        # decolorize = BatchDeColorize(num_classes)

        out_dict['path'] = img_path
        out_dict['label_ori'] = arr_class.copy()

        tmp = arr_class
        tmp[tmp == 255] = 182
            
        out_dict["parts"] = tmp[None, ]
        if self.dataset_mode == 'sample':
            pass
        else:
            person_mask = np.expand_dims(arr_class == 0, -1)
            arr_class = colorize(arr_class[None, ])
            
            arr_class = arr_class.squeeze(0).transpose([1, 2, 0])

        if arr_pose is not None:
            arr_pose = arr_pose.astype(np.float32) / 127.5 - 1
            
            out_dict['pose'] = arr_pose 
        
        arr_class = arr_class.astype(np.float32) / 127.5 - 1
        
        out_dict['label'] = arr_class 

        if self.use_canny:
            arr_canny = cv2.Canny(((arr_image+1)*127.5).astype(np.uint8), 100, 200)
            arr_canny = np.stack([arr_canny, arr_canny, arr_canny], -1)
            out_dict['label'] = arr_canny.astype(np.float32) / 127.5 - 1

        if arr_instance is not None:
            out_dict['instance'] = arr_instance[None, ]
        if arr_pose is not None and self._use_pose:
            if self.pose_only:
                out_dict["image"] = arr_pose
            else:
                out_dict["image"] = np.concatenate([arr_image, arr_class, arr_pose],-1)
        else:
            if self.use_canny:
                out_dict["image"] = np.concatenate([arr_image, arr_canny],-1)
            else:
                out_dict["image"] = np.concatenate([arr_image, arr_class],-1)
        
        if self.use_float16:
            out_dict["image"] = out_dict["image"].astype(np.float16)


        classes_str = ""
        num = 1
        for i in range(num_classes):
            if i in tmp and i != 255:
                classes_str += self.class_dict[i+1] + ";"
                if len(classes_str) >= 77:
                    break
                num += 1
        out_dict["caption2"] = classes_str

        
        caps = self.captions[img_path]
        if isinstance(caps, list):
            if self._train:
                caps = random.choice(caps)
            else:
                caps = caps[0]
        out_dict["caption"] = caps
        return out_dict

def resize_arr(arr_list, image_size, keep_aspect=True):
    arr_image, arr_class, arr_instance, arr_pose = arr_list

    while min(arr_image.shape[:2]) >= 2 * image_size:
        arr_image = cv2.resize(arr_image, (arr_image.shape[1] // 2, arr_image.shape[0] // 2), interpolation=cv2.INTER_LINEAR)

    if keep_aspect:
        scale = image_size / min(arr_image.shape[:2])
        arr_image = cv2.resize(arr_image, (round(arr_image.shape[1] * scale), round(arr_image.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)
    else:
        arr_image = cv2.resize(arr_image, (image_size, image_size), interpolation=cv2.INTER_CUBIC)

    arr_class = cv2.resize(arr_class, (arr_image.shape[1], arr_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    if arr_instance is not None:
        arr_instance = cv2.resize(arr_instance, (arr_image.shape[1], arr_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    if arr_pose is not None:
        arr_pose = cv2.resize(arr_pose, (arr_image.shape[1], arr_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    return arr_image, arr_class, arr_instance, arr_pose

def center_crop_arr(arr_list, image_size):
    arr_image, arr_class, arr_instance, arr_pose = arr_list

    while min(arr_image.shape[:2]) >= 2 * image_size:
        arr_image = cv2.resize(arr_image, (arr_image.shape[1] // 2, arr_image.shape[0] // 2), interpolation=cv2.INTER_LINEAR)

    scale = image_size / min(arr_image.shape[:2])
    arr_image = cv2.resize(arr_image, (round(arr_image.shape[1] * scale), round(arr_image.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)

    arr_class = cv2.resize(arr_class, (arr_image.shape[1], arr_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    if arr_instance is not None:
        arr_instance = cv2.resize(arr_instance, (arr_image.shape[1], arr_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    if arr_pose is not None:
        arr_pose = cv2.resize(arr_pose, (arr_image.shape[1], arr_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    crop_y = (arr_image.shape[0] - image_size) // 2
    crop_x = (arr_image.shape[1] - image_size) // 2

    cropped_arr_image = arr_image[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    cropped_arr_class = arr_class[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    
    cropped_arr_instance = None
    if arr_instance is not None:
        cropped_arr_instance = arr_instance[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    
    cropped_arr_pose = None
    if arr_pose is not None:
        cropped_arr_pose = arr_pose[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

    return cropped_arr_image, cropped_arr_class, cropped_arr_instance, cropped_arr_pose

def random_crop_arr(arr_list, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    arr_image, arr_class, arr_instance, arr_pose = arr_list

    while min(arr_image.shape[:2]) >= 2 * smaller_dim_size:
        arr_image = cv2.resize(arr_image, (arr_image.shape[1] // 2, arr_image.shape[0] // 2), interpolation=cv2.INTER_LINEAR)

    scale = smaller_dim_size / min(arr_image.shape[:2])
    arr_image = cv2.resize(arr_image, (round(arr_image.shape[1] * scale), round(arr_image.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)

    arr_class = cv2.resize(arr_class, (arr_image.shape[1], arr_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    if arr_instance is not None:
        arr_instance = cv2.resize(arr_instance, (arr_image.shape[1], arr_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    if arr_pose is not None:
        arr_pose = cv2.resize(arr_pose, (arr_image.shape[1], arr_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    crop_y = random.randrange(arr_image.shape[0] - image_size + 1)
    crop_x = random.randrange(arr_image.shape[1] - image_size + 1)
    
    cropped_arr_image = arr_image[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    cropped_arr_class = arr_class[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    
    cropped_arr_instance = None
    if arr_instance is not None:
        cropped_arr_instance = arr_instance[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    
    cropped_arr_pose = None
    if arr_pose is not None:
        cropped_arr_pose = arr_pose[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

    return cropped_arr_image, cropped_arr_class, cropped_arr_instance, cropped_arr_pose


if __name__ == '__main__':
    from omegaconf import OmegaConf

    config = OmegaConf.load("configs/latent-diffusion/config.yaml")
    os.makedirs('tmp2', exist_ok=True)
    kwargs = {'cfg': config.data.params.validation.params, 
              'label_captions': True,
              'person_only': True}
    dataset = Halpe_coco_wholebody_136(**kwargs)
    dataloader = DataLoader(dataset, batch_size=1)

    for idx, item in enumerate(dataloader):
        
        if idx > 20:
            break
