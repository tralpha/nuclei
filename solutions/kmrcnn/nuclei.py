"""
Mask R-CNN
Configurations and data loading code for Nuclei from DSB2018.

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""
import os
import time
import numpy as np
import pandas as pd

from config import Config
import utils
import model as modellib
import skimage
import copy
from sklearn.model_selection import StratifiedShuffleSplit
from IPython.core.debugger import set_trace

############################################################
#  Configurations
############################################################


class NucleiConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "nuclei"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 nuclei

    # Image Dimensions
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 512

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 34 // (IMAGES_PER_GPU * GPU_COUNT)

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 34 // (IMAGES_PER_GPU * GPU_COUNT)

    # MEAN_PIXEL = [43.53287505, 39.56061986, 48.22454996, 255.]
    MEAN_PIXEL = [44.57284587, 40.71265898, 48.6901747]

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100



############################################################
#  Dataset
############################################################


class NucleiDataset(utils.Dataset):
    def load_nuclei(self,
                    dataset_dir="../../input",
                    subset="train",
                    image_ids=None):
        """Load a subset of the nuclei dataset.
        dataset_dir: The root directory of the nuclei dataset.
        subset: What to load (train, test)
        image_ids: load these image_ids instead of walking the repo
        """
        # Add Classes
        # assert image_ids and subset == "train"
        self.add_class("nuclei", 1, "nu")

        image_dir = "{}/stage1_{}".format(dataset_dir, subset)

        if image_ids is not None:
            assert subset == "train", "Build train & val from train set"
            image_ids = image_ids
        else:
            image_ids = next(os.walk(image_dir))[1]
        # set_trace()
        # Add Images
        self.real_to_id = {}
        for idx, i in enumerate(image_ids):
            i_path = os.path.join(image_dir, i, 'images', i + '.png')
            m_path = os.path.join(image_dir, i, 'masks')
            self.add_image("nuclei", image_id=i, path=i_path, m_path=m_path)
            self.real_to_id[i] = int(idx)

    def load_image(self, image_id, remove_alpha=True):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        if isinstance(image_id, str):
            image_id = self.real_to_id[image_id]
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if remove_alpha == True and image.shape[-1] != 3:
            # print("Image {} has a shape of {}".format(image_id, image.shape))
            if image.shape[-1] == 4:
                image = image[:, :, :3]
        if image.ndim != 3:
            print("Image dimension not 3")
            image = skimage.color.gray2rgb(image)
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different dataset use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        if isinstance(image_id, str):
            image_id = self.real_to_id[image_id]
        image = skimage.io.imread(self.image_info[image_id]['path'])
        mask_path = os.path.join(self.image_info[image_id]['m_path'])
        instance_masks = []
        class_ids = []
        for mask_file in next(os.walk(mask_path))[2]:
            m = skimage.io.imread(os.path.join(mask_path, mask_file))
            instance_masks.append(m)
            class_ids.append(1)
            if m.shape[0] != image.shape[0] or m.shape[1] != image.shape[1]:
                print("Mask and image have different shape for {}"
                      .format(image_id))
            if m.max() < 1:
                print("Mask {} has a zero area")
        mask = np.stack(instance_masks, axis=2)
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids

    def split_dataset(self, val_size):
        """
        Loads validation set from Allen's `classes.csv` file
        Arguments:
            val_size: the size of the validation set
        returns:
            dataset_val: the Validation dataset directly in the NucleiDataset format.
            It should contain at least one image from the different experiment conditions.
        """
        cl_df = pd.read_csv('classes.csv')
        cl_df['im_type'] = cl_df.foreground.str.cat(cl_df.background)
        cl_df['filename'] = cl_df['filename'].apply(lambda x: x[:-4])
        classes_count = cl_df.groupby(['im_type']).count()
        # Obtain only from train set
        train_ids = list(self.real_to_id)
        # train_ids = [im + '.png' for im in self.real_to_id.keys()]
        train_classes = cl_df.loc[np.where(
            np.in1d(cl_df['filename'], train_ids))[0]]
        test_classes = cl_df.loc[np.where(~np.in1d(cl_df['filename'],
                                                   train_ids))[0]]
        train_count = train_classes.groupby(['im_type']).count()
        test_count = test_classes.groupby(['im_type']).count()
        X, y = train_classes['filename'], train_classes['im_type']
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=val_size, random_state=0)
        for train_index, val_index in sss.split(X, y):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        # train_classes = cl_df.loc[np.where(cl_df == train_ids)]
        return X_train, X_val

    def extract_train_val(self):
        """Extracts the train and validation set images from the dataset

        Returns:
        dataset_train: The train dataset which is a NucleiDataset file
        dataset_val: The val dataset which is a NucleiDataset file
        """
        train, val = self.split_dataset(0.05)
        # set_trace()
        # val = [
        #     "4dbbb275960ab9e4ec2c66c8d3000f7c70c8dce5112df591b95db84e25efa6e9",
        #     "12aeefb1b522b283819b12e4cfaf6b13c1264c0aadac3412b4edd2ace304cb40",
        #     "3f9fc8e63f87e8a56d3eaef7db26f1b6db874d19f12abd5a752821b78d47661e",
        #     "a90cad45551d62c5cfa89517df8eb5e8f2f87f1a6e6678e606907afcbad91731"
        # ]
        dataset_train = NucleiDataset()
        dataset_val = NucleiDataset()
        dataset_train.load_nuclei(image_ids=train)
        dataset_val.load_nuclei(image_ids=val)
        # dataset_val = NucleiDataset().load_nuclei(image_ids=val)
        # set_trace()
        return dataset_train, dataset_val
        # dataset_train = copy.deepcopy(self)
        # dataset_val = copy.deepcopy(self)
        # datasets = {"train": [train, dataset_train], "val": [val, dataset_val]}
        # real_ids = list(self.real_to_id.keys())
        # for d in datasets:
        #     d_bool = np.in1d([im + '.png' for im in real_ids], datasets[d][0])
        #     datasets[d][1].image_info = []
        #     datasets[d][1]._image_ids = []
        #     datasets[d][1].real_to_id = {}
        #     for idx, i in enumerate(d_bool):
        #         if i:
        #             real_id = real_ids[idx]
        #             datasets[d][1].image_ids.append(idx)
        #             img_info = {
        #                 "source": "nuclei_" + d,
        #                 "id": real_id,
        #                 "path": self.image_info[idx]['path'],
        #                 "m_path": self.image_info[idx]['m_path']
        #             }
        #             datasets[d][1].image_info.append(img_info)
        #             datasets[d][1].real_to_id[real_id] = i
        # return datasets
