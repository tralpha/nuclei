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
from scipy import ndimage

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
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 nuclei

    # Image Dimensions
    IMAGE_MIN_DIM = 1600
    IMAGE_MAX_DIM = 2048

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

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
    MINI_MASK_SHAPE = (28, 28)

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 400

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 400


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
        # Data structure to hold read_id to index_id mapping
        self.real_to_id = {}
        # Data structure to hold the double nuclei masks
        self.dn_masks = {}
        bad_masks = open("bad_masks", "r")
        b_lines = bad_masks.readlines()
        self.b_masks = (m for m in b_lines)
        for b_mask in self.b_masks:
            _, image_id, _, mask_id = b_mask.split("\\")
            mask_id = mask_id[:-5]
            self.dn_masks[mask_id] = image_id
        # self.issues = {
        #     'holes': [],
        #     'nuclei_tgt': [{
        #         "ef3ef194e5657fda708ecbd3eb6530286ed2ba23c88efb9f1715298975c73548":
        #         "9d2eb605a9e1b87b213cf0d2a366e461049cac5942db4b1a40a967dd54417792"
        #     },{} ]
        # }
        for idx, i in enumerate(image_ids):
            i_path = os.path.join(image_dir, i, 'images', i + '.png')
            m_path = os.path.join(image_dir, i, 'masks')
            self.add_image("nuclei", image_id=i, path=i_path, m_path=m_path)
            self.real_to_id[i] = int(idx)
        # set_trace()

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

    def separate_masks(self, mask_path, mask_file):
        """
        Function takes a mask file, and separates the masks contained in the 
        image
        Arguments:
            mask_path: path to the file which contains the mask.
            mask_file: file which contains the mask
        Returns:
            masks: A list of numpy arrays which contain each separated mask
        """
        mask = skimage.io.imread(os.path.join(mask_path + '_1', mask_file))
        # set_trace()
        labels, nlabels = ndimage.label(mask)
        masks = []
        class_ids = []
        for label in range(1, nlabels + 1):
            m = np.where(labels == label, 1, 0)
            m = np.where(m == 1, 255, 0)
            masks.append(m.astype('uint8'))
            class_ids.append(1)
        # set_trace()
        return masks, class_ids

    def load_mask(self, image_id, mask_id=None, m_correction=True):
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
        mask_path = os.path.join(self.image_info[image_id]['m_path'])
        if mask_id:
            if m_correction:
                mask, class_id = self.separate_masks(mask_path, mask_id)
                # print(mask[1].shape)
                # set_trace()
                masks = np.stack(mask, axis=2)
                class_id = np.array(class_id)
            else:
                m_path = os.path.join(mask_path, mask_id)
                mask = skimage.io.imread(m_path)
                masks = mask[:, :, None]
                class_id = np.array([1], dtype=np.int32)
            return masks, class_id
        image = skimage.io.imread(self.image_info[image_id]['path'])
        instance_masks = []
        class_ids = []
        for mask_file in next(os.walk(mask_path))[2]:
            if mask_file[:-4] in self.dn_masks and m_correction:
                # Try to transform the double nuclei mask into multiple masks
                print("Found a double nuclei mask")
                print(mask_file)
                m, c_id = self.separate_masks(mask_path, mask_file)
                # set_trace()
            else:
                m = [skimage.io.imread(os.path.join(mask_path, mask_file))]
                c_id = [1]
            instance_masks.extend(m)
            class_ids.extend(c_id)
            if m[0].shape[0] != image.shape[0] or m[0].shape[1] != image.shape[
                    1]:
                print("Mask and image have different shape for {}"
                      .format(image_id))
            if m[0].max() < 1:
                print("Mask {} has a zero area")
        mask = np.stack(instance_masks, axis=2)
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids


    def loop_over_masks(self, image_id, mask_id=None, m_correction=True):
        """Loops over the different masks of an image. Attempt to identify
        troublesome masks
        Arguments:
            image_id: The id of the image either in the form of a string or 
            an integer
            m_correction: Whether to load the correction of the mask or 
            simply the original mask itself

        Returns:
        mask: This function is a generator, and it'll yield the masks one 
        at the time
        class_id: This is the class_id of the yielded mask
        """
        if isinstance(image_id, str):
            image_id = self.real_to_id[image_id]
        mask_path = os.path.join(self.image_info[image_id]['m_path'])
        for mask_file in next(os.walk(mask_path))[2]:
            if mask_file[:-4] in self.dn_masks and m_correction:
                print("Found a double nuclei mask")
                m, c_id = self.separate_masks(mask_path, mask_file)
            else:
                m = [skimage.io.imread(os.path.join(mask_path, mask_file))]
                c_id = [1]
            mask = np.stack(m, axis=2)
            class_ids = np.array(c_id, dtype=np.int32)
            yield mask, class_ids, mask_file



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


    def calc_mask_sizes(self):
        """Calculates the sizes of all the masks in the dataset

        Returns:
        mask_sizes: An array containing the sizes of the bounding boxes
        of the masks
        """
        m_sizes = []
        for im_id in range(len(self.image_ids)):
            masks, class_ids = self.load_mask(im_id, m_correction=True)
            boxes = utils.extract_bboxes(masks)
            m_widths = boxes[:,2] - boxes[:,0]
            m_heights = boxes[:,3] - boxes[:,1]
            assert (m_widths>0).all(), "You got some negative widths"
            assert (m_heights>0).all(), "You got some negative heights"
            m_size = np.stack((m_widths, m_heights),axis=1)
            m_sizes.append(m_size)
        mask_sizes = np.vstack(m_sizes)
        # Functions to plot the sizes
        # import seaborn as sns
        # sns.regplot(x=m_sizes[:,0],y=m_sizes[:,1],fit_reg=False)
        # sns.jointplot(x=m_sizes[:,0], y=m_sizes[:,1], kind="kde")
        return mask_sizes


    def calc_im_sizes(self):
        """Calculates the sizes of all the masks in the dataset

        Returns:
        mask_sizes: An array containing the sizes of the bounding boxes
        of the masks
        """
        im_sizes = []
        for im_id in range(len(self.image_ids)):
            # masks, class_ids = self.load_mask(im_id, m_correction=True)
            image = self.load_image(im_id)
            im_sizes.append(image.shape[:-1])
        im_sizes = np.vstack(im_sizes)
        # Functions to plot the sizes
        # import seaborn as sns
        # sns.regplot(x=m_sizes[:,0],y=m_sizes[:,1],fit_reg=False)
        # sns.jointplot(x=m_sizes[:,0], y=m_sizes[:,1], kind="kde")
        return im_sizes


