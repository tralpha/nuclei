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
import glob

from config import Config
import nutils as utils
import model as modellib
import skimage
import copy
from sklearn.model_selection import StratifiedShuffleSplit
from IPython.core.debugger import set_trace
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.segmentation import (morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient)

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
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

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
    STEPS_PER_EPOCH = 533 // (IMAGES_PER_GPU * GPU_COUNT)

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 134 // (IMAGES_PER_GPU * GPU_COUNT)

    # MEAN_PIXEL = [43.53287505, 39.56061986, 48.22454996, 255.]
    MEAN_PIXEL = [44.57284587, 40.71265898, 48.6901747]

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False
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

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.15194419, 0.15036527, 0.28001702, 0.28143383])
    BBOX_STD_DEV = np.array([1.0, 1.0, 1.0, 1.0])


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
            image_ids = [iid for iid in image_ids]
        else:
            image_ids = next(os.walk(image_dir))[1]
        # set_trace()
        # Data structure to hold read_id to index_id mapping
        self.real_to_id = {}
        # Data structure to hold the double nuclei masks
        self.dn_masks = {}
        bad_masks = open("bad_masks", "r")
        ex_images = open("ex_images", "r")
        b_lines = bad_masks.readlines()
        self.ex_lines = ex_images.readlines()
        for ex_im in self.ex_lines:
            if ex_im[:-1] in image_ids:
                image_ids.remove(ex_im[:-1])
        # set_trace()
        self.b_masks = (m for m in b_lines)
        for b_mask in self.b_masks:
            _, image_id, _, mask_id = b_mask.split("\\")
            mask_id = mask_id[:-5]
            self.dn_masks[mask_id] = image_id
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
            # set_trace()
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
        new_masks_path = os.path.join(mask_path + '_1')
        if os.path.exists(new_masks_path):
            new_mask_path = os.path.join(mask_path + '_1', mask_file)
            if os.path.exists(new_mask_path):
                mask = skimage.io.imread(
                    os.path.join(mask_path + '_1', mask_file))
            else:
                # print("Using Lopuhin's corrections for {}".format(mask_file))
                mask_id = mask_file.split(".")[0]
                new_mask_ids = glob.glob(
                    os.path.join(new_masks_path, mask_id + "*"))
                # set_trace()
                masks = [skimage.io.imread(n_m_id) for n_m_id in new_mask_ids]
                class_ids = [1 for n_m_id in new_mask_ids]
                return masks, class_ids
        # set_trace()
        # if mask_file == "9790898c3892ac0b92a08b9f878f344333023374b7464ee571b0010b98dacc51.png": set_trace()
        if mask.ndim == 3:
            print("Very weird, found a 3D Mask for {}. We'll reshape it".
                  format(mask_file))
            mask = mask[:,:,0]
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
                # print("Found a double nuclei mask")
                # print(mask_file)
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

    def create_watershed(self, mask, markers):
        """
        Function to create the dilated image to feed the final
        watershed function
        """
        dilated_mask = skimage.morphology.dilation(mask)
        return dilated_mask

    def create_rw(self, mask, markers):
        """
        Function to create the random walker segmentation
        """
        labels = skimage.segmentation.random_walker(mask, markers, mode='bf')
        # dilated_mask = skimage.morphology.dilation(mask)
        return labels

    def create_gacig(self, mask, markers):
        """
        Function to create the random walker segmentation
        """
        gimage = inverse_gaussian_gradient(mask)
        init_ls = np.zeros(mask.shape, dtype=np.int8)
        height_pixels, width_pixels = np.where(mask == 255)
        min_height = height_pixels.min()
        max_height = height_pixels.max()
        min_width = width_pixels.min()
        max_width = width_pixels.max()
        ls_h_start = np.minimum(min_height, 10)
        ls_h_end = np.maximum(max_height, mask.shape[0] - 10)
        ls_w_start = np.minimum(min_width, 10)
        ls_w_end = np.maximum(max_width, mask.shape[1] - 10)
        init_ls[ls_h_start:ls_h_end, ls_w_start:ls_w_end] = 1
        gacig_mask = morphological_geodesic_active_contour(
            gimage.astype('float64'),
            230,
            init_ls,
            smoothing=2,
            balloon=-1,
            threshold=0.69)
        return gacig_mask

    def create_gacm(self, mask, markers):
        """
        Function to create the random walker segmentation
        """
        mimage = skimage.filters.median(mask.astype(np.uint8))
        mimage[mask == 0] = 1
        mimage[mask == 255] = 0
        init_ls = np.zeros(mask.shape, dtype=np.int8)
        init_ls[10:-10, 10:-10] = 1
        gacig_mask = morphological_geodesic_active_contour(
            mimage.astype('float64'),
            230,
            init_ls,
            smoothing=2,
            balloon=-1,
            threshold=0.5)
        return gacig_mask

    def base_watershed(self, new_mask, markers, mask_file):
        """
        Does a base watershed which is used by all other methodologies.
        Final function in pipeline, so it saves the image too
        """
        base_marker = np.where(markers == 2)
        # if new_mask[base_marker]
        res = skimage.morphology.watershed(
            new_mask, markers, compactness=0.0001)
        res[res == 1] = 0
        res[res == 2] = 255
        res = res.astype(np.uint8)
        skimage.io.imsave(mask_file, res)
        return res

    def base_markers(self, mask):
        """
        Function to create the markers to be used for watershed algorithm 
        by all of the other segmentation methods
        """
        dist = ndimage.morphology.distance_transform_edt(mask)
        max_pix = np.unravel_index(dist.argmax(), dist.shape)
        markers = np.zeros_like(mask).astype(np.uint8)
        # Check to see that there is no nuclei here in 0,0!
        markers[0, 0] = 1
        markers[max_pix[0], max_pix[1]] = 2
        return markers

    def new_masks(self, dataset_dir, subset="train"):
        """
        Function to create new masks just like Allen did
        """
        images_dir = "{}/stage1_{}".format(dataset_dir, subset)
        image_ids = next(os.walk(images_dir))[1]
        # Loop over all of the images
        for image_id in image_ids:
            if image_id + "\n" in self.ex_lines:
                continue
            mask_categories = {
                'watershed_masks': {
                    'present': False,
                    'create_func': self.create_watershed
                },
                'rw_masks': {
                    'present': False,
                    'create_func': self.create_rw
                },
                # 'gacm_masks': {
                #     'present': False,
                #     'create_func': self.create_gacm,
                # },
                # 'gacig_masks': {
                #     'present': False,
                #     'create_func': self.create_gacig
                # },
            }
            im_path = os.path.join(images_dir, image_id)
            for mask_category in mask_categories:
                if not os.path.isdir(os.path.join(im_path, mask_category)):
                    # set_trace()
                    print("{} not yet present".format(mask_category))
                    # mask_categories[mask_category] = True
                    os.makedirs(os.path.join(im_path, mask_category))
                # image = self.load_image(image_id)
                masks = self.load_mask(image_id)[0]
                for m_id in range(masks.shape[-1]):
                    m_file = os.path.join(
                        im_path, mask_category,
                        mask_category + "_" + str(m_id) + ".png")
                    if os.path.exists(m_file):
                        continue
                    else:
                        mask = masks[:, :, m_id]
                        markers = self.base_markers(mask)
                        modified_mask = mask_categories[mask_category][
                            'create_func'](mask, markers)
                        final_mask = self.base_watershed(modified_mask,
                                                         markers, m_file)
                        # if mask_category == "gacig_masks": set_trace()

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
        train, val = self.split_dataset(0.2)
        # set_trace()
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
            m_widths = boxes[:, 2] - boxes[:, 0]
            m_heights = boxes[:, 3] - boxes[:, 1]
            assert (m_widths > 0).all(), "You got some negative widths"
            assert (m_heights > 0).all(), "You got some negative heights"
            m_size = np.stack((m_widths, m_heights), axis=1)
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
