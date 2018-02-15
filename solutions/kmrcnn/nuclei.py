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

from config import Config
import utils
import model as modellib
import skimage


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

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 512

     # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50
    
    # MEAN_PIXEL = [43.53287505, 39.56061986, 48.22454996, 255.]
    MEAN_PIXEL = [44.57284587, 40.71265898, 48.6901747]
    
    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)
    

############################################################
#  Dataset
############################################################


class NucleiDataset(utils.Dataset):
    def load_nuclei(self, dataset_dir, subset):
        """Load a subset of the nuclei dataset.
        dataset_dir: The root directory of the nuclei dataset.
        subset: What to load (train, test)
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """
        # Add Classes
        self.add_class("nuclei", 1, "nu")
        
        image_dir = "{}/stage1_{}".format(dataset_dir, subset)
        
        image_ids = next(os.walk(image_dir))[1]
        
        # Add Images
        for i in image_ids:
            i_path = os.path.join(image_dir, i,'images',i+'.png')
            m_path = os.path.join(image_dir, i, 'masks')
            self.add_image(
            "nuclei", image_id=i,
            path=i_path,
            m_path=m_path)
                
    def load_image(self, image_id, remove_alpha=True):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if remove_alpha == True and image.shape[-1] != 3:
            # print("Image {} has a shape of {}".format(image_id, image.shape))
            if image.shape[-1] == 4:
                image = image[:,:,:3]
        if image.ndim != 3:
            print("Image dimension not 3")
            image = skimage.color.gray2rgb(image)
        return image
        
    
    
    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image = self.load_image(image_id)
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
        
