from common import *
from utility.file import *
from utility.draw import *

from dataset.reader import *


def run_make_test_annotation():

    split = 'test1_ids_all_65'
    ids = read_list_from_file(DATA_DIR + '/split/' + split, comment='#')

    num_ids = len(ids)
    for i in range(num_ids):
        folder = ids[i].split('/')[0]
        name = ids[i].split('/')[1]
        image_file = DATA_DIR + '/__download__/%s/%s/images/%s.png' % (
            folder, name, name)

        #image
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)

        ## save and show -------------------------------------------
        image_show('image', image)

        cv2.imwrite(DATA_DIR + '/image/%s/images/%s.png' % (folder, name),
                    image)
        # set_trace()
        cv2.waitKey(1)


def run_make_train_annotation():

    split = 'train1_ids_all_667'
    ids = read_list_from_file(DATA_DIR + '/split/' + split, comment='#')

    data_dir = DATA_DIR + '/image/stage1_train'
    os.makedirs(data_dir + '/multi_masks', exist_ok=True)
    os.makedirs(data_dir + '/overlays', exist_ok=True)
    os.makedirs(data_dir + '/images', exist_ok=True)

    num_ids = len(ids)
    for i in range(num_ids):
        id = ids[i]

        name = id.split('/')[-1]
        folder = id.split('/')[0]
        image_files = glob.glob(DATA_DIR + '/__download__/%s/%s/images/*.png' %
                                (folder, name))
        assert (len(image_files) == 1)
        image_file = image_files[0]
        print(id)

        #image
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        # im = skimage.io.imread(image_file)
        from IPython.core.debugger import set_trace
        H, W, C = image.shape
        if C > 3: set_trace()
        multi_mask = np.zeros((H, W), np.int32)

        bad_masks = pd.read_csv(
            DATA_DIR + "/__download__/bad_masks", sep='\\', header=None)
        # set_trace()
        bm_im_id = bad_masks[1].unique()
        if name in bm_im_id:
            mask_type = "masks_1"
            print("Bad Masks for image {}".format(name))
        else:
            mask_type = "masks"

        mask_files = glob.glob(DATA_DIR + '/__download__/%s/%s/%s/*.png' % (
            folder, name, mask_type))
        # if mask_type == "masks_1": set_trace()
        mask_files.sort()
        num_masks = len(mask_files)
        for i in range(num_masks):
            mask_file = mask_files[i]
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            # Check for overlap here.
            # set_trace()
            # if multi_mask[np.where(mask > 128)].astype(np.bool).any():
            #     set_trace()
            multi_mask[np.where(mask > 128)] = i + 1

        #check
        color_overlay = multi_mask_to_color_overlay(multi_mask, color='summer')
        color1_overlay = multi_mask_to_contour_overlay(
            multi_mask, color_overlay, [255, 255, 255])
        contour_overlay = multi_mask_to_contour_overlay(multi_mask, image,
                                                        [0, 255, 0])
        all = np.hstack((
            image,
            contour_overlay,
            color1_overlay, )).astype(np.uint8)

        cv2.imwrite(data_dir +'/images/%s.png'%(name),image)

        np.save(data_dir + '/multi_masks/%s.npy' % name, multi_mask)
        cv2.imwrite(data_dir + '/multi_masks/%s.png' % name, color_overlay)
        cv2.imwrite(data_dir + '/overlays/%s.png' % name, all)
        cv2.imwrite(data_dir + '/images/%s.png' % name, image)

        image_show('all', all)
        if mask_type == "masks_1": cv2.waitKey()

    # main #################################################################


if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_make_test_annotation()

    print('sucess!')
