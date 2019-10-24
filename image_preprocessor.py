import re
import random
import numpy as np
import os.path
import scipy.misc
from glob import glob

from global_settings import GLOBAL

#Reshape image
def prepare_image(image_path):
    image = scipy.misc.imresize(scipy.misc.imread(image_path), GLOBAL.IMAGE_SHAPE)
    return image

#Reshape ground truth and save in one-hot format
def prepare_gt(gt_path):
    gt_image = scipy.misc.imresize(scipy.misc.imread(gt_path), GLOBAL.IMAGE_SHAPE)

    background_color = np.array([255, 255, 255])
    gt_bg = np.all(gt_image == background_color, axis=2)
    gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
    gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
    return gt_image

#Loop through dataset and prepare images for training
def prepare_dataset():
    image_paths = glob(os.path.join(GLOBAL.RAW_IMAGE_DIRECTORY,
                                    GLOBAL.IMAGE_DIRECTORY_NAME,
                                    '*.' + GLOBAL.IMAGE_EXTENSION))
    
    label_paths = {
        re.sub('_maska_s', '', os.path.basename(path)): path
        for path in glob(os.path.join(GLOBAL.RAW_IMAGE_DIRECTORY,
                                    GLOBAL.GT_DIRECTORY_NAME,
                                    '*.' + GLOBAL.IMAGE_EXTENSION))
    }

    processed_image_directory = os.path.join(GLOBAL.PROCESSED_IMAGE_DIRECTORY, GLOBAL.IMAGE_DIRECTORY_NAME)
    processed_gt_directory = os.path.join(GLOBAL.PROCESSED_IMAGE_DIRECTORY, GLOBAL.GT_DIRECTORY_NAME)
    
    if not os.path.exists(processed_image_directory):
            os.makedirs(processed_image_directory)

    if not os.path.exists(processed_gt_directory):
        os.makedirs(processed_gt_directory)

    i=0
    for image_path in image_paths:
        basename = os.path.basename(image_path)
        scipy.misc.imsave(os.path.join(GLOBAL.PROCESSED_IMAGE_DIRECTORY,
                                    GLOBAL.IMAGE_DIRECTORY_NAME,
                                    basename),
                                    prepare_image(image_path))
        gt_path = label_paths[os.path.basename(image_path)]
        np.save(os.path.join(GLOBAL.PROCESSED_IMAGE_DIRECTORY,
                                    GLOBAL.GT_DIRECTORY_NAME,
                                    basename),
                                    prepare_gt(gt_path))
        if (i % 50 == 0):
            print("Transformed " + str(i) + " images so far.")
        i += 1

#Returns batch function for training
def create_batch_function():
    def batch_function():
        image_paths = glob(os.path.join(GLOBAL.PROCESSED_IMAGE_DIRECTORY,
                                        GLOBAL.IMAGE_DIRECTORY_NAME,
                                        '*' + GLOBAL.IMAGE_EXTENSION))
        label_paths = {
            re.sub('.npy', '', os.path.basename(path)) : path
            for path in glob(os.path.join(GLOBAL.PROCESSED_IMAGE_DIRECTORY,
                                        GLOBAL.GT_DIRECTORY_NAME,
                                        '*.npy'))
        }

        random.shuffle(image_paths)
        for batch in range(0, len(image_paths), GLOBAL.BATCH_SIZE):
            images = []
            gts = []

            for image_path in image_paths[batch : batch + GLOBAL.BATCH_SIZE]:
                gt_image_path = label_paths[os.path.basename(image_path)]

                images.append(scipy.misc.imread(image_path))
                gts.append(np.load(gt_image_path))
            yield np.array(images), np.array(gts)
    
    return batch_function



if __name__ == "__main__":
    prepare_dataset()
