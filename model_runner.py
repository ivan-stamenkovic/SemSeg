import tensorflow as tf
import numpy as np
import scipy.misc
import os.path
from glob import glob

import image_preprocessor
from global_settings import GLOBAL


def run_single_image(file_src, file_dst, session, image_placeholder, logits, keep_prob):
    image = image_preprocessor.prepare_image(file_src)

    #Inference
    image_softmax = session.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 1.0, image_placeholder: [image]})

    #Lose background, reshape to GLOBAL.IMAGE_SHAPE
    image_softmax = image_softmax[0][:, 1].reshape(GLOBAL.IMAGE_SHAPE[0], GLOBAL.IMAGE_SHAPE[1])
    #Prediction to 0 and 1
    segmentation = (image_softmax > 0.5).reshape(GLOBAL.IMAGE_SHAPE[0], GLOBAL.IMAGE_SHAPE[1], 1)

    #Apply mask to original image
    mask = np.dot(segmentation, np.array([[0,255,0,127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    final_image = scipy.misc.toimage(image)
    final_image.paste(mask, box=None, mask=mask)
    scipy.misc.imsave(file_dst, final_image)


def run_image_dir(dir_src, dir_dst, session, image_placeholder, logits, keep_prob):
    for image_src in glob(os.path.join(dir_src,GLOBAL.IMAGE_DIRECTORY_NAME, '*' + GLOBAL.IMAGE_EXTENSION)):
        image_dst = os.path.join(dir_dst,GLOBAL.OUTPUT_DIRECTORY_NAME, os.path.basename(image_src))
        run_single_image(image_src, image_dst, session, image_placeholder, logits, keep_prob)
