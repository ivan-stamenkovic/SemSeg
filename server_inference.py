import tensorflow as tf
import os.path
import time

import model_runner
from global_settings import GLOBAL


class InferenceManager:

    def __init__(self):
        self.initialized = False


    def initialize(self):
        self.initialized = True

        tf.reset_default_graph()

        #Open session and import graph from file
        imported_graph = tf.train.import_meta_graph(os.path.join(GLOBAL.SAVED_MODEL_PATH, GLOBAL.SAVED_MODEL_NAME + ".meta"))
        self.session = tf.Session()
        imported_graph.restore(self.session, os.path.join(GLOBAL.SAVED_MODEL_PATH, GLOBAL.SAVED_MODEL_NAME))

        #Get relevant tensors from imported graph
        graph = tf.get_default_graph()
        self.image_input = graph.get_tensor_by_name('image_input:0')
        self.keep_prob = graph.get_tensor_by_name('keep_prob:0')
        self.logits = graph.get_tensor_by_name('fcn_logits:0')


    def process_image(self, image_file):
        filename = str(time.time()) + "." + GLOBAL.IMAGE_EXTENSION
        file_src = os.path.join(GLOBAL.TMP_IMAGE_DIRECTORY, filename)
        file_dst = os.path.join(GLOBAL.TMP_OUTPUT_DIRECTORY, filename)

        image_file.save(file_src)
        model_runner.run_single_image(file_src, file_dst, self.session, self.image_input, self.logits, self.keep_prob)
        return file_dst

    def deinitialize(self):
        if (self.initialized == True):
            self.initialized = False
            self.session.close()
            tf.reset_default_graph()

