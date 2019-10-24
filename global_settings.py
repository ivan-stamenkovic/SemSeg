class GLOBAL:
    #Paths and Names
    RAW_IMAGE_DIRECTORY = "./dataset/raw"
    PROCESSED_IMAGE_DIRECTORY = "./dataset/prepared"
    TEST_IMAGE_DIRECTORY = "./dataset/test"
    TMP_IMAGE_DIRECTORY = "./tmp/image"
    TMP_OUTPUT_DIRECTORY = "./tmp/output"
    IMAGE_DIRECTORY_NAME = "image"
    GT_DIRECTORY_NAME = "gt_s"
    OUTPUT_DIRECTORY_NAME = "output"
    VGG_PATH = "./models/vgg"
    SAVED_MODEL_PATH = "./models/fcn"
    SAVED_MODEL_NAME = "nn"
    IMAGE_EXTENSION = "jpg"
    #Training Constants
    NUMBER_OF_CLASSES = 2
    IMAGE_SHAPE = (384, 512)
    EPOCHS = 30
    BATCH_SIZE = 4
    DROPOUT = 0.75
    KEEP_PROB_VALUE = 0.5
    LEARNING_RATE = 0.001