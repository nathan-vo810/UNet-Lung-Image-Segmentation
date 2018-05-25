import os
import cv2

# dir & path
DIR = os.path.dirname(__file__)

TRAIN_IMAGE_PATH = os.path.join(DIR, "./dataset/training_data/train/")
TEST_IMAGE_PATH = os.path.join(DIR, "./dataset/training_data/test/")

IMAGE_PATH = TRAIN_IMAGE_PATH + "img/"
LABEL_PATH = TRAIN_IMAGE_PATH + "label/"
SAVE_PATH = TRAIN_IMAGE_PATH + "result/"

INPUT_SHAPE = (512, 512)


class LoadData:
    def __init__(self):
        self.image_path = IMAGE_PATH
        self.label_path = LABEL_PATH

    def load(self, path):
        files = os.listdir(path)
        for file in files:
            file_path = path + file
            image = cv2.imread(file_path)
            width = image.shape[0]
            height = image.shape[1]
            image = cv2.resize(image, INPUT_SHAPE)
            up_scale_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(SAVE_PATH + file, up_scale_image)

    def load_images(self):
        print("Loading images...")
        self.load(self.image_path)
        print("Images loaded.")

    def load_labels(self):
        print("Loading labels...")
        self.load(self.label_path)
        print("Labels loaded.")
