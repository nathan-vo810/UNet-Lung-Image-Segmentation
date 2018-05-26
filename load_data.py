from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
import cv2

# dir & path
DIR = os.path.dirname(__file__)

TRAIN_IMAGE_PATH = os.path.join(DIR, "./dataset/training_data/train/")
TEST_IMAGE_PATH = os.path.join(DIR, "./dataset/training_data/test/")
NUMPY_PATH = os.path.join(DIR,"./dataset/training_data/npydata/")

IMAGE_PATH = TRAIN_IMAGE_PATH + "img/"
LABEL_PATH = TRAIN_IMAGE_PATH + "label/"


class dataProcess(object):
    """
    A class used to prepare data for train and test
    It will first store images in form of numpy array
    Then, it will help the main program to load the numpy array for training
    """

    def __init__(self, width, height, data_path=IMAGE_PATH, label_path=LABEL_PATH, npy_path=NUMPY_PATH,
                 test_path=TEST_IMAGE_PATH,
                 img_type=".jpg"):
        self.width = width
        self.height = height
        self.data_path = data_path
        self.label_path = label_path
        self.img_type = img_type
        self.test_path = test_path
        self.npy_path = npy_path

    def createTrainData(self):
        i = 0
        print("-" * 30)
        print("Creating train images...")
        print("-" * 30)

        imgs = os.listdir(self.data_path)
        imgs = [img for img in imgs if img.endswith(self.img_type)]
        print(len(imgs))
        # Create numpy array to store all data
        images_train = np.ndarray((len(imgs), self.width, self.height, 1), dtype=np.uint8)
        labels_train = np.ndarray((len(imgs), self.width, self.height, 1), dtype=np.uint8)
        # Loop over all images
        for img in imgs:
            # Load image and label
            image = load_img(self.data_path + img, target_size=(self.width, self.height), grayscale=True)
            label = load_img(self.label_path + img, target_size=(self.width, self.height), grayscale=True)
            # Transform image and label to arrays
            image = img_to_array(image)
            label = img_to_array(label)
            # Store image and label arrays in numpy array
            images_train[i] = image
            labels_train[i] = label
            i += 1
            print("Done {0}/{1} images".format(i, len(imgs)))
        print("Loading done.")
        np.save(self.npy_path + "images_train.npy", images_train)
        np.save(self.npy_path + "labels_train.npy", labels_train)
        print("Saving data to .npy files done.")

    def createTestData(self):
        i = 0
        print("-" * 30)
        print("Creating test images...")
        print("-" * 30)

        imgs = os.listdir(self.test_path)
        imgs = [img for img in imgs if img.endswith(self.img_type)]
        print(len(imgs))
        # Create numpy array to store all data
        images_test = np.ndarray((len(imgs), self.width, self.height, 1), dtype=np.uint8)
        # Loop over all images
        for img in imgs:
            # Load image
            image = load_img(self.test_path + img, target_size=(self.width, self.height), grayscale=True)
            # Transform image to arrays
            image = img_to_array(image)
            # Store image array in numpy array
            images_test[i] = image
            i += 1
            print("Done {0}/{1} images".format(i, len(imgs)))
        print("Loading done.")
        np.save(self.npy_path + "images_test.npy", images_test)
        print("Saving data to .npy files done.")

    def loadTrainData(self):
        print("-" * 30)
        print("Load train images...")
        print("-" * 30)
        # Load numpy files
        images_train = np.load(self.npy_path + "images_train.npy")
        labels_train = np.load(self.npy_path + "labels_train.npy")
        # Prepocess data
        images_train = images_train.astype('float32')
        labels_train = labels_train.astype('float32')
        images_train /= 255
        labels_train /= 255
        labels_train[labels_train > 0.5] = 1
        labels_train[labels_train <= 0.5] = 0
        return images_train, labels_train

    def loadTestData(self):
        print("-" * 30)
        print("Load test images...")
        print("-" * 30)
        # Load numpy files
        images_test = np.load(self.npy_path + "images_test.npy")
        # Prepocess data
        images_test = images_test.astype('float32')
        images_test /= 255
        return images_test

if __name__ == "__main__":
    mydata = dataProcess(512,512)
    # mydata.createTrainData()
    mydata.createTestData()