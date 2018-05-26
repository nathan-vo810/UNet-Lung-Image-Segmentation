from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
import cv2

# dir & path
DIR = os.path.dirname(__file__)

TRAIN_IMAGE_PATH = os.path.join(DIR, "./dataset/training_data/train/")
TEST_IMAGE_PATH = os.path.join(DIR, "./dataset/training_data/test/")

IMAGE_PATH = TRAIN_IMAGE_PATH + "img/"
LABEL_PATH = TRAIN_IMAGE_PATH + "label/"
MERGE_PATH = TRAIN_IMAGE_PATH + "merge/"
AUG_MERGE_PATH = TRAIN_IMAGE_PATH + "aug/"
AUG_IMAGE_PATH = AUG_MERGE_PATH + "img/"
AUG_LABEL_PATH = AUG_MERGE_PATH + "label/"


class myAugmentation(object):
    """
    A class used to generate more data
    Firstly, it reads train image and label separately. Then, it will merge them together
    Secondly, it uses keras preprocessing to generate more image (augmentation)
    Finally, it separate augmented images apart into train image and label
    """

    def __init__(self, image_path=IMAGE_PATH, label_path=LABEL_PATH, merge_path=MERGE_PATH,
                 aug_merge_path=AUG_MERGE_PATH, aug_image_path=AUG_IMAGE_PATH, aug_label_path=AUG_LABEL_PATH,
                 img_type="jpg"):
        """
        Using glob to get all image of type "image type" from path
        """

        self.train_imgs = glob.glob(image_path + "*." + img_type)
        self.label_imgs = glob.glob(label_path + "*." + img_type)
        self.train_path = image_path
        self.label_path = label_path
        self.merge_path = merge_path
        self.img_type = img_type
        self.aug_merge_path = aug_merge_path
        self.aug_train_path = aug_image_path
        self.aug_label_path = aug_label_path

        self.slices = len(self.train_imgs)
        self.data_generator = ImageDataGenerator(rotation_range=0.2, width_shift_range=0.05, height_shift_range=0.05,
                                                 shear_range=0.05, zoom_range=0.05, horizontal_flip=True,
                                                 fill_mode='nearest')

    def augmentation(self):
        """
        Start augmentation
        """
        trains = self.train_imgs
        labels = self.label_imgs
        path_train = self.train_path
        path_label = self.label_path
        path_merge = self.merge_path
        img_type = self.img_type
        path_aug_merge = self.aug_merge_path

        # Check validity of trains and labels
        if len(trains) != len(labels) or len(trains) == 0 or len(labels) == 0:
            print("Train can't match label")
            return 0

        # Loop over train image and augmentate data
        filenames = glob.glob(path_train + "*." + img_type)
        filenames = os.listdir(path_train)
        for file in filenames:
            # Load image and label
            if file.endswith(self.img_type):
                img_train = load_img(path_train + file)
                label_train = load_img(path_label + file)

            # Convert image and label into arrays
            img_array = img_to_array(img_train)
            label_array = img_to_array(label_train)

            # Merge
            img_array[:, :, 2] = label_array[:, :, 0]
            img_temp = array_to_img(img_array)
            img_temp.save(path_merge + file)
            img = img_array
            img = img.reshape((1,) + img.shape)

            # Do augmentate over merge image
            savedir = path_aug_merge + file
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            self.doAugmentate(img, savedir, file)

    def doAugmentate(self, img, save_to_dir, save_name, batch_size=1, imgnum=30):
        """
        Augmentate one image
        """
        save_prefix = save_name.split(".")[0]
        save_format = save_name.split(".")[1]
        data_generator = self.data_generator
        i = 0
        for batch in data_generator.flow(img, batch_size=batch_size, save_to_dir=save_to_dir, save_prefix=save_prefix,
                                         save_format=save_format):
            i += 1
            if i > imgnum:
                break

    def splitMerge(self):
        """
        Split merged image apart
        """
        path_merge = self.aug_merge_path
        path_train = self.aug_train_path
        path_label = self.aug_label_path

        filenames = os.listdir(path_merge)
        for file in filenames:
            if file.endswith(self.img_type):
                aug = cv2.imread(file)
                img_aug = aug[:,:,2]
                label_aug = aug[:,:,0]
                cv2.imwrite(path_train + file, img_aug)
                cv2.imwrite(path_label + file, label_aug)

if __name__ == "__main__":
    aug = myAugmentation()
    aug.augmentation()
    aug.splitMerge()
