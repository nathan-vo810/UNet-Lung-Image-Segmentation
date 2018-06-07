from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.layers.merge import concatenate
from keras.models import Model
from keras.optimizers import Adam
import os
import numpy as np
from keras.preprocessing.image import array_to_img

from src import dataProcess

DIR = os.path.dirname(__file__)
WEIGHTS_PATH = os.path.join(DIR, "../weights/unet-new.hdf5")
RESULT_PATH = os.path.join(DIR, "../dataset/training_data/result/")


class UNet(object):
    """UNet Implemenation"""

    def __init__(self, image_width=512, image_height=512):
        self.image_width = image_width
        self.image_height = image_height

    def load_data(self):
        mydata = dataProcess(self.image_width, self.image_height)
        images_train, labels_train = mydata.loadTrainData()
        images_test = mydata.loadTestData()
        return images_train, labels_train, images_test

    def build_unet(self):
        inputs = Input(shape=(self.image_width, self.image_height, 1))

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = UpSampling2D(size=(2, 2))(drop5)
        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up6)
        # merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
        merge6 = concatenate([drop4, up6], axis=-1)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = UpSampling2D(size=(2, 2))(conv6)
        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up7)
        # merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
        merge7 = concatenate([conv3, up7], axis=-1)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = UpSampling2D(size=(2, 2))(conv7)
        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up8)
        # merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
        merge8 = concatenate([conv2, up8], axis=-1)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = UpSampling2D(size=(2, 2))(conv8)
        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
        # merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
        merge9 = concatenate([conv1, up9], axis=-1)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=conv10)

        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def train(self):
        print("Loading data")
        images_train, labels_train, images_test = self.load_data()
        print("Loading data done")
        model = self.build_unet()
        print("Got UNet")
        model.summary()
        # model.load_weights(WEIGHTS_PATH)
        # print("Weights loaded")
        weight_path = "unet-{loss:.2f}.hdf5"
        model_checkpoint = ModelCheckpoint(weight_path, monitor='loss', verbose=1, save_best_only=True)
        print("Fitting model...")
        model.fit(images_train, labels_train, batch_size=4, epochs=10, verbose=1, validation_split=0.2, shuffle=True,
                  callbacks=[model_checkpoint])

        print("Predict test data")
        labels_test = model.predict(images_test, batch_size=1, verbose=1)
        np.save(RESULT_PATH + "labels_test.npy", labels_test)

    def save_images(self):
        print("Array to images")
        images = np.load(RESULT_PATH + "labels_test.npy")
        for i in range(images.shape[0]):
            image = images[i]
            image = array_to_img(image)
            image.save(RESULT_PATH + "{}.jpg".format(i))


if __name__ == "__main__":
    print("Training new label...")
    myUNet = UNet()
    myUNet.train()
    myUNet.save_images()
