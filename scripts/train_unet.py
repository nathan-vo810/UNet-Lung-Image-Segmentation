from src.UNet_model import UNet
from src.load_data import dataProcess

MODEL_PATH = '../pretrained_models/unet.pkl'

WIDTH = 512
HEIGHT = 512

data_loader = dataProcess(WIDTH, HEIGHT)
# images_train, labels_train = data_loader.loadTrainData()
images_test = data_loader.loadTestData()

model = UNet()

# Train
# model.fit(images_train, labels_train)
# model.save(MODEL_PATH)

# Predict
model.load(MODEL_PATH)
model.predict(images_test)
