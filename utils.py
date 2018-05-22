import numpy as np

class dataProcess(object):
    def __init__(self, out_rows, out_cols):
        self.out_rows = out_rows
        self.out_cols = out_cols

    def load_train_data(self):
        print('Load training data')
        training_images =  np.load(self.npy_path)
