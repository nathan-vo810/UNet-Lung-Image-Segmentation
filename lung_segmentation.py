from UNet_model import *

UNET_MODEL = ''


def parse_args():
    """Parse input arguments"""
    import argparse
    parser = argparse.ArgumentParser(description='Lung Region Segmentaion')
    args = parser.parse_args()
    return args


def main(args):
    # Load test image
    test_image = ''

    # Load Model
    segmentation_model = UNet(512, 512)
    segmentation_model.load_weights(UNET_MODEL)

    # Predict
    result_image = segmentation_model.predict(test_image)


if __name__ == '__main__':
    main(parse_args())
