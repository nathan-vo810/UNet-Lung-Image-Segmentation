def parse_args():
    """Parse input arguments"""
    import argparse
    parser = argparse.ArgumentParser(description='Lung Region Segmentaion')
    args = parser.parse_args()
    return args

if __name == '__main__':
    main(parse_args())