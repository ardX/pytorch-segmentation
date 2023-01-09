import cv2
import numpy as np
import os
import glob
import argparse
from pathlib import Path

# Dice similarity function
def dice(pred, true, k = 1):
    intersection = np.sum(pred[true==k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice

def main():
    args = parse_arguments()
    path = Path('.')

    #source = 'data/annotations/validation-ori'
    gt = args.gt
    path_gt = path/gt

    #pred = 'outputs/UNet'
    pred = args.pred
    path_pred = path/pred

    filenames = [img for img in glob.glob(str(path_gt/"*.*"))]
    filenames.sort()

    with open(args.output, 'w') as f:
        for img in filenames:
            img_path = f'{img}'
            img_split = Path(img_path).stem

            y_true = cv2.imread(os.path.join(path_gt, img_split+".png")) 
            y_pred = cv2.imread(os.path.join(path_pred, img_split+".png"))
            
            n_true = np.sum(y_true == 255)
            n_pred = np.sum(y_pred == 255)

            diff = abs(n_true-n_pred)/n_true

            dice_score = dice(y_pred, y_true, k = 255) #255 in my case, can be 1 
            print ("{0} DSC: {1} NGT: {2} NPR: {3} ERR: {4}".format(img_split, dice_score, n_true, n_pred, diff), file=f)

def parse_arguments():
    parser = argparse.ArgumentParser(description='DSC')
    parser.add_argument('-g', '--gt', default='data/annotations/validation-ori',type=str,
                        help='Ground Truth')
    parser.add_argument('-p', '--pred', default='outputs', type=str,
                        help='Prediction')
    parser.add_argument('-o', '--output', default='outputs', type=str,  
                        help='Output Path')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
