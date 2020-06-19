import argparse
import os
import cv2
import torch
import torch.nn as nn
from vis_flux import vis_flux
import torch.nn.functional as F
from model.fluxnet import FluxNet
from datasets import FluxSkeletonTestDataset
from torch.utils.data import Dataset, DataLoader

DATASET = 'sklarge'
SNAPSHOT_DIR = './snapshots/'
SAVE_DIR = 'test_pred_skl/'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Super-BPD Network")
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help="Dataset for training.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--save-pred-dir", type=str, default=SAVE_DIR,
                        help="Where to save output of the model.")
    return parser.parse_args()

args = get_arguments()

def main():

    if not os.path.exists(args.save_pred_dir + args.dataset):
        os.makedirs(args.save_pred_dir + args.dataset)

    model = FluxNet(model='resnet101')

    model.load_state_dict(torch.load(args.snapshot_dir + args.dataset + '_120000.pth'))

    model.eval()
    model.cuda()
    
    dataloader = DataLoader(FluxSkeletonTestDataset(dataset=args.dataset), batch_size=1, shuffle=True, num_workers=4)

    for i_iter, batch_data in enumerate(dataloader):

        Input_image, vis_image, image_name = batch_data

        # if '08c2' in image_name[0]:
        pred_flux, pred_skl = model(Input_image.cuda())

        pred_skl = torch.sigmoid(pred_skl)


        # vis_flux(vis_image, pred_flux, pred_flux, pred_skl, image_name[0], 'test_pred_skl/sklarge/')

        cv2.imwrite(args.save_pred_dir + args.dataset + '/' + image_name[0][:-3] + 'png', 255*(pred_skl.data.cpu().numpy()[0,0]))

if __name__ == '__main__':
    main()