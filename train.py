import argparse
import os
import torch
import torch.nn as nn
from model.fluxnet import FluxNet
from vis_flux import vis_flux
from datasets import FluxSkeletonDataset
from torch.utils.data import Dataset, DataLoader

INI_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 5e-4
EPOCHES = 10000
DATASET = 'sklarge'
SNAPSHOT_DIR = './snapshots/'
TRAIN_DEBUG_VIS_DIR = './train_debug_vis/'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Super-BPD Network")
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help="Dataset for training.")
    parser.add_argument("--train-debug-vis-dir", type=str, default=TRAIN_DEBUG_VIS_DIR,
                        help="Directory for saving vis results during training.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    return parser.parse_args()

args = get_arguments()

def loss_calc(pred_flux, gt_flux, pred_skl, gt_skl, dilmask):

    device_id = pred_flux.device
    gt_flux = gt_flux.cuda(device_id)

    pos_sum = (dilmask > 0).sum().float()
    neg_sum = (dilmask == 0).sum().float()

    pos_matrix = (neg_sum / (pos_sum + neg_sum)) * dilmask
    neg_matrix = (pos_sum / (pos_sum + neg_sum)) * (1 - dilmask)

    weight_matrix = (pos_matrix + neg_matrix).cuda(device_id)

    flux_loss = weight_matrix * (pred_flux - gt_flux) ** 2
    flux_loss = flux_loss.sum() / 2.

    skl_pos_sum = (gt_skl > 0).sum().float()
    skl_neg_sum = (gt_skl == 0).sum().float()

    skl_pos_matrix = (skl_neg_sum / (skl_pos_sum + skl_neg_sum)) * gt_skl
    skl_neg_matrix = (skl_pos_sum / (skl_pos_sum + skl_neg_sum)) * (1 - gt_skl)

    skl_weight_matrix = (skl_pos_matrix + skl_neg_matrix).cuda(device_id)

    skl_criterion = nn.BCEWithLogitsLoss(reduction='none').cuda(device_id)

    gt_skl = gt_skl.cuda(device_id)
    skl_loss = skl_weight_matrix * skl_criterion(pred_skl, gt_skl)
    skl_loss = skl_loss.sum()

    return flux_loss, skl_loss

def get_params(model, key, bias=False):

    # for added layer
    if key == "added":
        for m in model.named_modules():
            if "layer" not in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    if not bias:
                        yield m[1].weight
                    else:
                        yield m[1].bias

def adjust_learning_rate(optimizer, step):
    
    if step == 8e4:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

def main():

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    if not os.path.exists(args.train_debug_vis_dir + args.dataset):
        os.makedirs(args.train_debug_vis_dir + args.dataset)

    model = FluxNet()

    model.train()
    model.cuda()
    
    optimizer = torch.optim.Adam(
        params=[
            {
                "params": get_params(model, key="added", bias=False),
                "lr": 10 * INI_LEARNING_RATE  
            },
            {
                "params": get_params(model, key="added", bias=True),
                "lr": 20 * INI_LEARNING_RATE   
            },
        ],
        lr = INI_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    dataloader = DataLoader(FluxSkeletonDataset(dataset=args.dataset), batch_size=1, shuffle=True, num_workers=4)

    global_step = 0

    for epoch in range(1, EPOCHES):

        for i_iter, batch_data in enumerate(dataloader):

            global_step += 1

            Input_image, vis_image, gt_skl, dilmask, gt_flux, dataset_length, image_name = batch_data

            optimizer.zero_grad()

            pred_flux, pred_skl = model(Input_image.cuda())

            flux_loss, skl_loss = loss_calc(pred_flux, gt_flux, pred_skl, gt_skl, dilmask)

            total_loss = flux_loss + skl_loss

            total_loss.backward()

            optimizer.step()

            if global_step % 100 == 0:
                print('epoche {} i_iter/total {}/{} flux_loss {:.2f} skl_loss {:.2f}'.format(\
                       epoch, i_iter, int(dataset_length.data), flux_loss, skl_loss))
                
            if global_step % 500 == 0:
                vis_flux(vis_image, pred_flux, gt_flux, pred_skl, str(global_step) + '_' + image_name[0], args.train_debug_vis_dir + args.dataset + '/')

            if global_step % 1e4 == 0:
                torch.save(model.state_dict(), args.snapshot_dir + args.dataset + '_' + str(global_step) + '.pth')
                
            if global_step % 12e4 == 0:
                return

if __name__ == '__main__':
    main()