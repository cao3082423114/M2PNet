import sys

sys.path.append('core')

import os
import cv2
import glob
import torch
import argparse
import numpy as np

import PIL
from PIL import Image

from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder
import inspect
import torch.nn as nn

import torchvision.transforms.functional as TF

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid


def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2


def flow_warp(x, flow12, pad='border', mode='bilinear'):
    B, _, H, W = x.size()

    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW

    v_grid = norm_grid(base_grid + flow12)  # BHW2
    if 'align_corners' in inspect.getfullargspec(torch.nn.functional.grid_sample).args:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad, align_corners=True)
    else:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad)
    return im1_recons


def calculate_binary_flow_weights(org_img, warped_img, thr_value, thr_type):
    diff = abs(org_img - warped_img)[0].permute(1, 2, 0).cpu().numpy().mean(2)
    diff = diff / diff.max()
    if thr_type is not None:
        thr = np.percentile(diff, thr_value)
        diff[diff > thr] = 1
        diff[diff <= thr] = 0

    weights = 1 - diff
    return weights


def writeFlowFile(filename, uv):
    """
    According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein
    Contact: dqsun@cs.brown.edu
    Contact: schar@middlebury.edu
    """
    TAG_STRING = np.array(202021.25, dtype=np.float32)
    if uv.shape[2] != 2:
        sys.exit("writeFlowFile: flow must have two bands!")
    H = np.array(uv.shape[0], dtype=np.int32)
    W = np.array(uv.shape[1], dtype=np.int32)
    with open(filename, 'wb') as f:
        f.write(TAG_STRING.tobytes())
        f.write(W.tobytes())
        f.write(H.tobytes())
        f.write(uv.tobytes())


def load_image(imfile, resolution=None):
    img = Image.open(imfile)
    if resolution:
        img = img.resize(resolution, PIL.Image.ANTIALIAS)
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def predict(args):
    model = RAFT(args)
    state_dict = torch.load(args.model)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png').replace("\\", "/")) + \
                 glob.glob(os.path.join(args.path, '*.jpg').replace("\\", "/"))

        folder = os.path.basename(args.path)
        floout = os.path.join(args.outroot, folder).replace("\\", "/")
        rawfloout = os.path.join(args.raw_outroot, folder).replace("\\", "/")

        os.makedirs(floout, exist_ok=True)
        os.makedirs(rawfloout, exist_ok=True)

        gap = args.gap
        images = sorted(images)
        images_ = images[:-gap]

        for index, imfile1 in enumerate(images_):
            if args.reverse:
                image1 = load_image(images[index + gap])
                image2 = load_image(imfile1)
                svfile = images[index + gap]
            else:
                image1 = load_image(imfile1)
                image2 = load_image(images[index + gap])
                svfile = imfile1

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

            flopath = os.path.join(floout, os.path.basename(svfile)).replace("\\", "/")
            rawflopath = os.path.join(rawfloout, os.path.basename(svfile)).replace("\\", "/").replace("jpg","png")

            # Optical flow reconstruction of the image1
            re_image1 = flow_warp(image2.cpu(), flow_up.cpu())

            flo = flow_up[0].permute(1, 2, 0).cpu().numpy()
            # Handling incorrect optical flow
            binary_weights = calculate_binary_flow_weights(image1.cpu(), re_image1, args.thr_value, args.thr_type)
            flo = flo * binary_weights[:, :, None]

            # save raw flow
            writeFlowFile(rawflopath[:-4] + '.flo', flo)

            # save image.
            flo = flow_viz.flow_to_image(flo)
            cv2.imwrite(flopath[:-4] + '.png', flo[:, :, [2, 1, 0]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--resolution', nargs='+', type=int)
    parser.add_argument('--model', default="../models/raft-things.pth", help="restore checkpoint")
    parser.add_argument('--path', default="../data/DAVIS2016/JPEGImages/480p/horsejump-high", help="dataset for prediction")
    parser.add_argument('--gap', default=1, type=int, help="gap between frames")
    parser.add_argument('--outroot', default="../data/123", help="path for output flow as image")
    parser.add_argument('--reverse', default=0, type=int, help="video forward or backward")
    parser.add_argument('--raw_outroot', default="../data/456", help="path for output flow as xy displacement")

    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--thr_type', type=str, default='percentile')
    parser.add_argument('--thr_value', type=int, default=90)
    args = parser.parse_args()

    predict(args)
