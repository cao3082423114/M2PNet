import re
import sys
import glob as gb
sys.path.append('core')

from PIL import Image
from glob import glob
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from configs.submissions import get_cfg
from core.utils.misc import process_cfg
from core import datasets
from core.utils import flow_viz
from core.utils import frame_utils
import cv2
import math
import os.path as osp

from core.FlowFormer import build_flowformer

from core.utils.utils import InputPadder, forward_interpolate
import itertools
import inspect
import torch.nn as nn

TRAIN_SIZE = [480, 856]


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

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

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

def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
  if min_overlap >= TRAIN_SIZE[0] or min_overlap >= TRAIN_SIZE[1]:
    raise ValueError(
        f"Overlap should be less than size of patch (got {min_overlap}"
        f"for patch size {patch_size}).")
  if image_shape[0] == TRAIN_SIZE[0]:
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0]))
  else:
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0] - min_overlap))
  if image_shape[1] == TRAIN_SIZE[1]:
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1]))
  else:
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1] - min_overlap))

  # Make sure the final patch is flush with the image boundary
  hs[-1] = image_shape[0] - patch_size[0]
  ws[-1] = image_shape[1] - patch_size[1]
  return [(h, w) for h in hs for w in ws]

def compute_weight(hws, image_shape, patch_size=TRAIN_SIZE, sigma=1.0, wtype='gaussian'):
    patch_num = len(hws)
    h, w = torch.meshgrid(torch.arange(patch_size[0]), torch.arange(patch_size[1]))
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    c_h, c_w = 0.5, 0.5
    h, w = h - c_h, w - c_w
    weights_hw = (h ** 2 + w ** 2) ** 0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * (weights_hw) ** 2)

    weights = torch.zeros(1, patch_num, *image_shape)
    for idx, (h, w) in enumerate(hws):
        weights[:, idx, h:h+patch_size[0], w:w+patch_size[1]] = weights_hw
    weights = weights.cuda()
    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(weights[:, idx:idx+1, h:h+patch_size[0], w:w+patch_size[1]])

    return patch_weights

def compute_flow(model, image1, image2, weights=None):
    print(f"computing flow...")

    image_size = image1.shape[1:]

    image1, image2 = image1[None].cuda(), image2[None].cuda()

    hws = compute_grid_indices(image_size)
    if weights is None:     # no tile
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_pre, _ = model(image1, image2)

        flow_pre = padder.unpad(flow_pre)
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()
    else:                   # tile
        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            flow_pre, _ = model(image1_tile, image2_tile)
            padding = (w, image_size[1]-w-TRAIN_SIZE[1], h, image_size[0]-h-TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow_pre = flows / flow_count
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()

    return flow

def compute_adaptive_image_size(image_size):
    target_size = TRAIN_SIZE
    scale0 = target_size[0] / image_size[0]
    scale1 = target_size[1] / image_size[1]

    if scale0 > scale1:
        scale = scale0
    else:
        scale = scale1

    image_size = (int(image_size[1] * scale), int(image_size[0] * scale))

    return image_size

def prepare_image(root_dir, raw_outroot,img_root_dir, fn1, fn2, keep_size):
    print(f"preparing image...")
    print(f"root dir = {root_dir}, fn = {fn1}")

    image1 = frame_utils.read_gen(osp.join(root_dir, fn1))
    image2 = frame_utils.read_gen(osp.join(root_dir, fn2))
    image1 = np.array(image1).astype(np.uint8)[..., :3]
    image2 = np.array(image2).astype(np.uint8)[..., :3]
    if not keep_size:
        dsize = compute_adaptive_image_size(image1.shape[0:2])
        image1 = cv2.resize(image1, dsize=dsize, interpolation=cv2.INTER_CUBIC)
        image2 = cv2.resize(image2, dsize=dsize, interpolation=cv2.INTER_CUBIC)
    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float()


    dirname = osp.dirname(fn1)
    filename = osp.splitext(osp.basename(fn1))[0]

    viz_dir = os.path.basename(dirname)

    os.makedirs(osp.join(root_dir, img_root_dir, viz_dir), exist_ok=True)
    os.makedirs(osp.join(root_dir, raw_outroot, viz_dir), exist_ok=True)
    viz_fn = osp.join(root_dir,img_root_dir,viz_dir, filename + '.png')
    raw_fn=osp.join(root_dir,raw_outroot,viz_dir, filename + '.flo')

    return image1, image2, viz_fn,raw_fn

def build_model():
    print(f"building  model...")
    cfg = get_cfg()
    model = build_flowformer(cfg)
    model=model.cuda()

    # Load pre-training weights
    pretrained_weights = torch.load(cfg.model)

    # Remove the "module." prefix to match the model structure.
    new_state_dict = {}
    for key, value in pretrained_weights.items():
        if key.startswith("module."):
            new_key = key[7:]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    model.load_state_dict(new_state_dict)

    model.cuda()
    model.eval()

    return model

def visualize_flow(root_dir, raw_outroot,img_root_dir, model, img_pairs, keep_size):
    weights = None
    for img_pair in img_pairs:
        fn1, fn2 = img_pair
        print(f"processing {fn1}, {fn2}...")

        image1, image2, viz_fn,raw_fn = prepare_image(root_dir, raw_outroot,img_root_dir, fn1, fn2, keep_size)
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        flow = compute_flow(model, image1, image2, weights)

        #optim
        # Optical flow reconstruction of the image1
        re_image1 = flow_warp(image2.unsqueeze(0).cpu(), torch.from_numpy(flow).permute(2,0,1).float().unsqueeze(0).cpu())

        # Handling incorrect optical flow
        binary_weights = calculate_binary_flow_weights(image1.unsqueeze(0).cpu(), re_image1, args.thr_value, args.thr_type)
        flow = flow * binary_weights[:, :, None]

        # save raw flow
        writeFlowFile(raw_fn , flow)

        flow_img = flow_viz.flow_to_image(flow)
        cv2.imwrite(viz_fn, flow_img[:, :, [2,1,0]])

def process_sintel(sintel_dir):
    img_pairs = []
    for scene in os.listdir(sintel_dir):
        dirname = osp.join(sintel_dir, scene)
        image_list = sorted(glob(osp.join(dirname, '*.png')))
        for i in range(len(image_list)-1):
            img_pairs.append((image_list[i], image_list[i+1]))

    return img_pairs

def generate_pairs(dirname,reverse,gap):
    img_pairs = []
    all_files=os.listdir(dirname)
    all_files=sorted(all_files)
    # image_numbers=[int(re.search(r'\d+',filename).group()) for filename in all_files]
    # image_numbers.sort()
    # image_count=len(image_numbers)
    # last_image_number=image_numbers[-1]
    last_image_filename=all_files[-1]
    for idx in range(0, int(last_image_filename.split('.')[0])-gap+1):
        if reverse:
            img1 = osp.join(dirname, f'{idx+gap:05}.jpg')
            img2 = osp.join(dirname, f'{idx:05}.jpg')
            # img1 = f'{idx:06}.png'
            # img2 = f'{idx+1:06}.png'
        else:
            img1 = osp.join(dirname, f'{idx:05}.jpg')
            img2 = osp.join(dirname, f'{idx+gap:05}.jpg')
            # img1 = f'{idx:06}.png'
            # img2 = f'{idx+1:06}.png'
        img_pairs.append((img1, img2))

    return img_pairs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_type', default='seq')
    parser.add_argument('--root_dir', default='.')
    # parser.add_argument('--sintel_dir', default='datasets/Sintel/test/clean')
    parser.add_argument('--seq_dir', default='data/SegTrackv2/JPEGImages')
    # parser.add_argument('--start_idx', type=int, default=0)     # starting index of the image sequence
    # parser.add_argument('--end_idx', type=int, default=81)    # ending index of the image sequence
    parser.add_argument('--viz_root_dir', default='data/SegTrackv2')
    parser.add_argument('--keep_size',type=bool,default=True)     # keep the image size, or the image will be adaptively resized.
    parser.add_argument('--thr_type', type=str, default='percentile')
    parser.add_argument('--thr_value', type=int, default=90)

    args = parser.parse_args()
    root_dir = args.root_dir
    viz_root_dir = args.viz_root_dir
    model = build_model()



    if args.eval_type == 'seq':
        gap = [1]
        # gap = [1, 2]
        reverse = [0, 1]
        folder = gb.glob(os.path.join(args.seq_dir, '*').replace("\\", "/"))
        folder=sorted(folder)
        for r in reverse:
            for g in gap:
                for f in folder:
                    f = f.replace("\\", "/")
                    img_pairs = generate_pairs(f,r,g)
                    if r==1:
                        raw_outroot=viz_root_dir+'/Flows_gap-{}/'.format(g)
                        img_root_dir=viz_root_dir+'/FlowImages_gap-{}/'.format(g)
                    elif r==0:
                        raw_outroot = viz_root_dir + '/Flows_gap{}/'.format(g)
                        img_root_dir=viz_root_dir+'/FlowImages_gap{}/'.format(g)
                    with torch.no_grad():
                        visualize_flow(root_dir, raw_outroot, img_root_dir, model, img_pairs, args.keep_size)
