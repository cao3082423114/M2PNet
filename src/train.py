import os
import time
import einops
import sys
import cv2
import numpy as np
import utils as ut
import config as cg
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
from model import SlotAttentionAutoEncoder
# from model_pic import SlotAttentionAutoEncoder
# from model_newname import SlotAttentionAutoEncoder
from eval import eval
from tqdm import tqdm
from contrib import Ranger
from ranger21 import Ranger21


def init_weights(layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight.data)


def loss_calc1(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    labels = torch.ge(label, 0.5).float()
    #
    batch_size = label.size()
    # print(batch_size)
    num_labels_pos = torch.sum(labels)
    #
    batch_1 = batch_size[0] * batch_size[2]
    batch_1 = batch_1 * batch_size[3]
    weight_1 = torch.div(num_labels_pos, batch_1)  # pos ratio
    weight_1 = torch.reciprocal(weight_1)
    # print(num_labels_pos, batch_1)
    weight_2 = torch.div(batch_1 - num_labels_pos, batch_1)
    # print('postive ratio', weight_2, weight_1)
    weight_22 = torch.mul(weight_1,
                          torch.ones(batch_size[0], batch_size[1], batch_size[2], batch_size[3]).cuda())
    # weight_11 = torch.mul(weight_1,  torch.ones(batch_size[0], batch_size[1], batch_size[2]).cuda())
    criterion = torch.nn.BCELoss(
        weight=weight_22)  # weight = torch.Tensor([0,1]) .cuda() #torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    # loss = class_balanced_cross_entropy_loss(pred, label).cuda()

    return criterion(pred, label)


def loss_calc2(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    # Variable(label.long()).cuda()
    criterion = torch.nn.L1Loss()  # .cuda() #torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()

    return criterion(pred, label)


def main(args):
    global eval_freq, frame_mean_iou
    lr = args.lr
    epsilon = 1e-5
    num_slots = args.num_slots
    iters = args.num_iterations
    batch_size = args.batch_size
    warmup_it = args.warmup_steps
    decay_step = args.decay_steps
    num_it = args.num_train_steps
    resume_path = args.resume_path
    args.resolution = (128, 224)

    # setup log and model path, initialize tensorboard,
    [logPath, modelPath, resultsPath] = cg.setup_path(args)
    writer = SummaryWriter(logdir=logPath)
    # initialize dataloader (validation bsz has to be 1 for FBMS, because of different resolutions, otherwise, can be >1)
    trn_dataset, val_dataset, resolution, in_out_channels, use_flow, loss_scale, ent_scale, cons_scale = cg.setup_dataset(
        args)
    trn_loader = ut.FastDataLoader(
        trn_dataset, num_workers=4, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = ut.FastDataLoader(
        val_dataset, num_workers=2, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)
    # initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SlotAttentionAutoEncoder(resolution=resolution,
                                     num_slots=num_slots,
                                     in_out_channels=in_out_channels,
                                     iters=iters)

    model.to(device)
    # initialize training
    criterion = nn.MSELoss()
    optimizer = Ranger21(model.parameters(), lr=lr,weight_decay=lr*0.001,num_epochs=3e5,num_batches_per_epoch=len(trn_loader))

    model.apply(init_weights)

    it = 0
    if resume_path:
        print('resuming from checkpoint')

        pretrain_dict = torch.load(resume_path)
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(pretrain_dict['optimizer_state_dict'])
        it = pretrain_dict['iteration']
        loss = pretrain_dict['loss']


    else:
        print('training from scratch')

    # save every eval_freq iterations
    moca = False
    monitor_train_iou = True
    log_freq = 100  # report train iou to tensorboard
    if args.dataset == "DAVIS":
        eval_freq = 1e3
    elif args.dataset == "MoCA":
        eval_freq = 1e4
        monitor_train_iou = False  # this is slow due to moca evaluation
        moca = True
    elif args.dataset == "FBMS":
        eval_freq = 1e3
        monitor_train_iou = False  # there is no train IoU to monitor
    elif args.dataset == "STv2":
        eval_freq = 1e3

    print('======> start training {}, {}, use {}.'.format(args.dataset, args.verbose, device))
    iou_best = 0
    timestart = time.time()

    # overfit single batch for debug
    # sample = next(iter(loader))
    total_iters = len(trn_loader) * num_it
    model.train()
    while it < num_it:
        for _, sample in enumerate(trn_loader):
            # #inference / evaluate on validation set
            if it % eval_freq == 0:
                frame_mean_iou = eval(val_loader, model, device, moca, use_flow, it, writer=writer, train=True)

            optimizer.zero_grad()
            flow, gt, res_img, imgAtt = sample
            gt = gt.float().to(device)
            flow = flow.float().to(device)
            imgAtt = imgAtt.float().to(device)
            res_img = res_img.float().to(device)

            flow = einops.rearrange(flow, 'b t c h w -> (b t) c h w')
            imgAtt = einops.rearrange(imgAtt, 'b t c h w -> (b t) c h w')
            res_img = einops.rearrange(res_img, 'b t c h w -> (b t) c h w')

            if monitor_train_iou:
                gt = einops.rearrange(gt, 'b t c h w -> (b t) c h w')

            recon_image, recons, masks, _ = model(flow, res_img, imgAtt)

            recon_loss = loss_scale * criterion(flow, recon_image)

            entropy_loss = ent_scale * -(masks * torch.log(masks + epsilon)).sum(dim=1).mean()
            # consistency loss, need to consider the permutation invariant nature.
            tmasks = einops.rearrange(masks, '(b t) s c h w -> b t s c h w', b=batch_size)
            mask_t_1 = tmasks[:, 0]
            mask_t = tmasks[:, 1]
            mask_t = einops.rearrange(mask_t, 'b s c h w -> b c s h w')

            temporal_diff = torch.pow((mask_t_1 - mask_t), 2).mean([-1, -2])
            consistency_loss = cons_scale * temporal_diff.view(-1, 2 * 2).min(1)[0].mean()

            ####################################################################################
            # loss_calc = loss_calc1(flow, recon_image)+loss_calc2(flow, recon_image)*0.8
            ####################################################################################

            loss = recon_loss + entropy_loss + consistency_loss

            loss.backward()
            optimizer.step()
            print('iteration {},'.format(it),
                  'time {:.01f}s,'.format(time.time() - timestart),
                  'loss {:.02f}.'.format(loss.detach().cpu().numpy()),
                  'Current learning rate: {}'.format(optimizer.param_groups[0]['lr']))

            if it % log_freq == 0:
                writer.add_scalar('Loss/total', loss.detach().cpu().numpy(), it)
                writer.add_scalar('Loss/reconstruction', recon_loss.detach().cpu().numpy(), it)
                writer.add_scalar('Loss/entropy', entropy_loss.detach().cpu().numpy(), it)
                writer.add_scalar('Loss/consistency', consistency_loss.detach().cpu().numpy(), it)
                if monitor_train_iou:
                    # calculate iou, choose best mask
                    iou, _ = ut.hungarian_iou(masks, gt)
                    iou = iou.detach().cpu().numpy()
                    writer.add_scalar('IOU/train', iou, it)

            # save model
            if it % eval_freq == 0 and frame_mean_iou > iou_best:
                filename = os.path.join(modelPath,
                                        'checkpoint_{}_iou_{}.pth'.format(it, np.round(frame_mean_iou, 3))).replace(
                    "\\", "/")
                torch.save({
                    'iteration': it,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, filename)
                iou_best = frame_mean_iou
                print("Saved parameters of the model at iteration {},with IOU {}".format(it, frame_mean_iou))
            elif it % eval_freq == 0 and frame_mean_iou <= iou_best:
                # save final model
                filename = os.path.join(modelPath, 'final_checkpoint.pth').replace("\\", "/")
                torch.save({
                    'iteration': it,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, filename)

            # LR warmup
            if it < warmup_it:
                ut.set_learning_rate(optimizer, lr * it / warmup_it)

            # LR decay
            if it % decay_step == 0 and it > 0:
                ut.set_learning_rate(optimizer, lr * (0.5 ** (it // decay_step)))
                ent_scale = ent_scale * 5.0
                cons_scale = cons_scale * 5.0

            it += 1
            timestart = time.time()

            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser()
    # optimization
    parser.add_argument('--batch_size', type=int, default=1)  # not to set 2,STV2 mast 1
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--num_train_steps', type=int, default=5e9)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--decay_steps', type=int, default=3e4)
    parser.add_argument('--decay_rate', type=float, default=0.7)
    # settings
    parser.add_argument('--dataset', type=str, default='DAVIS', choices=['DAVIS', 'MoCA', 'FBMS', 'STv2'])
    parser.add_argument('--flow_to_rgb', type=bool, default=True)
    # architecture
    parser.add_argument('--num_slots', type=int, default=2)
    parser.add_argument('--num_iterations', type=int, default=5)#FBMS:10
    # misc
    parser.add_argument('--verbose', type=str, default="None")
    parser.add_argument('--resume_path', type=str, default="")
    args = parser.parse_args()
    args.inference = False
    main(args)
