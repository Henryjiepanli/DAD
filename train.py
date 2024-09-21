import os
import ast
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
from datetime import datetime
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from network.DAD import DAD
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
from utils.data_val import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, poly_lr, min_poly_lr
import warnings
warnings.filterwarnings("ignore")

class Trainer:
    def __init__(self, config):
        self.config = config
        win_list = [2,4,6,8]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DAD(method = config.method, group = config.group, win_size=win_list, backbone_name=config.backbone, channel=64).to(self.device)
        # for param in self.model.backbone.parameters():
        #     param.requires_grad = False
        self.optimizer = torch.optim.Adam(self.model.parameters(), config.lr)
        self.writer = SummaryWriter(config.save_path + 'summary')
        self.best_mae = float('inf')
        self.best_dice = float('-inf')
        self.best_iou = float('-inf')
        self.best_sm = float('-inf')
        self.best_epoch = 0
        self.step = 0
        # Set up logging
        logging.basicConfig(filename=config.save_path + 'log.log',
                            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                            level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
        logging.info("Network-Train")
        logging.info(f'Config: {config}')

        # Load model if specified
        if config.load:
            self.model.load_state_dict(torch.load(config.load), strict=False)
            logging.info(f'Loaded model from {config.load}')

        # Prepare data loaders
        self.train_loader = get_loader(
            img_root=config.train_root + 'Imgs/',
            gt_root=config.train_root + 'GT/',
            trainsize=config.trainsize,
            mosaic_ratio=config.replace_ratio,
            batchsize=config.batchsize,
            num_workers=4  # Adjust as needed
        )
        self.val_loader = test_dataset(
            image_root=config.val_root + 'Imgs/',
            gt_root=config.val_root + 'GT/',
            testsize=config.trainsize
        )

    def structure_loss(self, pred, mask):
        """
        Custom loss function (ref: F3Net-AAAI-2020)
        """
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()



    def train(self, epoch):
        self.model.train()
        loss_all = 0
        epoch_step = 0

        scales = [448, 480, 512]  # Define scales

        for i, (images, gts) in enumerate(self.train_loader, start=1):
            self.optimizer.zero_grad()
            images, gts = images.to(self.device), gts.to(self.device)

            # Randomly choose a scale for this batch
            scale = random.choice(scales)
            images = F.interpolate(images, size=[scale, scale], mode='bilinear', align_corners=False)
            gts = F.interpolate(gts, size=[scale, scale], mode='nearest')  # Use 'nearest' for labels

            preds = self.model(images)
            loss_1 = self.structure_loss(preds[0], gts) 
            loss_2 = self.structure_loss(preds[1], gts)

            loss = loss_1 + loss_2
            loss.backward()

            clip_gradient(self.optimizer, self.config.clip)
            self.optimizer.step()

            self.step += 1
            epoch_step += 1
            loss_all += loss.item()

            if i % 20 == 0 or i == len(self.train_loader) or i == 1:
                log_msg = (f'{datetime.now()} Epoch [{epoch:03d}/{self.config.epoch:03d}], '
                        f'Step [{i:04d}/{len(self.train_loader):04d}], '
                        f'Total_loss: {loss.item():.4f}, Loss1: {loss_1.item():.4f}, Loss2: {loss_2.item():.4f}')
                print(log_msg)
                logging.info(log_msg)
                self.writer.add_scalars('Loss_Statistics',
                                        {'Loss_1': loss_1.item(), 'Loss_2': loss_2.item(), 'Loss_total': loss.item()},
                                        global_step=self.step)

        loss_all /= epoch_step
        logging.info(f'[Train Info]: Epoch [{epoch:03d}/{self.config.epoch:03d}], Loss_AVG: {loss_all:.4f}')
        self.writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)


    def val_dice(self, epoch):
        self.model.eval()
        DSC = 0.0

        with torch.no_grad():
            for i in range(len(self.val_loader)):
                image, gt, name, _ = self.val_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                image = image.to(self.device)

                res = self.model(image)
                res = F.interpolate(res[1], size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)

                input_flat = np.reshape(res, (-1))
                target_flat = np.reshape(gt, (-1))
                intersection = input_flat * target_flat
                dice = (2 * intersection.sum() + 1) / (input_flat.sum() + target_flat.sum() + 1)
                DSC += dice

            dice = DSC / len(self.val_loader)
            self.writer.add_scalar('DICE', dice, global_step=epoch)

            log_msg = f'Epoch: {epoch}, DICE: {dice:.4f}, bestDICE: {self.best_dice:.4f}, bestEpoch: {self.best_epoch}'
            print(log_msg)
            logging.info(f'[Val Info]: {log_msg}')

            if dice > self.best_dice:
                self.best_dice = dice
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), self.config.save_path + 'Net_epoch_best.pth')
                logging.info(f'Saved state_dict successfully! Best epoch: {epoch}.')

    def val_iou(self, epoch):
        self.model.eval()
        IOU = 0.0
        mae_sum = 0.0

        with torch.no_grad():
            for i in range(len(self.val_loader)):
                image, gt, name, _ = self.val_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                image = image.to(self.device)

                res = self.model(image)
                res = F.interpolate(res[1], size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)

                mae_sum += np.sum(np.abs(res - gt)) / (gt.shape[0] * gt.shape[1])
                
                input_flat = np.reshape(res, (-1))
                target_flat = np.reshape(gt, (-1))
                intersection = input_flat * target_flat
                dice = (2 * intersection.sum() + 1) / (input_flat.sum() + target_flat.sum() + 1)
                iou = dice / (2 - dice)
                IOU += iou

            mae = mae_sum / len(self.val_loader)
            iou = IOU / len(self.val_loader)
            self.writer.add_scalar('IOU', iou, global_step=epoch)

            log_msg = (f'Epoch: {epoch}, IOU: {iou:.4f}, MAE: {mae:.4f}, '
                    f'bestIOU: {self.best_iou:.4f}, bestMAE: {self.best_mae:.4f}, bestEpoch: {self.best_epoch}')
            print(log_msg)
            logging.info(f'[Val Info]: {log_msg}')

            if mae < self.best_mae:
                self.best_mae = mae
            if iou > self.best_iou:
                self.best_iou = iou
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), self.config.save_path + 'Net_epoch_best.pth')
                logging.info(f'Saved state_dict successfully! Best epoch: {epoch}.')

    def validate(self, epoch):
        """
        Validation function
        """
        self.model.eval()
        WFM, SM, EM, M = WeightedFmeasure(), Smeasure(), Emeasure(), MAE()

        with torch.no_grad():
            for i in range(len(self.val_loader)):
                image, gt, name, _ = self.val_loader.load_data()
                gt = np.array(gt)
                image = image.to(self.device)

                res = self.model(image)[1]
                res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().cpu().numpy().squeeze()
                res[res >= 0.5] = 1
                res[res < 0.5] = 0

                WFM.step(pred=res, gt=gt)
                SM.step(pred=res, gt=gt)
                EM.step(pred=res, gt=gt)
                M.step(pred=res, gt=gt)

            wfm = WFM.get_results()["wfm"]
            sm = SM.get_results()["sm"]
            em = EM.get_results()["em"]["curve"].mean()
            mae = M.get_results()["mae"]

            self.writer.add_scalar('wFm', wfm, global_step=epoch)
            self.writer.add_scalar('Sm', sm, global_step=epoch)
            self.writer.add_scalar('Em', em, global_step=epoch)
            self.writer.add_scalar('MAE', mae, global_step=epoch)

            log_msg = (f'Epoch: {epoch}, wFm: {wfm:.4f}, Sm: {sm:.4f}, Em: {em:.4f}, MAE: {mae:.4f}, '
                       f'bestSm: {self.best_sm:.4f}, bestEpoch: {self.best_epoch}')
            print(log_msg)
            logging.info(f'[Val Info]: {log_msg}')

            if sm > self.best_sm:
                self.best_sm = sm
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), self.config.save_path + 'Net_epoch_best.pth')
                logging.info(f'Saved state_dict successfully! Best epoch: {epoch}.')


    def run(self):
        """
        Run the training and validation process
        """
        if self.config.task == 'MSD':
            for epoch in range(1, self.config.epoch + 1):
                if self.config.strategy == 'Poly':
                    cur_lr = poly_lr(self.optimizer, self.config.lr, epoch, self.config.epoch)
                elif self.config.strategy == 'Linear':
                    cur_lr = adjust_lr(self.optimizer, self.config.lr, epoch, 0.1, 50)
                self.writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
                self.train(epoch)
                self.val_iou(epoch)

        elif self.config.task == 'Poly':
            for epoch in range(1, self.config.epoch + 1):
                if self.config.strategy == 'Poly':
                    cur_lr = poly_lr(self.optimizer, self.config.lr, epoch, self.config.epoch)
                elif self.config.strategy == 'Linear':
                    cur_lr = adjust_lr(self.optimizer, self.config.lr, epoch, 0.1, 50)
                self.writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
                self.train(epoch)
                self.val_dice(epoch)

        else:
            for epoch in range(1, self.config.epoch + 1):
                if self.config.strategy == 'Poly':
                    cur_lr = poly_lr(self.optimizer, self.config.lr, epoch, self.config.epoch)
                if self.config.strategy == 'min_Poly':
                    cur_lr = min_poly_lr(self.optimizer, self.config.lr, epoch, self.config.epoch)
                elif self.config.strategy == 'Linear':
                    cur_lr = adjust_lr(self.optimizer, self.config.lr, epoch, 0.1, 10)
                self.writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
                self.train(epoch)
                self.validate(epoch)


def seed_everything(seed):
    """
    Set random seed for reproducibility
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=40, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='Training batch size')
    parser.add_argument('--trainsize', type=int, default=480, help='Training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='Gradient clipping margin')
    parser.add_argument('--load', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID to use')
    parser.add_argument('--backbone', type=str, default='', help='Backbone network')
    parser.add_argument('--save_path', type=str, default='./Experiments/DAD/', help='Path to save model and log')
    parser.add_argument('--replace_ratio', type=float, default=0.25, help='Replace ratio')
    parser.add_argument('--task', type=str, default='COD', help='Task type (COD/SOD/Poly/MSD)')
    parser.add_argument('--strategy', type=str, default='Poly', help='Training Strategy type (Poly/Linear)')
    parser.add_argument('--group', type=int, default=4, help='group numbers')
    parser.add_argument('--seed', type=int, default=2333, help='seed numbers')
    parser.add_argument('--method', type=str, default='cosine', help='group numbers')
    opt = parser.parse_args()
    opt.save_path = opt.save_path + opt.backbone + '/' + opt.task + '/'

    if opt.task == 'COD':
        opt.train_root = '/home/user/COD/Data/COD_data/TrainDataset/'
        opt.val_root = '/home/user/COD/Data/COD_data/TestDataset/CAMO/'
    elif opt.task == 'SOD':
        opt.train_root = '/home/user/COD/Data/SOD/DUTS-TR/'
        opt.val_root = '/home/user/COD/Data/SOD/PASCAL-S/'
    elif opt.task == 'MSD':
        opt.train_root = '/home/user/COD/Data/MSD/train/'
        opt.val_root = '/home/user/COD/Data/MSD/test/'
    elif opt.task == 'Poly':
        opt.train_root = '/home/user/COD/Data/Poly_datset/TrainDataset/'
        opt.val_root = '/home/user/COD/Data/Poly_datset/TestDataset/CVC-ClinicDB/'

    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    cudnn.benchmark = True
    print(f'Using GPU {opt.gpu_id}')

    # Seed for reproducibility
    seed_everything(opt.seed)

    # Initialize Trainer
    trainer = Trainer(opt)
    trainer.run()
