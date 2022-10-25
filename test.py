import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from scipy import misc
import cv2
from utils.data_val import test_dataset
from network import DAD



parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=416, help='testing size——352——416——512')
parser.add_argument('--pth_path', type=str, default='./trained_model/pvtv2_DAD_COD.pth')
parser.add_argument("-nocrf", "--nocrf", action="store_false")
opt = parser.parse_args()


for _data_name in['DUT-OMRON', 'DUTS-TE', 'ECSSD','HKU-IS', 'PASCAL-S', 'SOD']:
    data_path = './dataset/SOD/{}'.format(_data_name)
    save_path = './test_result/{}/'.format(_data_name)
    model = DAD.DAD_pvt_v2().cuda()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()
    print("success!")
    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        ori_image, image, gt, name, _ = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        output = model(image)

        res = F.upsample(output[2], size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = res * 255
        if not opt.nocrf:#we use crf for mirror detection
            print('crf')
            from crf import crf_refine
            res = crf_refine(np.array(ori_image.convert('RGB')).astype(np.uint8), res.astype(np.uint8))
        print('> {} - {}'.format(_data_name, name))
        # misc.imsave(save_path+name, res)
        # If `mics` not works in your environment, please comment it and then use CV2
        # cv2.imwrite(save_path + name,res*255)
        cv2.imwrite(save_path + name, res)
