import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from scipy import misc
import cv2
from utils.data_val import test_dataset
from network.DAD import DAD
from tqdm import tqdm
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=480, help='testing size——352——416——512')
parser.add_argument('--backbone', type=str, default='resnet', help='backbone')
parser.add_argument('--pth_path', type=str, default='')
parser.add_argument('--task', type=str, default='COD')
parser.add_argument("-nocrf", "--nocrf", action="store_false")
parser.add_argument("--coarse", default=False, action='store_true')
parser.add_argument('--group', type=int, default=4, help='group numbers')
parser.add_argument('--method', type=str, default='cosine', help='group numbers')
opt = parser.parse_args()

if opt.task == 'COD':
    data_names = ['CAMO', 'COD10K', 'CHAMELEON', 'NC4K']
    data_source = '/home/user/COD/Data/COD_data/TestDataset'
elif opt.task == 'SOD':
    data_names = ['DUT-OMRON','DUTS-TE','ECSSD', 'HKU-IS', 'PASCAL-S', 'SOD']
    data_source = '/home/user/COD/Data/SOD'
win_list = [2,4,6,8]
# Pre-load model to avoid reloading for each dataset
model = DAD(method = opt.method, group = opt.group, win_size=win_list, backbone_name=opt.backbone, channel=64).cuda()
model.load_state_dict(torch.load(opt.pth_path + opt.backbone + '/' + opt.task + '/Net_epoch_best.pth'), strict=False)
model.eval()

for _data_name in data_names:
    data_path = os.path.join(data_source, _data_name)
    save_path = os.path.join('./Official_results/', opt.pth_path.split('/')[-2], 'coarse' if opt.coarse else 'refine', opt.backbone, _data_name)
    os.makedirs(save_path, exist_ok=True)

    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    WFM, SM, EM, M = WeightedFmeasure(), Smeasure(), Emeasure(), MAE()

    with torch.no_grad():
        for i in tqdm(range(test_loader.size)):
            image, gt, name, _ = test_loader.load_data()
            gt = np.array(gt)
            image = image.cuda()

            output = model(image)[0] if opt.coarse else model(image)[1]
            res_0 = F.interpolate(output, size=gt.shape, mode='bilinear', align_corners=False)
            res = (res_0.sigmoid().cpu().numpy().squeeze() > 0.5).astype(np.uint8)

            WFM.step(pred=res, gt=gt)
            SM.step(pred=res, gt=gt)
            EM.step(pred=res, gt=gt)
            M.step(pred=res, gt=gt)

            res = res * 255
            if not opt.nocrf:
                from crf import crf_refine
                res = crf_refine(np.array(ori_image.convert('RGB')).astype(np.uint8), res)

            cv2.imwrite(os.path.join(save_path, name), res)

    results = {
        'size': opt.testsize,
        "Smeasure": SM.get_results()["sm"],
        "wFmeasure": WFM.get_results()["wfm"],
        "MAE": M.get_results()["mae"],
        "meanEm": EM.get_results()["em"]["curve"].mean(),
    }

    print(results)
    with open(os.path.join(opt.pth_path, opt.backbone, opt.task, 'eval.txt'), "a") as file:
        output_string = '{} {} {} {} & {:.3f} & {:.3f} & {:.3f} & {:.3f}\n'.format(
            opt.testsize, opt.pth_path.split('/')[-1], opt.backbone, _data_name, results['Smeasure'],
            results['meanEm'], results['wFmeasure'], results['MAE'])
        file.write(output_string)
