import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import PIL
from PIL import ImageEnhance


def cv_random_flip(img, label):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label

def randomCrop_Mosaic(image, label, crop_win_width, crop_win_height):
    image_width = image.size[0]
    image_height = image.size[1]
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region)

def randomCrop(image,label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region)

def randomRotation(image, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return image, label

def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))

def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255
    return Image.fromarray(img)

# dataset for training
class ChangeDataset(data.Dataset):
    def __init__(self, img_root, gt_root, trainsize, mosaic_ratio=0.25):
        self.trainsize = trainsize
        # get filenames
        self.image_root =  img_root
        self.gt_root = gt_root
        self.mosaic_ratio = mosaic_ratio
        self.images = [self.image_root + f for f in os.listdir(self.image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [self.gt_root + f for f in os.listdir(self.gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize),interpolation=PIL.Image.NEAREST),
            transforms.ToTensor()])
        # get size of dataset
        self.size = len(self.images)

    def __getitem__(self, index):
        # read imgs/gts/grads/depths
        p_ratio = random.random()
        if p_ratio > self.mosaic_ratio:
            image, gt = self.load_img_and_mask(index)
            image, gt = cv_random_flip(image, gt)
            image, gt = randomCrop(image, gt)
            image, gt = randomRotation(image, gt)
            image = colorEnhance(image)
        else:
            image, gt = self.load_mosaic_img_and_mask(index)
            image, gt = cv_random_flip(image, gt)
            image, gt = randomRotation(image, gt)
            image = colorEnhance(image)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        return image, gt


    def load_img_and_mask(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.gts[index]).convert('L')
        return image, mask

    def load_mosaic_img_and_mask(self, index):
       indexes = [index] + [random.randint(0, self.size - 1) for _ in range(3)]
       img_a, mask_a = self.load_img_and_mask(indexes[0])
       img_b, mask_b = self.load_img_and_mask(indexes[1])
       img_c, mask_c = self.load_img_and_mask(indexes[2])
       img_d, mask_d = self.load_img_and_mask(indexes[3])

       w = self.trainsize
       h = self.trainsize

       start_x = w // 4
       strat_y = h // 4
        # The coordinates of the splice center
       offset_x = random.randint(start_x, (w - start_x))
       offset_y = random.randint(strat_y, (h - strat_y))

       crop_size_a = (offset_x, offset_y)
       crop_size_b = (w - offset_x, offset_y)
       crop_size_c = (offset_x, h - offset_y)
       crop_size_d = (w - offset_x, h - offset_y)

       croped_a, mask_crop_a = randomCrop_Mosaic(img_a.copy(), mask_a.copy(),crop_size_a[0], crop_size_a[1]) 
       croped_b, mask_crop_b = randomCrop_Mosaic(img_b.copy(), mask_b.copy(),crop_size_b[0], crop_size_b[1])
       croped_c, mask_crop_c = randomCrop_Mosaic(img_c.copy(), mask_c.copy(),crop_size_c[0], crop_size_c[1])
       croped_d, mask_crop_d = randomCrop_Mosaic(img_d.copy(), mask_d.copy(),crop_size_d[0], crop_size_d[1])

       croped_a, mask_crop_a = np.array(croped_a), np.array(mask_crop_a)
       croped_b, mask_crop_b = np.array(croped_b), np.array(mask_crop_b)
       croped_c, mask_crop_c = np.array(croped_c), np.array(mask_crop_c)
       croped_d, mask_crop_d = np.array(croped_d), np.array(mask_crop_d)

       top = np.concatenate((croped_a, croped_b), axis=1)
       bottom = np.concatenate((croped_c, croped_d), axis=1)
       img = np.concatenate((top, bottom), axis=0)


       top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
       bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
       mask = np.concatenate((top_mask, bottom_mask), axis=0)
       mask = np.ascontiguousarray(mask)

       img = np.ascontiguousarray(img)

       img = Image.fromarray(img)
       mask = Image.fromarray(mask)

       return img, mask

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)

        self.images = images
        self.gts = gts


    def __len__(self):
        return self.size

def get_loader(img_root, gt_root, trainsize, mosaic_ratio, batchsize, num_workers=1, shuffle=True, pin_memory=True):

    dataset =ChangeDataset(img_root = img_root, gt_root = gt_root, trainsize = trainsize, mosaic_ratio = mosaic_ratio)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader



class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root)]
        # self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        ori_image = image
        image = self.transform(image).unsqueeze(0)
        # print(self.gts)
        # print(len(self.images),len(self.gts))

        gt = self.binary_loader(self.gts[self.index])

        name = self.images[self.index].split('/')[-1]

        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        self.index += 1
        self.index = self.index % self.size

        return image, gt, name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


