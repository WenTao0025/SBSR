import json

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from configs.Vit.config import config_argument

arg = config_argument()
#!/usr/bin/env python
"""
Created by zhenlinx on 11/14/19
"""
# from scipy.ndimage.morphology import binary_dilation
from torchvision.transforms.functional import to_grayscale
from PIL import Image, ImageDraw
import random

#!/usr/bin/env python
"""
Created by zhenlinx on 11/13/2020
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

"""
function for creating LOG kernel comes from https://gist.github.com/Seanny123/10452462
"""
range_inc = lambda start, end: range(start, end + 1)  # Because this is easier to write and read


def create_log(sigma, size=7):
    w = math.ceil(float(size) * float(sigma))

    # If the dimension is an even number, make it uneven
    if (w % 2 == 0):
        w = w + 1

    # Now make the mask
    l_o_g_mask = []

    w_range = int(math.floor(w / 2))
    # 	print("Going from " + str(-w_range) + " to " + str(w_range))
    for i in range_inc(-w_range, w_range):
        for j in range_inc(-w_range, w_range):
            l_o_g_mask.append(l_o_g(i, j, sigma))
    l_o_g_mask = np.array(l_o_g_mask)
    l_o_g_mask = l_o_g_mask.reshape(w, w)
    return l_o_g_mask


def l_o_g(x, y, sigma):
    # Formatted this way for readability
    nom = ((y ** 2) + (x ** 2) - 2 * (sigma ** 2))
    denom = ((2 * math.pi * (sigma ** 6)))
    expo = math.exp(-((x ** 2) + (y ** 2)) / (2 * (sigma ** 2)))
    return nom * expo / denom


class LaplacianOfGaussianFiltering:
    def __init__(self, size=3, sigma=1.0, input_channel=3, normalization=True,
                 identity_prob=0.0, device='cpu'):
        super(LaplacianOfGaussianFiltering, self).__init__()

        _kernel = torch.from_numpy(create_log(sigma, size=size)).unsqueeze(0).unsqueeze(0).float()
        self.kernel = _kernel.repeat(input_channel, 1, 1, 1).to(device)
        self.device = device
        self.size = size
        self.normalization = normalization
        self.identity_prob = identity_prob
        self.input_channel = input_channel

    def __call__(self, input):
        if (self.identity_prob > 0 and torch.rand(1) > self.identity_prob):
            return input
        output = F.conv2d(input.unsqueeze(0).to(self.device), self.kernel, groups=self.input_channel, padding=self.size // 2)
        if self.normalization:
            output = (output - output.mean(dim=(1, 2, 3))) / output.std(dim=(1, 2, 3))
        return output.squeeze(0).detach().cpu()
class ToBinary(object):
    """Convert multiclass label to binary
    """

    def __init__(self, target_id):
        self.target_id = target_id

    def __call__(self, target):
        return (target == self.target_id).long()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class GreyToColor(object):
    """Convert Grey Image label to binary
    """

    def __call__(self, image):
        if len(image.size()) == 3 and image.size(0) == 1:
            return image.repeat([3, 1, 1])
        elif len(image.size())== 2:
            return
        else:
            return image

    def __repr__(self):
        return self.__class__.__name__ + '()'

class IdentityTransform():
    """do nothing"""

    def __call__(self, image):
        return image



class ToGrayScale():
    def __init__(self, outout_channel=3):
        self.output_channel = outout_channel

    def __call__(self, image):
        return to_grayscale(image, self.output_channel)


class AddGeometricPattern:
    def __init__(self, intensity, shape='rect', num=20, region='bg', maxsize=2):
        self.shape = shape
        self.num = num
        self.region = region
        self.intensity = intensity
        self.maxsize = maxsize

    def __call__(self, image):
        if self.shape is None:
            return image

        pattern = Image.new('L', image.size, 0)
        draw = ImageDraw.Draw(pattern)
        img_size = image.size
        size_min = 2
        # size_max = int(np.asarray(image).shape[0] * 0.07)
        size_max = self.maxsize
        # print(size_min, size_max, img_size)

        for i in range(self.num):
            bbox_size = random.randint(size_min, size_max)
            loc_x = random.randint(0, img_size[0] - bbox_size - 1)
            loc_y = random.randint(0, img_size[1] - bbox_size - 1)
            if self.shape == 'rect':
                draw.rectangle((loc_x, loc_y, loc_x + bbox_size, loc_y + bbox_size),
                               outline=(self.intensity))
            elif self.shape == 'cross':
                # draw.line((loc_x, loc_y + bbox_size // 2, loc_x + bbox_size, loc_y + bbox_size // 2), fill=self.intensity, width=1)
                # draw.line((loc_x + bbox_size // 2, loc_y, loc_x + bbox_size // 2, loc_y + bbox_size), fill=self.intensity, width=1)
                draw.line((loc_x, loc_y, loc_x + bbox_size, loc_y + bbox_size), fill=self.intensity, width=1)
                draw.line((loc_x + bbox_size, loc_y, loc_x, loc_y + bbox_size), fill=self.intensity, width=1)
            elif self.shape == 'corner_tl':
                draw.line((loc_x, loc_y, loc_x, loc_y + bbox_size), fill=self.intensity, width=1)
                draw.line((loc_x, loc_y, loc_x + bbox_size, loc_y), fill=self.intensity, width=1)
            elif self.shape == 'corner_br':
                draw.line((loc_x+ bbox_size, loc_y, loc_x, loc_y), fill=self.intensity, width=1)
                draw.line((loc_x+ bbox_size, loc_y, loc_x + bbox_size, loc_y+ bbox_size), fill=self.intensity, width=1)
            elif self.shape == 'line_up':
                draw.line((loc_x + bbox_size, loc_y, loc_x, loc_y + bbox_size), fill=self.intensity, width=1)
            elif self.shape == 'line_down':
                draw.line((loc_x, loc_y, loc_x + bbox_size, loc_y + bbox_size), fill=self.intensity, width=1)

            else:
                raise NotImplementedError("Pattern {} is not implemented".format(self.shape))
        # draw.rectangle((20, 20, 22, 22), fill=None, outline=(128))
        img_arr = np.asarray(image)
        pattern_arr = np.asarray(pattern)
        fg_mask = img_arr > 0
        pattern_mask = pattern_arr/255.0

        if self.region == 'bg':
            img = Image.fromarray(np.uint8(img_arr*fg_mask+(1-fg_mask)*pattern_arr))
        elif self.region == 'fg':
            img = Image.fromarray(np.uint8(img_arr * (1 - pattern_mask)))
            # img = Image.fromarray(np.uint8(img_arr - pattern_arr*fg_mask))
            # img = Image.fromarray(np.uint8(img_arr + pattern_arr * (1 - fg_mask)))

        return img
class SketchModelDatasets(Dataset):
    def __init__(self,type):
        data_dic,cate_dic = self.read_json(type)
        self.cate_dic = cate_dic
        self.data_path_list = data_dic['data_list']#json中所有的文件路径名
        self.tr = self.get_transforms()
    def __getitem__(self, item):
        tr = transforms.Compose([

            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


        ])
        query_img_path = self.data_path_list[item]['query_img']
        render_imgs_path = self.data_path_list[item]['render_img']
        #类别标签
        label_cat = self.data_path_list[item]['category']
        #转成数字
        label_cat = self.cate_dic[label_cat]
        label_cat = torch.tensor(label_cat)

        #得到查询图像
        query_img = Image.open(query_img_path).convert('RGB')
        query_img = self.tr(query_img)

        render_imgs = []
        for i, render_img_path in enumerate(render_imgs_path):
            render_img = Image.open(render_img_path).convert('RGB')
            render_img = tr(render_img)
            render_imgs.append(render_img)
        render_imgs = torch.concat(render_imgs,dim=0)
        return {'render_img':render_imgs,'query_img':query_img,'label_cat':label_cat}
    def __len__(self):
        return len(self.data_path_list)
    def read_json(self,type):
        with open('../data_dict.json', 'r') as json_file:
            category_dict = json.load(json_file)
        if type == 'train':
            with open('../data_train_dict.json', 'r') as json_file:
                data_dic = json.load(json_file)
        if type == 'test':
            with open('../data_test_dict.json', 'r') as json_file:
                data_dic = json.load(json_file)
        return data_dic, category_dict
    def get_transforms(self):
        transform = transforms.Compose([
            transforms.Resize(arg.img_size),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
            if arg.color_jitter else IdentityTransform(),
            ToGrayScale(3) if arg.grey else IdentityTransform(),
            transforms.ToTensor(),
            GreyToColor(),
            LaplacianOfGaussianFiltering(size=3, sigma=1.0, identity_prob=0.5) if arg.LoG else IdentityTransform(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

        ])
        return transform
def get_transforms_test_train():
    transform = transforms.Compose([
        transforms.Resize(arg.img_size),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        if arg.color_jitter else IdentityTransform(),
        ToGrayScale(3) if arg.grey else IdentityTransform(),
        transforms.ToTensor(),
        GreyToColor(),
        LaplacianOfGaussianFiltering(size=3, sigma=1.0, identity_prob=0.5) if arg.LoG else IdentityTransform(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

    ])
    return transform
def load_model_query_datasets(type):
    batch_size = arg.batch_train_size
    shuffle = arg.shuffle
    datasets = SketchModelDatasets(type)
    datas = DataLoader(datasets,batch_size,shuffle=shuffle,drop_last=True)
    return datas
