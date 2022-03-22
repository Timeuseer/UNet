#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Author : Gabriel
@About : 
'''
import colorsys
import copy
import time
import yaml
import numpy as np
import os
from PIL import Image
from nets.unet import Unet as unet


class Unet(object):
    def __init__(self):
        self.config = yaml.load(open('./config/unet_config.yaml',encoding='utf-8'))
        self.__dict__.update(self.config)
        self.generate()

    def generate(self):
        self.model = unet(self.model_img_size, self.num_classes)
        '''
        加载预训练权重
        '''
        if self.model_path != '' and os.path.exists(self.model_path):
            self.model.load_weights(self.model_path)
            print(f'{self.model_path} loaded...')

        '''
        设置颜色
        '''
        hsv_tuples = [(x / self.num_classes, 1., 1.)
                      for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    def letterbox_img(self, img, size):
        img = img.convert('RGB')
        iw, ih = img.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        img = img.resize((nw, nh), Image.BICUBIC)
        new_img = Image.new('RGB', size, (128, 128, 128))
        new_img.paste(img, ((w - nw) // 2, (h - nh) // 2))

        return new_img, nw, nh

    def detect_img(self, img):
        '''
        将图像转换成RGB
        '''
        img = img.convert('RGB')

        '''
        对原图像备份
        '''
        old_img = copy.deepcopy(img)
        original_h = np.array(img).shape[0]
        original_w = np.array(img).shape[1]

        '''
        对图像进行预处理
        '''
        img, nw, nh = self.letterbox_img(img, (self.model_img_size[1], self.model_img_size[0]))
        img = np.asarray([np.array(img) / 255])

        '''
        输入图片
        '''
        pr = self.model.predict(img)[0]

        '''
        取出每一个像素点的分类
        '''
        pr = pr.argmax(axis=-1).reshape([self.model_img_size[0], self.model_img_size[1]])

        '''
        去掉预处理时增加的灰度条
        '''
        pr = pr[int((self.model_img_size[0] - nh) // 2):int((self.model_img_size[0] - nh) // 2 + nh),
             int((self.model_img_size[1] - nw) // 2):int((self.model_img_size[1] - nw) // 2 + nw)]

        '''
        绘制新图
        '''
        seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr[:, :] == c) * (self.colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((pr[:, :] == c) * (self.colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((pr[:, :] == c) * (self.colors[c][2])).astype('uint8')

        image = Image.fromarray(np.uint8(seg_img)).resize((original_w, original_h), Image.NEAREST)

        '''
        将预测结果与原图片混合
        '''
        if self.blend:
            image = Image.blend(old_img, image, 0.7)

        return image

    def get_FPS(self, image, test_interval):
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        img, nw, nh = self.letterbox_image(image, (self.model_img_size[1], self.model_img_size[0]))
        img = np.asarray([np.array(img) / 255])

        pr = self.model.predict(img)[0]
        pr = pr.argmax(axis=-1).reshape([self.model_img_size[0], self.model_img_size[1]])
        pr = pr[int((self.model_img_size[0] - nh) // 2):int((self.model_img_size[0] - nh) // 2 + nh),
             int((self.model_img_size[1] - nw) // 2):int((self.model_img_size[1] - nw) // 2 + nw)]

        image = Image.fromarray(np.uint8(pr)).resize((orininal_w, orininal_h), Image.NEAREST)

        t1 = time.time()
        for _ in range(test_interval):
            pr = self.model.predict(img)[0]
            pr = pr.argmax(axis=-1).reshape([self.model_img_size[0], self.model_img_size[1]])
            pr = pr[int((self.model_img_size[0] - nh) // 2):int((self.model_img_size[0] - nh) // 2 + nh),
                 int((self.model_img_size[1] - nw) // 2):int((self.model_img_size[1] - nw) // 2 + nw)]
            image = Image.fromarray(np.uint8(pr)).resize((orininal_w, orininal_h), Image.NEAREST)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
