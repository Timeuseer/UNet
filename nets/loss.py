#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Author : Gabriel
@About : 
'''

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import tensorflow as tf

from random import shuffle
from PIL import Image
from tensorflow import keras
from tensorflow.keras import backend as K


def dice_loss_with_ce(beta=1, smooth=1e-5):
    def _dice_loss_with_ce(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        ce_loss = - y_true[..., :-1] * K.log(y_pred)
        ce_loss = K.mean(K.sum(ce_loss, axis=-1))

        tp = K.sum(y_true[..., :-1] * y_pred, axis=[0, 1, 2])
        fp = K.sum(y_pred, axis=[0, 1, 2]) - tp
        fn = K.sum(y_true[..., :-1], axis=[0, 1, 2]) - tp

        score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        score = tf.reduce_mean(score)
        dice_loss = 1 - score

        return ce_loss + dice_loss

    return _dice_loss_with_ce


def ce():
    def _ce(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        ce_loss = - y_true[..., :-1] * K.log(y_pred)
        ce_loss = K.mean(K.sum(ce_loss, axis=-1))

        return ce_loss

    return _ce


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def letterbox_img(img, label, size):
    # 使用在周围填充灰框的方式，调整图片尺寸
    label = Image.fromarray(np.array(label))
    iw, ih = img.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    img = img.resize((nw, nh), Image.BICUBIC)
    new_img = Image.new('RGB', size, (128, 128, 128))
    new_img.paste(img, ((w - nw) // 2, (h - nh) // 2))

    label = label.resize((nw, nh), Image.NEAREST)
    new_label = Image.new('L', size, (0))
    new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))

    return new_img, new_label


class Generator(object):
    def __init__(self, batch_size, train_lines, img_size, num_classes, dataset_path):
        self.batch_size = batch_size
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.img_size = img_size
        self.num_classes = num_classes
        self.dataset_path = dataset_path

    def get_random_data(self, img, label, input_shape, jitter=0.3, hue=0.1, sat=1.5, val=1.5):
        label = Image.fromarray(np.array(label))

        h, w = input_shape
        # Resize
        rand_jit1 = rand(1 - jitter, 1 + jitter)
        rand_jit2 = rand(1 - jitter, 1 + jitter)
        new_ar = w / h * rand_jit1 / rand_jit2

        scale = rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw * new_ar)

        img = img.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.BICUBIC)
        label = label.convert("L")

        # Flip
        flip = rand() < 0.5
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Padding
        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_img = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new("L", (w, h), (0))
        new_img.paste(img, (dx, dy))
        new_label.paste(label, (dx, dy))
        img = new_img
        label = new_label

        # Distort
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < 0.5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < 0.5 else 1 / rand(1, val)
        x = cv2.cvtColor(np.array(img, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1::][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        img_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        return img_data, label

    def generate(self, random_data=True):
        i = 0
        length = len(self.train_lines)
        inputs = []
        targets = []
        while True:
            if i == 0:
                shuffle(self.train_lines)
            annotation_line = self.train_lines[i]
            name = annotation_line.split()[0]

            # 获取图像
            jpg = Image.open(os.path.join(os.path.join(self.dataset_path, 'JPEGImages'), name + '.jpg'))
            png = Image.open(os.path.join(os.path.join(self.dataset_path, 'SegmentationClass'), name + '.png'))

            if random_data:
                jpg, png = self.get_random_data(jpg, png, (int(self.img_size[1]), int(self.img_size[0])))
            else:
                jpg, png = letterbox_img(jpg, png, (int(self.img_size[1]), int(self.img_size[0])))

            inputs.append(np.array(jpg) / 255)

            png = np.array(png)
            png[png >= self.num_classes] = self.num_classes

            '''
            进行one_hot
            self.num_classes的原因是因为VOC等部分数据集的标签有白色部分，
            需要忽略掉白色部分
            '''
            seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
            seg_labels = seg_labels.reshape((int(self.img_size[1]), int(self.img_size[0]), self.num_classes + 1))

            targets.append(seg_labels)
            i = (i + 1) % length
            if len(targets) == self.batch_size:
                tmp_inp = np.array(inputs)
                tmp_target = np.array(targets)
                inputs = []
                targets = []
                yield tmp_inp, tmp_target



class LossHistory(keras.callbacks.Callback):
    def __init__(self, log_dir):
        import datetime
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time, '%Y%m%d-%H%M%S')
        self.log_dir = log_dir
        self.time_str = time_str
        self.save_path = os.path.join(self.log_dir, "loss_" + str(self.time_str))
        self.losses = []
        self.val_loss = []

        os.makedirs(self.save_path)

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        with open(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('loss')))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('val_loss')))
            f.write("\n")
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('A Loss Curve')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))

        plt.cla()
        plt.close("all")