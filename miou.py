#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Author : Gabriel
@About : 计算miou
'''

import os
import numpy as np
import itertools

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from unet import Unet


def fast_hist(a, b, n):
    '''
    设置标签宽和高
    a:  转化成1维数组的标签,(H x W)
    b:  转化成1维数组的预测结果,(HxW,)
    '''
    k = (a >= 0) & (a < n)
    '''
    np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    返回斜对角线上的分类正确的像素点
    '''
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)


def per_class_pa(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def compute_miou(gt_dir, pred_dir, png_name_list, num_classes, name_classes):
    print('Num classes', num_classes)

    '''
    创建全为0的混淆矩阵
    '''
    hist = np.zeros((num_classes, num_classes))

    '''
    获得验证集标签路径和图像分割结果路径
    '''
    gt_imgs = [os.path.join(gt_dir, x + ".png") for x in png_name_list]
    pred_imgs = [os.path.join(pred_dir, x + ".png") for x in png_name_list]

    for ind in range(len(gt_imgs)):
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))

        # 如果图像分割结果与标签的大小不一样，这张图片就不计算
        if len(label.flatten()) != len(pred.flatten()):
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        '''
        对一张图片计算21×21的hist矩阵，并累加
        '''
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        # 每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
        if ind > 0 and ind % 10 == 0:
            print(
                f'{ind:d} / {len(gt_imgs):d}: mIou-{100 * np.nanmean(per_class_iu(hist)):0.2f}; mPA-{100 * np.nanmean(per_class_pa(hist)):0.2f}')
    '''
    计算所有验证集图片的逐类别mIoU值
    '''
    mIoUs = per_class_iu(hist)
    mPA = per_class_pa(hist)
    plot_matrix(hist, name_classes, img_save_path, normalize=True, title='miou')
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\tmIou-' + str(round(mIoUs[ind_class] * 100, 2)) + '; mPA-' + str(
            round(mPA[ind_class] * 100, 2)))

    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)) + '; mPA: ' + str(round(np.nanmean(mPA) * 100, 2)))
    return mIoUs


def plot_matrix(cm, classes, save_path, normalize=False, title='Confusion matrix', cmap=plt.cm.get_cmap('Blues')):
    """
    绘制混淆矩阵
    """
    # plt.rc('font', family='sans-serif', size='4.5')
    # plt.figure(figsize=(20, 20))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("显示百分比：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print(cm)
    else:
        print('显示具体数字：')
        print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
    # plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.show()


class miou_Unet(Unet):
    def detect_image(self, image):
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        img, nw, nh = self.letterbox_img(image, (self.model_img_size[1], self.model_img_size[0]))
        img = np.asarray([np.array(img) / 255])

        pr = self.model.predict(img)[0]
        pr = pr.argmax(axis=-1).reshape([self.model_img_size[0], self.model_img_size[1]])
        pr = pr[int((self.model_img_size[0] - nh) // 2):int((self.model_img_size[0] - nh) // 2 + nh),
             int((self.model_img_size[1] - nw) // 2):int((self.model_img_size[1] - nw) // 2 + nw)]

        image = Image.fromarray(np.uint8(pr)).resize((orininal_w, orininal_h), Image.NEAREST)
        return image


def get_predict():
    unet = miou_Unet()

    image_ids = open("E:/011-Dataset/VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt", 'r').read().splitlines()

    if not os.path.exists("miou_pr_dir"):
        os.makedirs("miou_pr_dir")

    for image_id in tqdm(image_ids):
        image_path = "E:/011-Dataset/VOCdevkit/VOC2007/JPEGImages/" + image_id + ".jpg"
        image = Image.open(image_path)
        image = unet.detect_image(image)
        image.save("miou_pr_dir/" + image_id + ".png")


if __name__ == "__main__":
    get_predict()
    import datetime

    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Y%m%d-%H%M%S')
    gt_dir = "E:/011-Dataset/VOCdevkit/VOC2007/SegmentationClass"
    pred_dir = "E:/011-Dataset/VOCdevkit/VOC2007/SegmentationClass"
    png_name_list = open("E:/011-Dataset/VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt", 'r').read().splitlines()
    img_save_path = f'./logs/test_miou_{time_str}.jpg'
    # 需要加上背景
    num_classes = 21
    name_classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                    "tvmonitor"]
    compute_miou(gt_dir, pred_dir, png_name_list, num_classes, name_classes)  # 执行计算mIoU的函数
