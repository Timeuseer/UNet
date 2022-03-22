#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Author : Gabriel
@About : 加强特征网络
'''
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import layers
from tensorflow.keras import models

from nets.vgg16 import VGG16


def Unet(input_shape=(512, 512, 3), num_classes=21):
    inputs = layers.Input(input_shape)

    '''
    获得五个有效特征层
    feat1:  [512,512,64]
    feat2:  [256,256,128]
    feat3:  [128,128,256]
    feat4:  [64,64,512]
    feat5:  [32,32,512]
    '''

    feat1, feat2, feat3, feat4, feat5 = VGG16(inputs)
    channels = [64, 128, 256, 512]

    # [32,32,512]->[64,64,512]
    p5_up = layers.UpSampling2D(size=(2, 2))(feat5)
    # [64,64,512] + [64,64,512] -> [64,64,1024]
    p4 = layers.Concatenate(axis=3)([feat4, p5_up])
    # [64,64,1024] -> [64,64,512]
    p4 = layers.Conv2D(channels[3], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
        p4)
    p4 = layers.Conv2D(channels[3], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
        p4)

    # [64,64,512]->[128,128,512]
    p4_up = layers.UpSampling2D(size=(2, 2))(p4)
    # [128,128,256] + [128,128,512] -> [128,128,768]
    p3 = layers.Concatenate(axis=3)([feat3, p4_up])
    # [128,128,768] -> [128,128,256]
    p3 = layers.Conv2D(channels[2], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
        p3)
    p3 = layers.Conv2D(channels[2], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
        p3)

    # [128,128,256]->[256,256,256]
    p3_up = layers.UpSampling2D(size=(2, 2))(p3)
    # [256,256,256] + [256,256,128] -> [256,256,384]
    p2 = layers.Concatenate(axis=3)([feat2, p3_up])
    # [256,256,384] -> [256,256,128]
    p2 = layers.Conv2D(channels[1], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
        p2)
    p2 = layers.Conv2D(channels[1], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
        p2)

    # [256,256,128]->[512,512,128]
    p2_up = layers.UpSampling2D(size=(2, 2))(p2)
    # [512,512,128] + [512,512,64] -> [512,512,192]
    p1 = layers.Concatenate(axis=3)([feat1, p2_up])
    # [512,512,192] -> [512,512,64]
    p1 = layers.Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
        p1)
    p1 = layers.Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer=RandomNormal(stddev=0.02))(
        p1)

    # [512,512,64] -> [512,512,num_classes]
    p1 = layers.Conv2D(num_classes, 1, activation='softmax')(p1)

    model = models.Model(inputs=inputs, outputs=p1)

    return model
