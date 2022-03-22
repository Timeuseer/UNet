#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Author : Gabriel
@About : backbone
'''
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal


def VGG16(img_input):
    nets = {}
    '''
    Block 1
    [512,512,3]->[512,512,64]
    '''
    nets['block1_conv1'] = layers.Conv2D(64, (3, 3), activation='relu',
                                         padding='same', kernel_initializer=RandomNormal(stddev=0.02),
                                         name='block1_conv1')(img_input)
    nets['block1_conv2'] = layers.Conv2D(64, (3, 3), activation='relu',
                                         padding='same', kernel_initializer=RandomNormal(stddev=0.02),
                                         name='block1_conv2')(nets['block1_conv1'])

    feat1 = nets['block1_conv2']

    # [512,512,64]->[256,256,64]
    nets['block1_pool'] = layers.MaxPool2D((2, 2), strides=(2, 2), name='block1_pool')(nets['block1_conv2'])

    '''
    Block 2
    [256,256,64]->[256,256,128]
    '''
    nets['block2_conv1'] = layers.Conv2D(128, (3, 3), activation='relu',
                                         padding='same', kernel_initializer=RandomNormal(stddev=0.02),
                                         name='block2_conv1')(nets['block1_pool'])
    nets['block2_conv2'] = layers.Conv2D(128, (3, 3), activation='relu',
                                         padding='same', kernel_initializer=RandomNormal(stddev=0.02),
                                         name='block2_conv2')(nets['block2_conv1'])

    feat2 = nets['block2_conv2']

    # [256,256,128]->[128,128,128]
    nets['block2_pool'] = layers.MaxPool2D((2, 2), strides=(2, 2), name='block2_pool')(nets['block2_conv2'])

    '''
    Block 3
    [128,128,128]->[128,128,256]
    '''
    nets['block3_conv1'] = layers.Conv2D(256, (3, 3), activation='relu',
                                         padding='same', kernel_initializer=RandomNormal(stddev=0.02),
                                         name='block3_conv1')(nets['block2_pool'])
    nets['block3_conv2'] = layers.Conv2D(256, (3, 3), activation='relu',
                                         padding='same', kernel_initializer=RandomNormal(stddev=0.02),
                                         name='block3_conv2')(nets['block3_conv1'])
    nets['block3_conv3'] = layers.Conv2D(256, (3, 3), activation='relu',
                                         padding='same', kernel_initializer=RandomNormal(stddev=0.02),
                                         name='block3_conv3')(nets['block3_conv2'])

    feat3 = nets['block3_conv3']

    # [128,128,256]->[64,64,256]
    nets['block3_pool'] = layers.MaxPool2D((2, 2), strides=(2, 2), name='block3_pool')(nets['block3_conv3'])

    '''
    Block 4
    [64,64,256]->[64,64,512]
    '''
    nets['block4_conv1'] = layers.Conv2D(512, (3, 3), activation='relu',
                                         padding='same', kernel_initializer=RandomNormal(stddev=0.02),
                                         name='block4_conv1')(nets['block3_pool'])
    nets['block4_conv2'] = layers.Conv2D(512, (3, 3), activation='relu',
                                         padding='same', kernel_initializer=RandomNormal(stddev=0.02),
                                         name='block4_conv2')(nets['block4_conv1'])
    nets['block4_conv3'] = layers.Conv2D(512, (3, 3), activation='relu',
                                         padding='same', kernel_initializer=RandomNormal(stddev=0.02),
                                         name='block4_conv3')(nets['block4_conv2'])

    feat4 = nets['block4_conv3']

    # [64,64,512]->[32,32,512]
    nets['block4_pool'] = layers.MaxPool2D((2, 2), strides=(2, 2), name='block4_pool')(nets['block4_conv3'])

    '''
    Block 5
    [32,32,512]->[32,32,512]
    '''
    nets['block5_conv1'] = layers.Conv2D(512, (3, 3), activation='relu',
                                         padding='same', kernel_initializer=RandomNormal(stddev=0.02),
                                         name='block5_conv1')(nets['block4_pool'])
    nets['block5_conv2'] = layers.Conv2D(512, (3, 3), activation='relu',
                                         padding='same', kernel_initializer=RandomNormal(stddev=0.02),
                                         name='block5_conv2')(nets['block5_conv1'])
    nets['block5_conv3'] = layers.Conv2D(512, (3, 3), activation='relu',
                                         padding='same', kernel_initializer=RandomNormal(stddev=0.02),
                                         name='block5_conv3')(nets['block5_conv2'])

    feat5 = nets['block5_conv3']

    return feat1, feat2, feat3, feat4, feat5
