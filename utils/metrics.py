#! /usr/bin/python3
# -*- coding:utf-8 -*-

'''
@Author : Gabriel
@About : 
'''
import tensorflow as tf
from tensorflow.keras import backend as K


def Iou_score(smooth=1e-5, threshold=0.5):
    def _Iou_score(y_true, y_pred):
        y_pred = K.greater(y_pred, threshold)
        y_pred = K.cast(y_pred, K.floatx())
        intersection = K.sum(y_true[..., :-1] * y_pred, axis=[0, 1, 2])
        union = K.sum(y_true[..., :-1] + y_pred, axis=[0, 1, 2]) - intersection

        score = (intersection + smooth) / (union + smooth)
        score = tf.reduce_mean(score)
        return score

    return _Iou_score


def f_score(beta=1, smooth=1e-5, threshold=0.5):
    def _f_score(y_true, y_pred):
        y_pred = K.greater(y_pred, threshold)
        y_pred = K.cast(y_pred, K.floatx())

        tp = K.sum(y_true[..., :-1] * y_pred, axis=[0, 1, 2])
        fp = K.sum(y_pred, axis=[0, 1, 2]) - tp
        fn = K.sum(y_true[..., :-1], axis=[0, 1, 2]) - tp

        score = ((1 + beta ** 2) * tp + smooth) \
                / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        score = tf.reduce_mean(score)
        return score

    return _f_score
