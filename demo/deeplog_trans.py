#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys

sys.path.append('../')

from logdeep.models.transformers import deeplog, loganomaly, robustlog
from logdeep.tools.predict import Predicter
from logdeep.tools.train import Trainer
from logdeep.tools.utils import *

# Config Parameters

options = dict()
options['data_dir'] = '../data/'
options['window_size'] = 10
options['device'] = "cuda"

# Smaple
options['sample'] = "sliding_window"
options['window_size'] = 10  # if fix_window

# Features
options['sequentials'] = True
options['quantitatives'] = False
options['semantics'] = False
options['feature_num'] = sum(
    [options['sequentials'], options['quantitatives'], options['semantics']])

# Model
options['input_size'] = 1
options['hidden_size'] = 64
options['num_layers'] = 2
options['num_classes'] = 28

# Train
options['batch_size'] = 2048
options['accumulation_step'] = 1

options['optimizer'] = 'adam'
options['lr'] = 0.01
options['max_epoch'] = 370
options['lr_step'] = (300, 350)
options['lr_decay_ratio'] = 0.1

options['resume_path'] = None
options['model_name'] = "deeplog"
options['save_dir'] = "../result/deeplog/"

# Predict
options['model_path'] = "../result/deeplog/deeplog_last.pth"
options['num_candidates'] = 9

seed_everything(seed=1234)


def train():
    Model = deeplog(input_size=options['input_size'],
                    hidden_size=options['hidden_size'],
                    num_layers=options['num_layers'],
                    num_keys=options['num_classes'])
    trainer = Trainer(Model, options)
    trainer.start_train()


def predict():
    Model = deeplog(input_size=options['input_size'],
                    hidden_size=options['hidden_size'],
                    num_layers=options['num_layers'],
                    num_keys=options['num_classes'])
    predicter = Predicter(Model, options)
    return predicter.predict_unsupervised()


def get_times_mean(times):
    sum_P = 0.0
    sum_R = 0.0
    sum_F1 = 0.0
    for i in range(1, times):
        train()
        cur_P, cur_R, cur_F1 = predict()
        print(
            'current_P : {:.3f}%, current_R : {:.3f}%, current_F1 : {:.3f}%, current_lr: {:.5f}'
                .format(cur_P, cur_R, cur_F1, options['lr']))
        sum_P += cur_P
        sum_R += cur_R
        sum_F1 += cur_F1
    times = times - 1
    print('mean_P : {:.3f}%, mean_R : {:.3f}%, mean_F1 : {:.3f}%'.format(sum_P / times, sum_R / times, sum_F1 / times))
    return sum_P / times, sum_R / times, sum_F1 / times


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # print('Task working now!')
    # parser.add_argument('mode', choices=['train', 'predict'])

    # ----------------------------------
    # train()
    # P, R, F1 = predict()
    # ----------------------------------
    # args = parser.parse_args()

    for j in range(1, 10):
        get_times_mean(10)

    # best_P = 0.0
    # best_R = 0.0
    # best_F1 = 0.0
    #
    # best_P_lr = 0.0
    # best_R_lr = 0.0
    # best_F1_lr = 0.0
    #
    # options['lr'] = 0.001
    # for i in range(1, 100):
    #     print('current lr is :{:.5f}%'.format(options['lr']))
    #     train()
    #     P, R, F1 = predict()
    #     print(
    #         'current_P : {:.3f}%, current_R : {:.3f}%, current_F1 : {:.3f}%, current_lr: {:.5f}'
    #             .format(P, R, F1, options['lr']))
    #     if P > best_P:
    #         best_P = P
    #         best_P_lr = options['lr']
    #     if R > best_R:
    #         best_R = R
    #         best_R_lr = options['lr']
    #     if F1 > best_F1:
    #         best_F1 = F1
    #         best_F1_lr = options['lr']
    #     options['lr'] += 0.0001
    #
    # print(
    #     'best_P : {:.3f}%, best_R : {:.3f}%, best_F1: {:.3f}%, best_P_lr: {:.5f}, best_R_lr: {:.5f}, '
    #     'best_F1_lr-measure: {:.5f} '
    #         .format(best_P, best_R, best_F1, best_P_lr, best_R_lr, best_F1_lr))
    # ----------------------------------
    # mode = 'predict'
    # if mode == 'train':
    #     train()
    # else:
    #     predict()
