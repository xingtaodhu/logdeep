#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
sys.path.append('../')

from logdeep.models.lstm import deeplog, loganomaly, robustlog
from logdeep.tools.predict import Predicter
from logdeep.tools.train import Trainer
from logdeep.tools.utils import *


# Config Parameters

options = dict()
options['data_dir'] = '../data/'
options['window_size'] = 10
options['device'] = "cuda"

# Smaple
options['sample'] = "session_window"
options['window_size'] = -1

# Features
options['sequentials'] = False
options['quantitatives'] = False
options['semantics'] = True
options['feature_num'] = sum(
    [options['sequentials'], options['quantitatives'], options['semantics']])

# Model
options['input_size'] = 300
options['hidden_size'] = 128
options['num_layers'] = 2
options['num_classes'] = 2

# Train
options['batch_size'] = 256
options['accumulation_step'] = 1

options['optimizer'] = 'adam'
options['lr'] = 0.001
options['max_epoch'] = 60
options['lr_step'] = (40, 50)
options['lr_decay_ratio'] = 0.1

options['resume_path'] = None
options['model_name'] = "robustlog"
options['save_dir'] = "../result/robustlog/"

# Predict
options['model_path'] = "../result/robustlog/robustlog_last.pth"
options['num_candidates'] = -1

seed_everything(seed=1234)


def train():
    Model = robustlog(input_size=options['input_size'],
                      hidden_size=options['hidden_size'],
                      num_layers=options['num_layers'],
                      num_keys=options['num_classes'])
    trainer = Trainer(Model, options)
    trainer.start_train()


def predict():
    Model = robustlog(input_size=options['input_size'],
                      hidden_size=options['hidden_size'],
                      num_layers=options['num_layers'],
                      num_keys=options['num_classes'])
    predicter = Predicter(Model, options)
    return predicter.predict_supervised()


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

    get_times_mean(10)


    # parser = argparse.ArgumentParser()
    # parser.add_argument('mode', choices=['train', 'predict'])
    # args = parser.parse_args()
    # if args.mode == 'train':
    #     train()
    # else:
    #     predict()
