#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import numpy as np

def args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--manual_seed', type=int, default=42, help="manual_seed")
    
    # federated environment arguments
    parser.add_argument('--num_workers', type=int, default=100, help="number of workers: N (66 for Shakespeare)") #
    parser.add_argument('--num_validators', type=int, default=1, help="number of validators: N (66 for Shakespeare)") #
    
    
    # dataset and partition
    parser.add_argument('--data_folder', type=str, default='./data/shakespeare', help="data_folder of text")
    #parser.add_argument('--dictionary', type=str, default='./data/shakespeare/dictionary.pt', help="dictionary of text")
    parser.add_argument('--seq_len', type=int, default=80, help="seq_len for shakespeare")
    parser.add_argument('--train_frac', type=float, default=0.8, help="the fraction of training data")
    
    parser.add_argument('--femnist_data_folder', type=str, default='./data/femnist', help="data_folder of femnist")
    parser.add_argument('--syn_round_time', type=str, default='mobile', help="syn_round_time")
    parser.add_argument('--max_time_period', type=float, default=4320, help="max_time_period")
    
    
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--data_allocation_scheme', type=int, default=0, help="data_allocation_scheme") #
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    
    # learning arguments
    parser.add_argument('--maximum_round_index', type=int, default=100, help="maximum_round_index")
    parser.add_argument('--global_acc_threshold', type=int, default=100, help="global_acc_threshold")
    parser.add_argument('--max_iterations', type=int, default=None, help="max_iterations")
    parser.add_argument('--test_bs', type=int, default=128, help="test batch size")
    parser.add_argument('--bs', type=int, default=64, help="global train batch size")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0, help="SGD momentum (default: 0)")
    parser.add_argument('--weight_decay', type=float, default=0, help="SGD weight_decay (default: 0)")
    parser.add_argument('--decay_rate', type=float, default=1, help="decay_rate (default: 0.992; for shakespeare, default:1)")
    parser.add_argument('--max_norm', type=float, default=5.0, help='the maximum norm of gradient (default: 5.0)')
    
    # cnn model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    #parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of images")
    parser.add_argument('--gpu', type=int, default=-1, help="GPU ID, -1 for CPU")
    parser.add_argument('--cuda', type=str, default=None, help="")
    
    # lstm model arguments
    parser.add_argument('--ntoken', type=int, default=80, help="ntoken for lstm")
    parser.add_argument('--ninp', type=int, default=8, help="input dim for lstm")
    parser.add_argument('--nhid', type=int, default=256, help="hidden dim for lstm")
    parser.add_argument('--num_layers', type=int, default=2, help="num_layers for lstm")
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout dim for lstm")
    parser.add_argument('--tie_weights', type=bool, default=False, help="tie_weights for lstm")
    
    
    # device heterogeneity
    parser.add_argument('--local_ep', type=int, default=3, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    
    # other methods: fedavg, aggregation and fedprox, personalized FT, FB
    parser.add_argument('--frac', type=float, default=1.0, help="the fraction of clients: C")
    parser.add_argument('--protocol', type=str, default='FedAvg', help='protocol for model training in FL')
    parser.add_argument('--aggregation', type=str, default='WeightedAvg', help='aggregation method')
    
    args = parser.parse_args()
    return args
