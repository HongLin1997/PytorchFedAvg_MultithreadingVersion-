# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 20:57:36 2021

@author: admin
"""
from utils.options import args_parser
from utils.sampling import data_split, text_data_split, image_data_split
from utils.dataset_helper import TextDataset, ImageDataset
from utils.tools import init_random_seed
import torch,os

#from models.Nets import  CNN, CNNCifar, RNNModel
#from torch import nn
#from torch.utils.data import DataLoader

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) \
                               if torch.cuda.is_available() and args.gpu != -1\
                               else 'cpu')
    # parse args
    print(args)
    if args.manual_seed:
        init_random_seed(args.manual_seed)
    
    # load dataset and split workers
    if args.dataset=='shakespeare':
        args.local_bs=32
        dataset_train, dataset_train_label, \
        dataset_validate, dataset_validate_label, \
        centralized_test_data, centralized_test_label, \
        dict_workers = text_data_split(args)
        trn_len = len(dataset_train_label)
        tst_len = len(centralized_test_label)
    elif args.dataset == 'femnist':
        dataset_train, dataset_train_label, \
        dataset_validate, dataset_validate_label, \
        centralized_test_data, centralized_test_label, \
        dict_workers = image_data_split(args)
        trn_len = len(dataset_train_label)
        tst_len = len(centralized_test_label)
    else:
        dataset_train, dataset_test, validation_index, \
        dict_workers = data_split(args)
        trn_len = len(dataset_train)
        tst_len = len(dataset_test)
    args.sum_training_data = sum([len(data_indexs) for data_indexs in dict_workers.values()])
    print('args.sum_training_data: ', args.sum_training_data, '\n')   
    
    try:
        val_len = len(dataset_validate_label)
    except:
        pass
    
    if os.path.exists('data/%s%s/'%(args.dataset, args.data_allocation_scheme))==False:
        os.makedirs('data/%s%s/'%(args.dataset, args.data_allocation_scheme))
    for idx in range(args.num_workers):
        filename = 'data/%s%s/worker%s.pt'%(args.dataset, args.data_allocation_scheme, idx)
        if args.dataset == 'shakespeare':
            trainset = TextDataset(
                    dataset_train, dataset_train_label, trn_len,
                    dict_workers[idx],
                    poison_sentences=args.poison_sentences, 
                    after_poison_sentences=args.after_poison_sentences
                    )
                
        elif args.dataset == 'cifar' :
            trainset = ImageDataset(
                    dataset_train, None, trn_len, dict_workers[idx],
                    )
        else:
            trainset = ImageDataset(
                    dataset_train, dataset_train_label, trn_len,
                    dict_workers[idx],
                    )
                
        #else:
        #    trainset = DatasetSplit(dataset, data_idxs)
            
        torch.save(trainset, filename)
        print('worker %s: data_size %s %s...'%(idx, len(dict_workers[idx]), 
                                            len(trainset)))
        
    '''        
    validation_dataset_path = 'data/%s%s/validation.pt'%(args.dataset, args.data_allocation_scheme)
    # load validation data
    if args.dataset == 'shakespeare':
        evalset = TextDataset(
                dataset_validate, dataset_validate_label, val_len,
                range(val_len),
                poison_sentences=[], 
                after_poison_sentences=[]
                )
    elif args.dataset == 'cifar' :
        evalset= ImageDataset(
                dataset_test, None, tst_len, validation_index,
                )            
    else :
        evalset = ImageDataset(
                dataset_validate, dataset_validate_label, val_len,
                range(val_len),
                )
    torch.save(evalset, validation_dataset_path)
    print('validation data_size %s...'%len(evalset))
    '''    
        
    test_dataset_path = 'data/%s%s/test.pt'%(args.dataset, args.data_allocation_scheme)
    # load validation data
    if args.dataset == 'shakespeare':
        testset = TextDataset(
                centralized_test_data, centralized_test_label, tst_len,
                range(len(centralized_test_label)),
                poison_sentences=[], 
                after_poison_sentences=[]
                )
                
    elif args.dataset == 'cifar' :
        testset= ImageDataset(
                dataset_test, None, tst_len,
                set(range(len(dataset_test)))-set(validation_index),
                #dataset_train, None, dict_workers[idx],
                )            
    else :
        testset = ImageDataset(
                centralized_test_data, centralized_test_label, tst_len,
                range(len(centralized_test_label)),
                )    
    torch.save(testset, test_dataset_path)
    print('test data_size %s...'%len(testset))