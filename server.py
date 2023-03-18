# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 10:30:22 2022

@author: admin
"""
import torch, os
from torch.utils.data import DataLoader
import numpy as np
from models.aggregation_method import WeightedAvg, FedTrimmedMean, FedMedian
from models.aggregation_method import Krum, K_norm
from utils.text_helper import repackage_hidden
import torch.nn.functional as F
from threading import Thread
import time, random, copy
from operator import itemgetter
from models.Nets import  CNN, CNNCifar, RNNModel

class Server(Thread):
    def __init__(self, args, no=0, task=None):
        super(Server,self).__init__()  
        self.device = torch.device(
            'cuda:{}'.format(int(np.random.choice(range(torch.cuda.device_count()),1))) \
                if torch.cuda.is_available() else 'cpu'
                )
        self.tc = task
        self.no = no
        self.global_model_path = './save/{}{}-{}_seed{}_w{}v{}_E{}e{}b{}lr{}_{}-C{}/'.format(
                args.dataset[:5], args.data_allocation_scheme, 
                args.model, args.manual_seed, 
                args.num_clients, args.num_servers,
                args.maximum_round_index, args.local_ep, args.local_bs, args.lr,  
                args.protocol,
                args.frac
                )
        self.version_tag = 0
        
        if self.tc.args.model.lower() == 'cnn' and 'mnist' in self.tc.args.dataset:
            self.net = CNN(args=self.tc.args)
        elif self.tc.args.model.lower() == 'cnn' and self.tc.args.dataset == 'cifar':
            self.net = CNNCifar(args=self.tc.args)
        elif self.tc.args.model.lower() == 'lstm' and self.tc.args.dataset == 'shakespeare':
            self.net = RNNModel(args=self.tc.args)
        else:
            exit('Error: unrecognized model')
        print(self.net)
        self.SaveGlobalModel(self.net.state_dict())
        
        self.workerKeys = range(args.num_clients)
        
    
    def run(self):
        # keep checking task status
        while True:
            if len(self.tc.cached_local_update)==self.tc.args.num_clients*self.tc.args.frac:
                torch.cuda.empty_cache()
                self.workflow()
                torch.cuda.empty_cache()
                self.tc.cached_local_update =dict()
                self.tc.active_worker = np.random.choice(
                    range(self.tc.args.num_clients),
                    int(self.tc.args.num_clients*self.tc.args.frac),
                    replace= False)
                self.version_tag += 1     
                print("new round %s..."%self.version_tag)
            else:
                time.sleep(random.random()*5)
                
            
    def workflow(self):
        ######################## collect info #######################
        
        w_glob_old = torch.load(
            self.global_model_path +'ServerAggResult(Version%s).pt'%(self.version_tag),
            map_location=self.device
            )
        w_locals = dict()
        for idx in self.tc.cached_local_update.keys():
            w_locals[idx] = torch.load(
                        self.global_model_path + self.tc.cached_local_update[idx], 
                        map_location=self.device)
        print('len(w_locals): ', len(w_locals))
        
        ######################## aggregation #######################
        p_k = dict([(idx, self.tc.clients[idx].no) \
                    for idx in self.tc.cached_local_update.keys()])
        w_glob  = self.aggregation(w_glob_old, w_locals, p_k)       
        torch.save(
            w_glob, 
            self.global_model_path+'ServerAggResult(Version%s).pt'%(self.version_tag+1)
            )
        self.validate(w_glob)
        print("workflow done!")
        return 
    
    
    def SaveGlobalModel(self, net_glob): #.state_dict()
        if os.path.exists('./save/') == False:
            os.mkdir('./save/')
        if os.path.exists(self.global_model_path):
           torch.save(
               net_glob, 
               self.global_model_path+"ServerAggResult(Version%s).pt"%self.version_tag
               )
        else:
            os.mkdir(self.global_model_path)
            torch.save(
                net_glob, 
                self.global_model_path+"ServerAggResult(Version%s).pt"%self.version_tag
                )
        print("save model to: {}".format(self.global_model_path+"ServerAggResult(Version%s).pt"%self.version_tag))
        
        return
    
    def aggregation(self, w_glob, w_locals, p_k):
        net_glob = WeightedAvg(w_locals, p_k)
        return net_glob
    
    def validate(self, w_glob=None):
        self.ldr_eval = DataLoader(torch.load('data/%s%s/test.pt'%(
            self.tc.args.dataset, self.tc.args.data_allocation_scheme), 
            map_location=self.device), 
            batch_size=self.tc.args.test_bs, 
            shuffle=False)       
        self.net.load_state_dict(w_glob)
        self.net = self.net.to(self.device)
        self.net.eval()
        
        batch_loss = []  
        if self.tc.args.dataset == 'shakespeare':
            # initialize hidden state
            hidden = self.net.init_hidden(self.tc.args.test_bs) 
        last_append = 0
        data_size=0
        correct = 0
        for batch_idx, batch in enumerate(self.ldr_eval):
            data = batch[0].to(self.device)
            labels = batch[1].to(self.device)
            if self.tc.args.dataset == 'shakespeare':
                if data.shape[0] < self.tc.args.test_bs:
                    last_append = data.shape[0]
                    for _ in range(int(np.ceil((self.tc.args.test_bs - \
                                                data.shape[0])/data.shape[0]))):
                        if data.shape[0] * 2 < self.tc.args.test_bs:
                            data = torch.cat((data, data))
                            
                        else:
                            data = torch.cat((data,
                                              data[: (self.tc.args.test_bs - data.shape[0]),:]))
                data = data.t() 
                labels = labels.t().reshape(-1)
                # global forward
                hidden = repackage_hidden(hidden)
                net_outputs, hidden = self.net(data, hidden)
                # format transformation
                if last_append > 0:
                    net_outputs = net_outputs[-1 ,:last_append,:] 
                else:
                    net_outputs = net_outputs[-1: ,:,:]
                net_outputs = net_outputs.reshape(-1, self.tc.args.ntoken)
            else:
                # forward
                net_outputs = self.net(data)
                        
            loss = F.cross_entropy(net_outputs, labels, reduction='sum').item()
            y_pred = net_outputs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
            batch_loss.append(loss)
            data_size += len(labels)
        del self.ldr_eval, net_outputs, data, labels, loss
        if self.tc.args.dataset == 'shakespeare':
            del hidden
        torch.cuda.empty_cache()
        accuracy= float(correct.item() / data_size)*100
        print('Round {} Test set: Accuracy: {}/{} ({:.2f}%)'.format(
             self.version_tag, correct.item(), data_size, accuracy))
        
        return sum(batch_loss) / data_size, accuracy
    
    