# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 20:36:26 2021

@author: admin
"""

from utils.options import args_parser
from utils.tools import init_random_seed
from client import Client
from server import Server
import torch, time
import numpy as np

class Task(object):
    def __init__(self, args):
        self.args = args
        self.clients=dict()
        self.server = None
        self.cached_local_update = dict()
        self.active_worker = np.random.choice(range(self.args.num_clients),
                                              int(self.args.num_clients*self.args.frac),
                                              replace= False)
    def ReadWorkerList(self):
        return self.clients.keys()
    
    def ReadServer(self):
        return  self.server.no
    
    def RECRUITING(self, task):
        self.server = Server(args=self.args, task = task)
        print('initializing server finished...')
        
        ########################################################################
        
        for no in range(self.args.num_clients):
            self.clients[no] = Client(args=self.args, no=no, task = task)
        print('initializing client finished...')
        return
    
args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) \
                           if torch.cuda.is_available() and args.gpu != -1\
                           else 'cpu')
            
if __name__ == '__main__':
    # parse args
    print(args)
    if args.manual_seed:
        init_random_seed(args.manual_seed, args.cuda)
    initial_lr = args.lr
    
    #initialize server and clients
    tc = Task(args)
    tc.RECRUITING(tc)
    
    # start the learning
    tc.server.daemon = True
    tc.server.start()
    for t in tc.clients.values():
        t.daemon = True
        t.start()
          
    print('\nTraining starts!')
    curRound = 0
    while True:
        curRound = tc.server.version_tag
        if curRound >= args.maximum_round_index:
            print('algorithm terminated!')
            exit()
        time.sleep(60)
         
    exit()     