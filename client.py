import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import copy

from utils.text_helper import repackage_hidden
from threading import Thread
import time, random
all_gpus = range(torch.cuda.device_count())

class Client(Thread):
    def __init__(self, args, no=0, task=None):
        super(Client,self).__init__()  
        
        self.device = torch.device(
            'cuda:{}'.format(int(all_gpus[no%len(all_gpus)])) \
                if torch.cuda.is_available() else 'cpu'
                )
            
        self.tc = task
        self.no = no
        self.loss_func = nn.CrossEntropyLoss()#.to(self.device)
        self.filename = 'data/%s%s/worker%s.pt'%(
            args.dataset, args.data_allocation_scheme, no
            )
        self.global_model_path = './save/{}{}-{}_seed{}_w{}v{}_E{}e{}b{}lr{}_{}-C{}/'.format(
                args.dataset[:5], args.data_allocation_scheme, 
                args.model, args.manual_seed, 
                args.num_workers, args.num_validators,
                args.maximum_round_index, args.local_ep, args.local_bs, args.lr,  
                args.protocol,
                args.frac
                )
        self.version_tag = -1
        self.submission_count = 0
    
    def run(self):
        # keep checking the task status
        while True:
            if self.version_tag < self.tc.server.version_tag:
                try:
                    self.ldr_train,self.global_model = None, None
                except:
                    pass
                torch.cuda.empty_cache()
                print("worker %s (version %s) start training..."%(
                    self.no, self.tc.server.version_tag))
                self.version_tag = self.tc.server.version_tag
                self.train()
                
            else:
                time.sleep(random.random()*5)
                
                        
    def train(self):
        if type(self.ldr_train)==type(None):
            data_set = torch.load(self.filename, 
                                  map_location=self.device)
            self.ldr_train = DataLoader(data_set, 
                                        batch_size=self.tc.args.local_bs, 
                                        shuffle=True)
            del data_set
            self.global_model = copy.deepcopy(self.tc.server.net)
        self.global_model.load_state_dict(
            torch.load(self.global_model_path + \
                       'ServerAggResult(Version%s).pt'%(self.version_tag),
                       map_location=self.device)
            )
        self.global_model = self.global_model.to(self.device)
        self.global_model.train()
        for (n,p) in self.global_model.named_parameters():
            p.requires_grad = True
        
        # train and update
        optimizer = torch.optim.SGD(self.global_model.parameters(),
                                    lr= self.tc.args.lr*(self.tc.args.decay_rate**self.submission_count),
                                    momentum=self.tc.args.momentum,
                                    weight_decay=self.tc.args.weight_decay)
        
        
        epoch_loss = []
        for iter_ in range(self.tc.args.local_ep):
            if self.tc.args.dataset == 'shakespeare':
                # initialize hidden state
                hidden = self.global_model.init_hidden(self.tc.args.local_bs) 
            batch_loss = []
            last_append = 0
            correct = 0
            y_preds = []
            for batch_idx, batch in enumerate(self.ldr_train):
                optimizer.zero_grad()
                self.global_model.zero_grad()
                
                data = batch[0].to(self.device) 
                labels = batch[1].to(self.device)
                if self.tc.args.dataset == 'shakespeare':
                    if data.shape[0] < self.tc.args.local_bs:
                        last_append = data.shape[0]
                        for _ in range(int(np.ceil((self.tc.args.local_bs - data.shape[0])/data.shape[0]))):
                            if data.shape[0] * 2 < self.tc.args.local_bs:
                                data = torch.cat((data, data))
                                
                            else:
                                data = torch.cat((data,
                                                  data[: (self.tc.args.local_bs - data.shape[0]),:]))
                    data = data.t() 
                    labels = labels.t().reshape(-1)
                    
                    # global forward
                    hidden = repackage_hidden(hidden)
                    net_outputs, hidden = self.global_model(data, hidden)
                    # format transformation
                    if last_append > 0:
                        net_outputs = net_outputs[-1: ,:last_append,:] 
                    else:
                        net_outputs = net_outputs[-1: ,:,:]
                    net_outputs = net_outputs.reshape(-1, self.tc.args.ntoken)
                else:
                    # forward
                    net_outputs = self.global_model(data)
                y_pred = net_outputs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
                y_preds.extend([int(i) for i in y_pred])
                            
                # loss
                loss = self.loss_func(net_outputs, labels) 
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.global_model.parameters(), 
                                               max_norm=self.tc.args.max_norm, 
                                               norm_type=2)
                optimizer.step()
                
                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        accuracy = float(100 * correct.item() / len(y_preds))
        print('Worker {} (version {}) Train set: Average loss: {:.4f} Accuracy: {}/{} ({:.2f}%)'.format(
            self.no, self.version_tag,
            epoch_loss[-1], correct.item(), 
            len(y_preds), accuracy))
        self.update = 'WorkerUpdate%s(Version%s).pt'%(self.no, self.version_tag)
        torch.save(self.global_model.state_dict(), 
                   self.global_model_path+self.update)
        
        data, labels, \
        net_outputs, loss, optimizer = None,None,None,None,None
        if self.tc.args.dataset == 'shakespeare':
            hidden = None
        torch.cuda.empty_cache()
        
        
        ############################ submit ##############################
        self.tc.cached_local_update[self.no] = self.update
        self.submission_count += 1
        return 
    
    