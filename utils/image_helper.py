import torch
import json
import os
import numpy as np

class FEMNIST(object):
    def __init__(self, params):   
        self.path = params.femnist_data_folder
        self.TRAIN_DATA_NAME = f"{self.path}/data/train/"
        self.TEST_DATA_NAME = f"{self.path}/data/test/"
        
        print('loading train dataset...')
        self.train_file = []
        for file in os.listdir(self.TRAIN_DATA_NAME):
            print('train ', file)
            with open(self.TRAIN_DATA_NAME+file,'r') as f:
                self.train_file.append(json.load(f))
        
        print('loading test dataset...')
        self.test_file = []
        for file in os.listdir(self.TEST_DATA_NAME):
            print('test ', file)
            with open(self.TEST_DATA_NAME+file,'r') as f:
                self.test_file.append(json.load(f))
                 
        '''     
        with open(f'{self.TEST_DATA_NAME}','r') as f:
            self.test_file = json.load(f)
            '''
        self.clients = []
        for f in self.train_file:
            self.clients += f['users'] 
        
        self.n_k_train = []
        for f in self.train_file:
            self.n_k_train += f['num_samples'] 
        
        self.n_k_test = []
        for f in self.test_file:
            self.n_k_test += f['num_samples'] 
        
        self.n_k = np.array(self.n_k_train)+np.array(self.n_k_test)
        self.p_k = np.array(self.n_k_train)/sum(self.n_k_train)
        
        self.train, self.train_label, \
        self.test, self.test_label = self.produce()
    
    def produce(self):
        
        print('train data partition')
        print('sum:',sum(self.n_k_train))
        print('mean:',sum(self.n_k_train)/len(self.n_k_train))
        print('std:',np.std(np.array(self.n_k_train)))
        
        print('train data partition')
        print('sum:',sum(self.n_k_test))
        print('mean:',sum(self.n_k_test)/len(self.n_k_test))
        print('std:',np.std(np.array(self.n_k_test)))
        
        per_participant_ids_train = [] 
        per_participant_ids_train_label = []
        per_participant_ids_test = []
        per_participant_ids_test_label = []
        
        for no, user in enumerate(self.clients):
            print('client %s with %s data...'%(no,self.n_k_train[no]))
            for f in self.train_file:
                if user not in f['users']:
                    continue
                
                train, train_label = [], []
                for x, y in zip(f['user_data'][user]['x'], 
                                f['user_data'][user]['y']):
                    train.append(np.array(x).reshape(1,28,28))
                    train_label.append(y)
                
                per_participant_ids_train.append(torch.FloatTensor(np.stack(train)))
                per_participant_ids_train_label.append(torch.LongTensor(train_label))
            
            for f in self.test_file:
                if user not in f['users']:
                    continue
               
                test, test_label = [], []
                for x, y in zip(f['user_data'][user]['x'], 
                                f['user_data'][user]['y']):
                    test.append(np.array(x).reshape(1,28,28))
                    test_label.append(y)
                
                per_participant_ids_test.append(torch.FloatTensor(np.stack(test)))
                per_participant_ids_test_label.append(torch.LongTensor(test_label))
                
        print('total train:', sum([1 for u in per_participant_ids_train for w in u]))
        print('total test:', sum([1 for u in per_participant_ids_test for w in u]))
        
        return per_participant_ids_train, per_participant_ids_train_label,\
               per_participant_ids_test, per_participant_ids_test_label
