import torch
import json
import re
import numpy as np

filter_symbols = re.compile('[a-zA-Z]*')
ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)

class Dictionary(object):
    def __init__(self):
        ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
        self.NUM_LETTERS = len(ALL_LETTERS)
        self.word2idx = dict([(j,i) for i,j in enumerate(ALL_LETTERS)])
        self.idx2word = dict([(i,j) for i,j in enumerate(ALL_LETTERS) ])
    
    def add_word(self, word):
        raise ValueError("Please don't call this method, so we won't break the dictionary :) ")
    
    def __len__(self):
        return len(self.NUM_LETTERS)

def get_character_list(line, dictionary):
    splitted_words = line
    words = ['<bos>']
    for word in splitted_words:
        #word = filter_symbols.search(word)[0]
        if len(word)>1:
            if dictionary.word2idx.get(word, False):
                words.append(word)
            else:
                words.append('<unk>')
    words.append('<eos>')

    return words

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

 
def centralized(corpus, load_train = True):
    if load_train:
        centralized_train_data  = torch.cat(corpus.train)
        centralized_train_label = torch.cat(corpus.train_label)
    else:
        centralized_train_data, centralized_train_label = None, None
    centralized_test_data = torch.cat(corpus.test)
    centralized_test_label = torch.cat(corpus.test_label)
    validation_index = []#np.random.choice(
            #range(len(centralized_test_label)),int(len(centralized_test_label)*0.05)
            #)
    test_index = set(range(len(centralized_test_label)))-set(validation_index)
    centralized_validation_data = torch.cat(
            [centralized_test_data[idx].reshape(1,80) for idx in validation_index]
            )
    centralized_test_data = torch.cat(
            [centralized_test_data[idx].reshape(1,80) for idx in test_index]
            )
    centralized_validation_label = torch.cat(
            [centralized_test_label[idx].reshape(-1) for idx in validation_index]
            )
    centralized_test_label = torch.cat(
            [centralized_test_label[idx].reshape(-1) for idx in test_index]
            )
    print(len(test_index),centralized_test_data.shape, centralized_test_label.shape)
    
    return centralized_train_data, centralized_train_label,\
           centralized_validation_data, centralized_validation_label, \
           centralized_test_data, centralized_test_label
    
class Corpus(object):
    def __init__(self, params):   
        self.path = params.data_folder
        self.TRAIN_DATA_NAME = f"{self.path}/data/train/all_data_niid_0_keep_0_train_8.json"
        self.TEST_DATA_NAME = f"{self.path}/data/test/all_data_niid_0_keep_0_test_8.json"
        
        self.dictionary = Dictionary()
        with open(f'{self.TRAIN_DATA_NAME}','r') as f:
            self.train_file = json.load(f)
        with open(f'{self.TEST_DATA_NAME}','r') as f:
            self.test_file = json.load(f)
        self.clients = self.train_file['users']
        
        self.no_tokens = self.dictionary.NUM_LETTERS
        self.n_k_train, self.n_k_test, self.n_k, \
        self.p_k, self.train, self.train_label, \
        self.test, self.test_label = self.tokenize_train()
    
    def tokenize_train(self):
        
        p_k = [i/sum(self.train_file['num_samples']) for i in self.train_file['num_samples']]
        n_k = []
        n_k_train, n_k_test = [],[]
        print('train data partition')
        print('sum:',np.sum(np.array(self.train_file['num_samples'])))
        print('mean:',np.mean(np.array(self.train_file['num_samples'])))
        print('std:',np.std(np.array(self.train_file['num_samples'])))
        
        per_participant_ids_train = [] 
        per_participant_ids_train_label = []
        per_participant_ids_test = []
        per_participant_ids_test_label = []
        for user in self.clients:
            train, train_label = [], []
            for x, y in zip(self.train_file['user_data'][user]['x'], 
                            self.train_file['user_data'][user]['y']):
                train.append([self.dictionary.word2idx.get(c) for c in x])
                train_label.append(self.dictionary.word2idx.get(y))
            
            per_participant_ids_train.append(torch.LongTensor(train))
            per_participant_ids_train_label.append(torch.LongTensor(train_label))
            
            test, test_label = [], []
            for x, y in zip(self.test_file['user_data'][user]['x'], 
                            self.test_file['user_data'][user]['y']):
                test.append([self.dictionary.word2idx.get(c) for c in x])
                test_label.append(self.dictionary.word2idx.get(y))
            
            per_participant_ids_test.append(torch.LongTensor(test))
            per_participant_ids_test_label.append(torch.LongTensor(test_label))
            n_k.append(len(train)+len(test))
            n_k_train.append(len(train))
            n_k_test.append(len(test))
        print('total train:', sum([1 for u in per_participant_ids_train_label for w in u]))
        print('total test:', sum([1 for u in per_participant_ids_test_label for w in u]))
        
        return n_k_train, n_k_test, n_k, p_k, \
               per_participant_ids_train, per_participant_ids_train_label,\
               per_participant_ids_test, per_participant_ids_test_label
