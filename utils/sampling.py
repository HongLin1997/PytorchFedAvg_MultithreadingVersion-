import numpy as np
from torchvision import datasets, transforms
import os, torch
from .text_helper import Corpus, centralized
from .image_helper import FEMNIST


def get_datasets(args):
    # load datasets
    if args.dataset == 'mnist':
        img_size = torch.Size([args.num_channels, 28, 28])
        
        trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(
                root='./data/mnist/',
                train=True,
                download=True,
                transform=trans_mnist
            )

        dataset_test = datasets.MNIST(
                root='./data/mnist/',
                train=False,
                download=True,
                transform=trans_mnist
            )
    
    elif args.dataset == 'cifar':
        img_size = torch.Size([3, 32, 32])

        trans_cifar10_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        trans_cifar10_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
    
        dataset_train = datasets.CIFAR10(
                root='./data/cifar',
                train=True,
                download=True,
                transform=trans_cifar10_train
            )
    
        dataset_test = datasets.CIFAR10(
                root='./data/cifar',
                train=False,
                download=True,
                transform=trans_cifar10_val
                )
    else:
        exit('Error: unrecognized dataset')
        
        
    return dataset_train, dataset_test, img_size


def data_split(args):
    # load datasets
    dataset_train, dataset_test, img_size = get_datasets(args)
    validation_index = []#np.random.choice(
            #len(dataset_test),int(len(dataset_test)*0.05), replace=False
            #)
    
    # sampling
    if args.data_allocation_scheme == 0:
        data_size_group = 1
        data_size_means = [len(dataset_train)/args.num_workers]
        group_size = [args.num_workers]
        
    data_quantity = []
    for i in range(data_size_group):
        tmp = np.random.normal(data_size_means[i], data_size_means[i]/4, 
                               group_size[i])
        tmp2 = []
        small_index = np.where(tmp<=data_size_means[i])[0]
        if len(small_index) >= group_size[i]/2:
            tmp2 += list(tmp[small_index][:int(group_size[i]/2)])
            tmp2 += list(2*data_size_means[i]-\
                         tmp[small_index][:group_size[i]-int(group_size[i]/2)])
        else:
            large_index = np.where(tmp>=data_size_means[i])[0]
            tmp2 += list(tmp[large_index][:int(group_size[i]/2)])
            tmp2 += list(2*data_size_means[i]-\
                         tmp[large_index][:group_size[i]-int(group_size[i]/2)])
        #tmp2 = tmp2[:group_size[i]]
        if len(tmp2)<group_size[i]:
            tmp2 += tmp2 + tmp2[int(group_size[i]/2):
                                int(group_size[i]/2)+(group_size[i]-len(tmp2))]
        data_quantity += tmp2
    data_quantity = np.array([(int(np.round(i)) if int(np.round(i)) >=2 else 2) \
                              for i in data_quantity])
    data_quantity = sorted(data_quantity)
    print(data_quantity)
    if len(group_size) <= 1:     
        data_idx = list(range(sum(data_quantity)))
        #print(data_idx)
        np.random.shuffle(data_idx)
        workers_idxs = [[] for _ in range(args.num_workers)]
        for idx in range(args.num_workers):
            print('sampling worker %s...'%idx)
            workers_idxs[idx] = np.random.choice(data_idx, 
                        data_quantity[idx], replace=False)
            data_idx = list(set(data_idx)-set(workers_idxs[idx]))
            np.random.shuffle(data_idx)
    else:
        try:
            idxs_labels = np.array(dataset_train.train_labels)  
        except:
            idxs_labels = np.array(dataset_train.targets)
           
        class_num = dict([(c,0) for c in range(args.num_classes)])
        worker_classes = dict()
        for idx in range(args.num_workers):
            worker_classes[idx] = range(args.num_classes)
            for tmp, c in enumerate(worker_classes[idx]):
                if tmp == len(worker_classes[idx])-1:
                    class_num[c] += data_quantity[idx] - \
                        int(data_quantity[idx]/len(worker_classes[idx]))*(len(worker_classes[idx])-1)
                else:
                    class_num[c] += int(data_quantity[idx]/len(worker_classes[idx]))
                
        class_indexes = dict()
        for c,num in class_num.items():
            original_index = list(np.where(idxs_labels==c)[0])
            appended_index = []
            count=0
            while len(appended_index)<num:
                appended_index += [tmp+count*len(idxs_labels) for tmp in original_index]
                count+=1
            np.random.shuffle(appended_index)
            class_indexes[c] = appended_index  
            
        
        workers_idxs = [[] for _ in range(args.num_workers)]
        for idx in range(args.num_workers):
            print('sampling worker %s...'%idx)
            workers_idxs[idx] = []
            for tmp, c in enumerate(worker_classes[idx]):
                if tmp == len(worker_classes[idx])-1:
                    sampled_idx = list(np.random.choice(
                        class_indexes[c], 
                        data_quantity[idx] - \
                        int(data_quantity[idx]/len(worker_classes[idx]))*(len(worker_classes[idx])-1), 
                        replace=False))
                else:
                    sampled_idx = list(np.random.choice(
                        class_indexes[c], 
                        int(data_quantity[idx]/len(worker_classes[idx])),
                        replace=False))
                workers_idxs[idx] += sampled_idx
                class_indexes[c] = list(set(class_indexes[c])-set(sampled_idx))
                np.random.shuffle(class_indexes[c])
            np.random.shuffle(workers_idxs[idx])
            print(data_quantity[idx], len(workers_idxs[idx]),
                  worker_classes[idx], set([idxs_labels[tmp%len(idxs_labels)] for tmp in workers_idxs[idx]]))
    
    dict_workers = {i: workers_idxs[i] for i in range(len(workers_idxs))}
    x=[]
    combine = [] 
    for i in dict_workers.values():
        x.append(len(i))
        combine.append(len(i))
    print('train data partition')
    print('sum:',np.sum(np.array(x)))
    print('mean:',np.mean(np.array(x)))
    print('std:',np.std(np.array(x)))
    print('max:',max(np.array(x)))
    print('min:',min(np.array(x)))
    return dataset_train, dataset_test, validation_index, dict_workers


def text_data_split(args):
    print('Loading data')
    #### check the consistency of # of batches and size of dataset for poisoning
    
    corpus_file_name = f'{args.data_folder}/corpus_{args.train_frac}.pt.tar'
    if not os.path.exists(corpus_file_name):
        corpus = Corpus(args)
        torch.save(corpus, corpus_file_name)           
    else:
        corpus = torch.load(corpus_file_name)
        
    centralized_train_data, centralized_train_label,\
    dataset_validate, dataset_validate_label, \
    centralized_test_data, centralized_test_label = centralized(corpus, 
                                                                load_train=True)
    print('train data: ', len(centralized_train_label))
    print('test data: ', len(centralized_test_label))
           
    if args.data_allocation_scheme == 0:
        data_size_group = 1
        data_size_means = [len(centralized_train_label)/args.num_workers]
        group_size = [args.num_workers]
        
    data_quantity = []
    for i in range(data_size_group):
        tmp = np.random.normal(data_size_means[i], data_size_means[i]/4, 
                               group_size[i])
        tmp2 = []
        small_index = np.where(tmp<=data_size_means[i])[0]
        if len(small_index) >= group_size[i]/2:
            tmp2 += list(tmp[small_index][:int(group_size[i]/2)])
            tmp2 += list(2*data_size_means[i]-\
                         tmp[small_index][:group_size[i]-int(group_size[i]/2)])
        else:
            large_index = np.where(tmp>=data_size_means[i])[0]
            tmp2 += list(tmp[large_index][:int(group_size[i]/2)])
            tmp2 += list(2*data_size_means[i]-\
                         tmp[large_index][:group_size[i]-int(group_size[i]/2)])
        #tmp2 = tmp2[:group_size[i]]
        if len(tmp2)<group_size[i]:
            tmp2 += tmp2 + tmp2[int(group_size[i]/2):
                                int(group_size[i]/2)+(group_size[i]-len(tmp2))]
                
        data_quantity += tmp2
    data_quantity = np.array([(int(np.round(i)) if int(np.round(i)) >=2 else 2) \
                              for i in data_quantity])
    data_quantity = sorted(data_quantity)
    print(data_quantity)
    if len(group_size) <= 1:
        data_idx = list(range(sum(data_quantity)))
        #print(data_idx)
        np.random.shuffle(data_idx)
        workers_idxs = [[] for _ in range(args.num_workers)]
        for idx in range(args.num_workers):
            print('sampling worker %s...'%idx)
            workers_idxs[idx] = np.random.choice(data_idx, 
                        data_quantity[idx], replace=False)
            data_idx = list(set(data_idx)-set(workers_idxs[idx]))
            np.random.shuffle(data_idx)
    else:
        idxs_labels = np.array(centralized_train_label)
        #print(idxs_labels)
        classes_list = list(set(idxs_labels))
        class_num = dict([(int(c),0) for c in classes_list])
        classes_num = dict([(int(c),list(idxs_labels).count(c)) for c in classes_list])
        print(classes_num)
        worker_classes = dict()
        for idx in range(args.num_workers):
            worker_classes[idx] = classes_list
            sampled_num = 0
            worker_classes_num = sum([classes_num[c] for c in worker_classes[idx]])
            for tmp, c in enumerate(worker_classes[idx]):
                if tmp == len(worker_classes[idx])-1:
                    class_num[c] += data_quantity[idx] - sampled_num
                else:
                    sampled_num += max(1,int(data_quantity[idx]*classes_num[c]/worker_classes_num))
                    class_num[c] += max(1,int(data_quantity[idx]*classes_num[c]/worker_classes_num))#int(data_quantity[idx]/len(worker_classes[idx]))
        class_indexes = dict()
        for c,num in class_num.items():
            original_index = list(np.where(idxs_labels==c)[0])
            appended_index = []
            count=0
            while len(appended_index)<num:
                appended_index += [tmp+count*len(idxs_labels) for tmp in original_index]
                count+=1
            np.random.shuffle(appended_index)
            class_indexes[c] = appended_index  
            
            print('process class %s: '%c, num, len(appended_index))

        workers_idxs = [[] for _ in range(args.num_workers)]
        for idx in range(args.num_workers):
            print('sampling worker %s...'%idx)
            workers_idxs[idx] = []
            sampled_num = 0
            worker_classes_num = sum([classes_num[c] for c in worker_classes[idx]])
            for tmp, c in enumerate(worker_classes[idx]):
                if tmp == len(worker_classes[idx])-1:
                    sampled_idx = list(np.random.choice(
                        class_indexes[c], 
                        data_quantity[idx] - sampled_num, 
                        replace=False))
                else:
                    sampled_idx = list(np.random.choice(
                        class_indexes[c], 
                        max(1,int(data_quantity[idx]*classes_num[c]/worker_classes_num)),#int(data_quantity[idx]/len(worker_classes[idx])),
                        replace=False))
                    sampled_num += max(1,int(data_quantity[idx]*classes_num[c]/worker_classes_num))
                workers_idxs[idx] += sampled_idx
                class_indexes[c] = list(set(class_indexes[c])-set(sampled_idx))
                np.random.shuffle(class_indexes[c])
            np.random.shuffle(workers_idxs[idx])
            print(data_quantity[idx], len(workers_idxs[idx]),
                  worker_classes[idx], set([idxs_labels[tmp%len(idxs_labels)] for tmp in workers_idxs[idx]]))
           
    dict_workers = {i: np.array(workers_idxs[i], dtype='int64') \
                    for i in range(len(workers_idxs))}
    
    x=[]
    combine = [] 
    for i in dict_workers.values():
        x.append(len(i))
        combine.append(len(i))
    print('train data partition')
    print('sum:',np.sum(np.array(x)))
    print('mean:',np.mean(np.array(x)))
    print('std:',np.std(np.array(x)))
    print('max:',max(np.array(x)))
    print('min:',min(np.array(x)))
    return centralized_train_data, centralized_train_label, \
           dataset_validate, dataset_validate_label, \
           centralized_test_data, centralized_test_label, dict_workers

def image_data_split(args):
    print('Loading data')
    #### check the consistency of # of batches and size of dataset for poisoning
    
    file_name = f'{args.femnist_data_folder}/image.pt.tar'
    if not os.path.exists(file_name):
        images = FEMNIST(args)
        torch.save(images, file_name)           
    else:
        images = torch.load(file_name)
    
    centralized_train_data, centralized_train_label,\
    dataset_validate, dataset_validate_label, \
    centralized_test_data, centralized_test_label = centralized(images, 
                                                                load_train=True)
    print('train data: ', len(centralized_train_label))
    print('test data: ', len(centralized_test_label))
    
    if args.data_allocation_scheme == 0:
        data_size_group = 1
        data_size_means = [len(centralized_train_label)/args.num_workers]
        group_size = [args.num_workers]
    data_quantity = []
    for i in range(data_size_group):
        tmp = np.random.normal(data_size_means[i], data_size_means[i]/4, 
                               group_size[i])
        tmp2 = []
        small_index = np.where(tmp<=data_size_means[i])[0]
        if len(small_index) >= group_size[i]/2:
            tmp2 += list(tmp[small_index][:int(group_size[i]/2)])
            tmp2 += list(2*data_size_means[i]-\
                         tmp[small_index][:group_size[i]-int(group_size[i]/2)])
        else:
            large_index = np.where(tmp>=data_size_means[i])[0]
            tmp2 += list(tmp[large_index][:int(group_size[i]/2)])
            tmp2 += list(2*data_size_means[i]-\
                         tmp[large_index][:group_size[i]-int(group_size[i]/2)])
        #tmp2 = tmp2[:group_size[i]]
        if len(tmp2)<group_size[i]:
            tmp2 += tmp2 + tmp2[int(group_size[i]/2):
                                int(group_size[i]/2)+(group_size[i]-len(tmp2))]
                
        data_quantity += tmp2
    data_quantity = np.array([(int(np.round(i)) if int(np.round(i)) >=2 else 2) \
                              for i in data_quantity])
    
    #min_q = int(np.round(min(data_quantity)))
    #if args.data_size_variance>0:
    #    data_quantity = np.array([round(i+(50-min_q)) for i in data_quantity])
    #else:
    #data_quantity = np.array([round(i) for i in data_quantity])
    
    data_idx = list(range(sum(data_quantity)))
    np.random.shuffle(data_idx)
    workers_idxs = [[] for _ in range(args.num_workers)]
    for no, idx in enumerate(range(args.num_workers)):
        workers_idxs[idx] = np.random.choice(data_idx, 
                    data_quantity[no], replace=False)
        data_idx = list(set(data_idx)-set(workers_idxs[idx]))
        np.random.shuffle(data_idx)
    
    dict_workers = {i: np.array(workers_idxs[i], dtype='int64') \
                    for i in range(len(workers_idxs))}
    
    x=[]
    combine = [] 
    for i in dict_workers.values():
        x.append(len(i))
        combine.append(len(i))
    print('train data partition')
    print('sum:',np.sum(np.array(x)))
    print('mean:',np.mean(np.array(x)))
    print('std:',np.std(np.array(x)))
    print('max:',max(np.array(x)))
    print('min:',min(np.array(x)))
     
    
    return centralized_train_data, centralized_train_label, \
           dataset_validate, dataset_validate_label, \
           centralized_test_data, centralized_test_label, dict_workers
           
   