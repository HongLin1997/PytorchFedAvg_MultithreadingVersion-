# Pytorch_FedAvg

## Description
This project implements the classical federated learning algorithm called FedAvg using pytorch and multithreading tools in python.

To run the FedAvg algorithm, you first need to run the following command for splitting a dataset into multiple shards, with each shard belonging to one client in FL.

    python -u main_fed.py --num_clients=100 --dataset=cifar --num_classes=10 --data_allocation_scheme=0

After that, you can use the following command to run the algorithm.

    python -u main.py --num_clients=100 --dataset=cifar --num_classes=10 --data_allocation_scheme=0 \
    --frac=0.1 --local_ep=1 --local_bs=10 --lr=0.01 --decay_rate=0.995 

## Notation
If you think this project is helpful for you, please give us **a star** and feel free to fork this project. 

## Author
Hong Lin: honglin@zju.edu.cn
