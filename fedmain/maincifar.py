import argparse
import json
import os

import pandas as pd
from torch.optim import Adam
from torchvision.transforms import Normalize, ToTensor, Compose, RandomHorizontalFlip, ColorJitter
from dataset.datasets.partitioned_cifar import PartitionCIFAR
from dataset.datasets.partitioned_mnist import PartitionedMNIST
from fedECA.server import FedavgServer
from utils.utils import draw, setupSeed, find_project_root


def arg_parse():
    parser = argparse.ArgumentParser()

    # global config
    parser.add_argument("--seed", type=int, default="0925")
    parser.add_argument("--recordId", type=str, default="fedavg_cifar10_dirichlet_1")
    parser.add_argument("--device", type=str, default="cuda")

    # dataset config
    """
    balance = None, partition = dirichlet, dirAlpha = 需要的参数         //  一般的dirichlet
    balance = True, partition = dirichlet, dirAlpha = 需要的参数         //  平衡的dirichlet
    balance = False, partition = dirichlet, dirAlpha = 需要的参数        //  不平衡的dirichlet
    
    balance = None, partition = shards, numShards = 需要的参数(200)      //  一般的shards
    
    balance = True, partition = iid                                     //  平衡的iid
    balance = False, partition = iid                                    //  不平衡的iid
    """
    parser.add_argument("--datasetName", type=str, default="cifar10", choices=["cifar10", "mnist"])
    parser.add_argument("--balance", type=bool, default=None)
    parser.add_argument("--partition", type=str, default="dirichlet", choices=["dirichlet", "iid", "shards"])

    parser.add_argument("--unbalanceSgm", type=int, default=0)
    parser.add_argument("--numShards", type=int, default=None)
    parser.add_argument("--dirAlpha", type=int, default=1)

    # client config
    parser.add_argument("--model", type=str, default="FedAvgCNN")
    parser.add_argument("--modelConfig", type=dict, default={"dataset": "cifar10"})
    parser.add_argument("--batchSize", type=int, default=64)
    parser.add_argument("--localEpochs", type=int, default=10)
    parser.add_argument("--optimizer", type=str, default="torch.optim.SGD",
                        choices=["torch.optim.SGD", "torch.optim.Adam"])
    parser.add_argument("--optimConfig", type=dict, default={"lr": 0.01, "momentum": 0.9})
    # server config
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--numClients", type=int, default=100)
    parser.add_argument("--numGlobalEpochs", type=int, default=100)

    args = parser.parse_args()
    return args


def run(configs, preprocess=False):
    setupSeed(configs.seed)
    transform = Compose(
        [RandomHorizontalFlip(), ColorJitter(0.5, 0.5), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataPartitioner = PartitionCIFAR(dataName=configs.datasetName, numClients=configs.numClients,
                                     path="./clientdata/cifar10", preprocess=preprocess, download=True,
                                     balance=configs.balance, partition=configs.partition,
                                     unbalance_sgm=configs.unbalanceSgm, numShards=configs.numShards,
                                     dirAlpha=configs.dirAlpha, verbose=True, seed=None,
                                     transform=transform, targetTransform=None)

    if preprocess is True:
        draw(dataPartitioner, "./clientdata/cifar10/datainfo")

    print(configs.recordId)
    federatedServer = FedavgServer(dataPartitioner, configs)
    results = federatedServer.train()

    # save results
    root = find_project_root()
    recordPath = os.path.join(root, "results/cifar10")
    if not os.path.exists(recordPath):
        os.makedirs(recordPath)

    dataFrame = pd.DataFrame(results, columns=["loss", "accuracy", "ratio"])

    dataFrame.to_csv(os.path.join(recordPath, f"{configs.recordId}.csv"), index_label="epoch")


if __name__ == '__main__':
    configs = arg_parse()
    for i in range(10):
        configs.recordId = f"cifar10_dirichlet_1_100_0.1_{i}"
        run(configs, preprocess=True)

    # balance = None, partition = shards, numShards = 需要的参数(200)      //  一般的shards
    configs.partition = "shards"
    configs.numShards = 200
    for i in range(10):
        configs.recordId = f"cifar10_noniid-#label_2_100_0.1_{i}"
        run(configs, preprocess=True)

    # balance = None, partition = iid                                     //  一般的iid
    configs.partition = "iid"
    for i in range(10):
        configs.recordId = f"cifar10_iid_100_0.1_{i}"
        run(configs, preprocess=True)
