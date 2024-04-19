import argparse
import json
import os

from torchvision.transforms import ToTensor
from dataset.datasets.partitioned_mnist import PartitionedMNIST
from fedECA.server import FedavgServer
from utils.utils import draw, setupSeed, find_project_root
import pandas as pd


def arg_parse():
    parser = argparse.ArgumentParser()

    # global config
    parser.add_argument("--seed", type=int, default=925)
    parser.add_argument("--recordId", type=str, default="mnist_dirichlet_0.5_100_0.1")
    parser.add_argument("--device", type=str, default="cuda")

    # dataset config
    """
    partition = noniid-#label, majorClassesNum = 2 // 2label
    
    partition = noniid-labeldir, dirAlpha = 0.1 // dirichlet
    
    partition = iid // iid
    
    """
    parser.add_argument("--datasetName", type=str, default="mnist", choices=["mnist"])
    parser.add_argument("--partition", type=str, default="noniid-labeldir",
                        choices=["noniid-#labe", "noniid-labeldir", "iid"])
    parser.add_argument("--majorClassesNum", type=int, default=2)
    parser.add_argument("--dirAlpha", type=int, default=0.5)

    # client config
    parser.add_argument("--model", type=str, default="FedAvgCNN")
    parser.add_argument("--modelConfig", type=dict, default={"dataset": "mnist"})
    parser.add_argument("--batchSize", type=int, default=64)
    parser.add_argument("--localEpochs", type=int, default=10)
    parser.add_argument("--optimizer", type=str, default="torch.optim.SGD",
                        choices=["torch.optim.SGD", "torch.optim.Adam"])
    parser.add_argument("--optimConfig", type=dict, default={"lr": 0.01, "momentum": 0.9})
    # server config
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--numClients", type=int, default=100)
    parser.add_argument("--numGlobalEpochs", type=int, default=50)

    args = parser.parse_args()
    return args


def run(configs, data_preprocessed=False):
    dataPartitioner = PartitionedMNIST(numClients=configs.numClients, dataName=configs.datasetName,
                                       path="./clientdata/mnsit", data_preprocessed=data_preprocessed, download=True,
                                       partition=configs.partition, majorClassesNum=configs.majorClassesNum,
                                       dirAlpha=configs.dirAlpha, verbose=True, seed=None,
                                       transform=ToTensor())

    if data_preprocessed:
        draw(dataPartitioner, "./clientdata/mnsit/datainfo")

    print("avg")
    federatedServer = FedavgServer(dataPartitioner, configs)

    results = federatedServer.train()


    # save results
    root = find_project_root()
    recordPath = os.path.join(root, "results/mnist")
    if not os.path.exists(recordPath):
        os.makedirs(recordPath)

    dataFrame = pd.DataFrame(results, columns=["loss", "accuracy", "ratio"])

    dataFrame.to_csv(os.path.join(recordPath, f"{configs.recordId}.csv"), index_label="epoch")


if __name__ == '__main__':
    configs = arg_parse()
    for i in range(10):
        configs.recordId = f"mnist_dirichlet_0.5_100_0.1_{i}"
        run(configs, data_preprocessed=True)

    configs.partition = "noniid-#label"
    configs.majorClassesNum = 2
    for i in range(10):
        configs.recordId = f"mnist_noniid-#label_2_100_0.1_{i}"
        run(configs, data_preprocessed=True)

    configs.partition = "iid"
    for i in range(10):
        configs.recordId = f"mnist_iid_100_0.1_{i}"
        run(configs, data_preprocessed=True)