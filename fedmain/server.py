import torch
from abstractclass.server import Server
from fedECA.client import FedavgClient
from fedECA.monitor import RatioMonitor
from utils.utils import sendMail, AverageMeter, accuracy



class FedavgServer(Server):
    def __init__(self, dataPartitioner, configs):
        super(FedavgServer, self).__init__(dataPartitioner, configs)
        self.testDataloader = self.dataPartitioner.getDataloader(cid=None, type_="test")
        self.createClients(dataPartitioner, configs)
        self.device =  configs.device
        self.monitor = RatioMonitor(self.dataPartitioner.getDataset(), configs, self.globalModel,
                                    self.dataPartitioner.transform, self.dataPartitioner.targetTransform)

    @sendMail
    def train(self):
        """Train the global model using federated learning."""
        results = {"loss": [], "accuracy": [], "ratio": []}
        for epoch in range(self.configs.numGlobalEpochs):
            print(f"Global epoch: {epoch + 1}/{self.configs.numGlobalEpochs}")
            self.selectClient()
            self.transmitModel()
            selectedTotalSize = self.updateSelectedClients()
            self.averageModel(selectedTotalSize)

            ratio = self.monitor.predict_ratio(self.globalModel, self.selectClients)

            testLoss, testAccuracy = self.evaluate()
            results['loss'].append(testLoss)
            results['accuracy'].append(testAccuracy)
            results['ratio'].append(ratio)

            print(f"Test loss: {testLoss:.4f}, Test accuracy: {testAccuracy:.2f}%, Ratio: {ratio:.4f}")
        return results

    def evaluate(self, printFlag=False):
        self.globalModel.eval().to(self.device)
        lossAndAcc = AverageMeter("Test set:")
        with torch.no_grad():
            for data, labels in self.testDataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.globalModel(data)
                testLoss = torch.nn.CrossEntropyLoss()(outputs, labels).item()
                lossAndAcc.updateLoss(testLoss, data.size(0))
                lossAndAcc.updateAcc(accuracy(outputs, labels)[0], labels.size(0))
            # 如果设备是cuda，清理缓存
            if self.device == "cuda":
                torch.cuda.empty_cache()
        self.globalModel.to("cpu")
        if printFlag:
            print(lossAndAcc)
        return lossAndAcc.lossAvg, lossAndAcc.accAvg

    def createClients(self, dataPartition, configs):
        """Create clients for the server."""
        self.allClients = [FedavgClient(cid, dataPartition, configs) for cid in range(dataPartition.numClients)]
        print(f"Total {dataPartition.numClients} clients created.")
