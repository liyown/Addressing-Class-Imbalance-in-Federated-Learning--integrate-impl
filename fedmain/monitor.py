import copy

import numpy
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from dataset.datasets.basic_dataset import Subset, CIFARSubset


class RatioMonitor:
    def __init__(self, raw_dataset, configs, global_model_last, transform=None, target_transform=None):
        # 用于产生辅助数据的原始数据
        self.raw_dataset = raw_dataset
        self.global_model_last = copy.deepcopy(global_model_last)
        self.global_model_current = None
        self.local_models = []
        self.auxiliary_data_nums = 100
        self.configs = configs
        self.transform = transform
        self.target_transform = target_transform

    def generate_auxiliary_data(self):
        """
        产生辅助数据，在原始数据集中的每个类，随机抽取n个样本，返回num_class个类的数据
        """
        auxiliary_data_loader = []
        for class_num in range(len(self.raw_dataset.classes)):
            # 获取当前类的所有样本的索引
            class_index = numpy.where(numpy.array(self.raw_dataset.targets) == class_num)[0]
            # 在当前类的所有样本索引中随机抽取num_samples个样本索引
            random_index_for_class_index = torch.randperm(len(class_index))[:self.auxiliary_data_nums]
            # 获取当前类的辅助数据下标
            class_index = class_index[random_index_for_class_index]
            if self.configs.datasetName == "mnist":
                auxiliary_data = Subset(self.raw_dataset, class_index, transform=self.transform, target_transform=self.target_transform)
            if self.configs.datasetName == "cifar10":
                auxiliary_data = CIFARSubset(self.raw_dataset, class_index, transform=self.transform, target_transform=self.target_transform)
            auxiliary_data_loader.append(torch.utils.data.DataLoader(auxiliary_data, batch_size=len(auxiliary_data)))
        return auxiliary_data_loader

    def predict_ratio(self, global_model_current, local_models):
        self.global_model_current = copy.deepcopy(global_model_current)
        self.local_models = local_models

        # 计算各个类的辅助模型
        auxiliary_model = self.caculate_class_grad()

        # 返回pos
        pos = self.outlier_detect(self.global_model_last, auxiliary_model)

        # 计算选中模型样本总数
        num_samples = 0
        for local_model in self.local_models:
            num_samples += len(local_model)

        # 计算真实分布
        class_radio = self.caculate_true_class_radio(self.local_models)

        # 计算预测
        res_monitor, res_monitor_in = self.monitoring(auxiliary_model, pos, self.global_model_last, self.global_model_current, len(self.raw_dataset.classes), self.configs.numClients*self.configs.fraction, num_samples, self.configs)

        self.global_model_last = self.global_model_current

        sin = self.cosine_similarity(class_radio, res_monitor)

        return sin

    def caculate_class_grad(self):
        auxiliary_model = []
        auxiliary_data_loader = self.generate_auxiliary_data()
        for class_grad_inx in range(len(auxiliary_data_loader)):
            temp_model = copy.deepcopy(self.global_model_last).to(self.configs.device)
            temp_optimizer = eval(self.configs.optimizer)(temp_model.parameters(), **self.configs.optimConfig)
            for data, label in auxiliary_data_loader[class_grad_inx]:
                data, label = data.float().to(self.configs.device), label.long().to(self.configs.device)
                output = temp_model(data)
                temp_optimizer.zero_grad()
                loss = torch.nn.CrossEntropyLoss()(output, label)
                loss.backward()
                temp_optimizer.step()
            auxiliary_model.append(temp_model)
        return auxiliary_model

    def outlier_detect(self, w_global, w_local):
        w_global = w_global.classifier.linear2.weight.cpu().detach().numpy()
        w = []
        for i in range(len(w_local)):
            temp = (w_local[i].classifier.linear2.weight.cpu().detach().numpy() - w_global) * 100
            w.append(temp)
        res = self.search_neuron_new(w)
        return res

    def search_neuron_new(self, w):
        w = np.array(w)
        pos_res = np.zeros(w.shape)
        for i in range(w.shape[1]):  # 每个模型的线性层权重[26, 512]
            for j in range(w.shape[2]):  # 每个模型的的p类与前一轮的权重
                temp = []
                for p in range(len(w)):
                    temp.append(w[p, i, j])
                max_index = temp.index(max(temp))
                # pos_res[max_index, i, j] = 1
                outlier = np.where(np.abs(temp) / (abs(w[max_index, i, j]+1e-10)) > 0.8)
                if len(outlier[0]) < 2:
                    pos_res[max_index, i, j] = 1
        return pos_res

    def monitoring(self, cc_net, pos, w_glob_last, w_glob, num_class, num_users, num_samples, args):
        res_monitor = []
        res_monitor_in = []
        for cc_class in range(num_class):
            aux_sum = 0
            aux_other_sum = 0
            glob_sum = 0
            layer = 1
            temp_res = []
            for i in range(pos.shape[1]):
                for j in range(pos.shape[2]):
                    if pos[cc_class, i, j] == 1:
                        temp = []
                        last = w_glob_last.classifier.linear2.weight.cpu().detach().numpy()[i, j]
                        cc = cc_net[cc_class].classifier.linear2.weight.cpu().detach().numpy()[i, j]
                        for p in range(len(cc_net)):
                            temp.append(cc_net[p].classifier.linear2.weight.cpu().detach().numpy()[i, j] - last) # W^{j}_{p;i}
                        temp = np.array(temp)
                        temp = np.delete(temp, cc_class)
                        temp_ave = np.sum(temp) / (len(cc_net) - 1) #

                        aux_sum += cc - last
                        aux_other_sum += temp_ave

                        glob_temp = (w_glob.classifier.linear2.weight.cpu().detach().numpy()[
                                         i, j] - last) * num_users * self.auxiliary_data_nums
                        glob_sum += glob_temp
                        res_temp = (glob_temp - num_samples * temp_ave) / ((cc - last - temp_ave) + 1e-10)
                        # print(res_temp)
                        if 0 < res_temp < num_samples * 1.5 / num_class:
                            temp_res.append(res_temp)
            if len(temp_res) != 0:
                res_monitor.append(np.mean(temp_res))
            else:
                res_monitor.append(num_samples / num_class)
            if aux_sum - aux_other_sum == 0:
                res = 0
            else:
                res = (glob_sum - num_samples * aux_other_sum) / (aux_sum - aux_other_sum)
            res_monitor_in.append(res)

        res_monitor = np.array(res_monitor)
        res_monitor_in = np.array(res_monitor_in)
        return res_monitor, res_monitor_in

    def caculate_true_class_radio(self, local_models):
        """
        计算选中客户端真实类别的比例
        """
        class_radio = []
        for class_idx in range(len(self.raw_dataset.classes)):
            radio = 0
            for local_model in local_models:
                dataset = local_model.dataLoader.dataset
                targets = dataset.targets
                radio += len(numpy.where(numpy.array(targets) == class_idx)[0])
            class_radio.append(radio)
        return class_radio

    def cosine_similarity(self, x, y):
        res = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        return res



