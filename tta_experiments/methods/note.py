
import math
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY

from unmixtns import UnMixTNS1d, UnMixTNS2d

class HLoss(nn.Module):
    def __init__(self, temp_factor=1.0):
        super(HLoss, self).__init__()
        self.temp_factor = temp_factor

    def forward(self, x):

        softmax = F.softmax(x/self.temp_factor, dim=1)
        entropy = -softmax * torch.log(softmax+1e-6)
        b = entropy.mean()

        return b


@ADAPTATION_REGISTRY.register()
class NOTE(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super(NOTE, self).__init__(cfg, model, num_classes)
        self.mem = PBRS(capacity=self.cfg.NOTE.MEMORY_SIZE, num_class=self.num_classes)
        self.model = self.model
        self.update_frequency = cfg.NOTE.UPDATE_FREQUENCY  # actually the same as the size of memory bank
        self.current_instance = 0

        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

    def reset(self):
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved self.model/optimizer state")
        self.load_model_and_optimizer()
        self.current_instance = 0
        self.mem  = PBRS(capacity=self.cfg.NOTE.MEMORY_SIZE, num_class=self.num_classes)


    @torch.enable_grad()
    def forward_and_adapt(self, batch_data):
        # batch data
        batch_data = batch_data[0]
        with torch.no_grad():
            self.model.eval()
            out = self.model(batch_data)
            pseudo_label = torch.argmax(out, dim=1)

        # add into memory
        for i, data in enumerate(batch_data):
            p_l = pseudo_label[i].item()
            self.mem.add_instance([data, p_l, 0, 0, 0])
            self.current_instance += 1

            if self.current_instance % self.update_frequency == 0:
                self.update_model(self.model, self.optimizer)

        return out

    def update_model(self, model, optimizer):
        model.train()
        # get memory data
        feats, _, _ = self.mem.get_memory()
        feats = torch.stack(feats)
        entropy_loss = HLoss(temp_factor=1.0)

        feats = feats.cuda()
        preds_of_data = model(feats)
        
        loss = entropy_loss(preds_of_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    def configure_model(self):

        self.model = convert_iabn(self.model)
        for param in self.model.parameters():  # initially turn off requires_grad for all
                param.requires_grad = False
        for module in self.model.modules():

            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
                module.track_running_stats = True
                module.momentum = self.cfg.NOTE.ALPHA
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            elif isinstance(module, nn.InstanceNorm1d) or isinstance(module, nn.InstanceNorm2d): #ablation study
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            if isinstance(module, InstanceAwareBatchNorm2d) or isinstance(module, InstanceAwareBatchNorm1d) or isinstance(module, (UnMixTNS2d, UnMixTNS1d)):
                for param in module.parameters():
                    param.requires_grad = True

        return self.model


def timeliness_reweighting(ages):
    if isinstance(ages, list):
        ages = torch.tensor(ages).float().cuda()
    return torch.exp(-ages) / (1 + torch.exp(-ages))


def convert_iabn(module, **kwargs):
    module_output = module
    if isinstance(module, nn.BatchNorm2d):
        IABN = InstanceAwareBatchNorm2d
        module_output = IABN(
            num_channels=module.num_features,
            k=4,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
        )

        module_output._bn = copy.deepcopy(module)

    for name, child in module.named_children():
        module_output.add_module(
            name, convert_iabn(child, **kwargs)
        )
    del module
    return module_output


class PBRS():

    def __init__(self, capacity, num_class):
        self.num_class = num_class
        self.data = [[[], [], []] for _ in range(num_class)] #feat, pseudo_cls, domain, cls, loss
        self.counter = [0] * num_class
        self.marker = [''] * num_class
        self.capacity = capacity
        pass
    def print_class_dist(self):

        print(self.get_occupancy_per_class())
    def print_real_class_dist(self):

        occupancy_per_class = [0] * self.num_class
        for i, data_per_cls in enumerate(self.data):
            for cls in data_per_cls[3]:
                occupancy_per_class[cls] +=1
        print(occupancy_per_class)

    def get_memory(self):

        data = self.data

        tmp_data = [[], [], []]
        for data_per_cls in data:
            feats, cls, dls = data_per_cls
            tmp_data[0].extend(feats)
            tmp_data[1].extend(cls)
            tmp_data[2].extend(dls)

        return tmp_data

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls[0])
        return occupancy

    def get_occupancy_per_class(self):
        occupancy_per_class = [0] * self.num_class
        for i, data_per_cls in enumerate(self.data):
            occupancy_per_class[i] = len(data_per_cls[0])
        return occupancy_per_class

    def update_loss(self, loss_list):
        for data_per_cls in self.data:
            feats, cls, dls, _, losses = data_per_cls
            for i in range(len(losses)):
                losses[i] = loss_list.pop(0)

    def add_instance(self, instance):
        assert (len(instance) == 5)
        cls = instance[1]
        self.counter[cls] += 1
        is_add = True

        if self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance(cls)

        if is_add:
            for i, dim in enumerate(self.data[cls]):
                dim.append(instance[i])

    def get_largest_indices(self):

        occupancy_per_class = self.get_occupancy_per_class()
        max_value = max(occupancy_per_class)
        largest_indices = []
        for i, oc in enumerate(occupancy_per_class):
            if oc == max_value:
                largest_indices.append(i)
        return largest_indices

    def remove_instance(self, cls):
        largest_indices = self.get_largest_indices()
        if cls not in largest_indices: #  instance is stored in the place of another instance that belongs to the largest class
            largest = random.choice(largest_indices)  # select only one largest class
            tgt_idx = random.randrange(0, len(self.data[largest][0]))  # target index to remove
            for dim in self.data[largest]:
                dim.pop(tgt_idx)
        else:# replaces a randomly selected stored instance of the same class
            m_c = self.get_occupancy_per_class()[cls]
            n_c = self.counter[cls]
            u = random.uniform(0, 1)
            if u <= m_c / n_c:
                tgt_idx = random.randrange(0, len(self.data[cls][0]))  # target index to remove
                for dim in self.data[cls]:
                    dim.pop(tgt_idx)
            else:
                return False
        return True
    


class InstanceAwareBatchNorm1d(nn.Module):
    def __init__(self, num_channels, k=3.0, eps=1e-5, momentum=0.1, affine=True):
        super(InstanceAwareBatchNorm1d, self).__init__()
        self.num_channels = num_channels
        self.k = k
        self.eps = eps
        self.affine = affine
        self._bn = nn.BatchNorm1d(num_channels, eps=eps,
                                  momentum=momentum, affine=affine)

    def _softshrink(self, x, lbd):
        x_p = F.relu(x - lbd, inplace=True)
        x_n = F.relu(-(x + lbd), inplace=True)
        y = x_p - x_n
        return y

    def forward(self, x):
        print(x.shape)
        b, c, l = x.size()
        sigma2, mu = torch.var_mean(x, dim=[2], keepdim=True, unbiased=True)
        if self.training:
            _ = self._bn(x)
            sigma2_b, mu_b = torch.var_mean(x, dim=[0, 2], keepdim=True, unbiased=True)
        else:
            if self._bn.track_running_stats == False and self._bn.running_mean is None and self._bn.running_var is None: # use batch stats
                sigma2_b, mu_b = torch.var_mean(x, dim=[0, 2], keepdim=True, unbiased=True)
            else:
                mu_b = self._bn.running_mean.view(1, c, 1)
                sigma2_b = self._bn.running_var.view(1, c, 1)

        s_mu = torch.sqrt((sigma2_b + self.eps) / l) ##
        s_sigma2 = (sigma2_b + self.eps) * np.sqrt(2 / (l - 1))

        mu_adj = mu_b + self._softshrink(mu - mu_b, self.k * s_mu)
        sigma2_adj = sigma2_b + self._softshrink(sigma2 - sigma2_b, self.k * s_sigma2)
        sigma2_adj = F.relu(sigma2_adj)


        x_n = (x - mu_adj) * torch.rsqrt(sigma2_adj + self.eps)

        if self.affine:
            weight = self._bn.weight.view(c, 1)
            bias = self._bn.bias.view(c, 1)
            x_n = x_n * weight + bias

        return x_n


class InstanceAwareBatchNorm2d(nn.Module):
    def __init__(self, num_channels, k=3.0, eps=1e-5, momentum=0.1, affine=True):
        super(InstanceAwareBatchNorm2d, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.k=k
        self.affine = affine
        self._bn = nn.BatchNorm2d(num_channels, eps=eps,
                                  momentum=momentum, affine=affine)

    def _softshrink(self, x, lbd):
        x_p = F.relu(x - lbd, inplace=True)
        x_n = F.relu(-(x + lbd), inplace=True)
        y = x_p - x_n
        return y

    def forward(self, x):
        b, c, h, w = x.size()
        sigma2, mu = torch.var_mean(x, dim=[2, 3], keepdim=True, unbiased=True) #IN

        if self.training:
            _ = self._bn(x)
            sigma2_b, mu_b = torch.var_mean(x, dim=[0, 2, 3], keepdim=True, unbiased=True)
        else:
            if self._bn.track_running_stats == False and self._bn.running_mean is None and self._bn.running_var is None: # use batch stats
                sigma2_b, mu_b = torch.var_mean(x, dim=[0, 2, 3], keepdim=True, unbiased=True)
            else:
                mu_b = self._bn.running_mean.view(1, c, 1, 1)
                sigma2_b = self._bn.running_var.view(1, c, 1, 1)


        s_mu = torch.sqrt((sigma2_b + self.eps) / (h * w))
        s_sigma2 = (sigma2_b + self.eps) * np.sqrt(2 / (h * w - 1))

        mu_adj = mu_b + self._softshrink(mu - mu_b, self.k * s_mu)

        sigma2_adj = sigma2_b + self._softshrink(sigma2 - sigma2_b, self.k * s_sigma2)

        sigma2_adj = F.relu(sigma2_adj) #non negative

        x_n = (x - mu_adj) * torch.rsqrt(sigma2_adj + self.eps)
        if self.affine:
            weight = self._bn.weight.view(c, 1, 1)
            bias = self._bn.bias.view(c, 1, 1)
            x_n = x_n * weight + bias
        return x_n