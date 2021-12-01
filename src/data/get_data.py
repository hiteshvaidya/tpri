import os
import torch
import numpy as np
from torchvision import datasets as datasets
from torchvision.transforms import ToTensor, Normalize, Compose, Grayscale
from torch.utils.data import Dataset, DataLoader, Sampler, SubsetRandomSampler

mean_ref = dict(MNIST=(0.1307,), CIFAR=(0.4914, 0.4822, 0.4465), FashionMNIST=(0.2860,))
std_ref = dict(MNIST=(0.3081,), CIFAR=(0.247, 0.243, 0.261), FashionMNIST=(0.3202, ))
nb_classes_ref = dict(MNIST=10, CIFAR=10, FashionMNIST=10)
size_img_ref = dict(MNIST=[28, 28], CIFAR=[32, 32, 3], FashionMNIST=[28, 28])
nb_data_ref = dict(MNIST=60000, CIFAR=50000, FashionMNIST=60000)
test_set_proportion = dict(MNIST=1/6, CIFAR=1/5, temp_ord_1bit=1/10, addtask=1/10, FashionMNIST=1/6)
data_loader = dict(MNIST=datasets.MNIST, CIFAR=datasets.CIFAR10, FashionMNIST=datasets.FashionMNIST)

file_path = os.path.abspath(__file__)
data_folder = os.path.dirname(file_path)


def get_data(dataset='synth', nb_train=None, batch_size=None,
             vectorize=True, make_one_hot=False, normalize=True,
             max_length=60, fixed_length=True, with_permut=False,
             size_chunk=1, seed=1):
    """
    Create train and test loaders
    Args:
        dataset (str): choose between (temp_ord_1_bit, addtask, MNIST, CIFAR).
        temp_ord1bit and addtask generate samples randomly, while MNIST and CIFAR are classical image datasets
        nb_train (int): number of training examples (for temp_1_ord_bit and addtask, this defines the number of
        samples to check to compute the training loss. The number of testing samples is a fraction of the number
        of training samples (see test_set_proportion above)
        batch_size (int): size of the mini-batches generated during the optimization process
        vectorize (bool): whether to vectorize the input (for dataset in ['MNIST', 'CIFAR'])
        make_one_hot (bool): whether to encode the labels as a binary vector for e.g. a MSE Loss
        normalize (bool): whether the data is normalized (for dataset in ['MNIST', 'CIFAR'])
        max_length (int): max length for the synthetic tasks (for dataset in ['temp_ord_1_bit', 'addtask'])
        fixed_length (bool): whether the lengths in the synthetic tasks vary from a mini-batch to the other.
        with_permut (bool): whether to permute the pixels of the images (for dataset in ['MNIST', 'CIFAR'])
        seed (int): random seed fixed for torch

    Returns:
        train_data (torch.utils.data.DataLoader): data loading training samples
        test_data (torch.utils.data.DataLoader): data loading testing samples

    """
    torch.manual_seed(seed)

    if dataset == 'temp_ord_1bit':
        nb_train = 4000 if nb_train is None else nb_train
        nb_test = int(nb_train*test_set_proportion[dataset])
        train_set = TempOrdTask(max_length, nb_train, fixed_length=fixed_length)
        test_set = TempOrdTask(max_length, nb_test, fixed_length=fixed_length)
        train_data = DataLoader(train_set, batch_size=None)
        test_data = DataLoader(test_set, batch_size=None)

    elif dataset == 'addtask':
        nb_train = 4000 if nb_train is None else nb_train
        nb_test = int(nb_train*test_set_proportion[dataset])
        train_set = Addtask(max_length, nb_train, fixed_length=fixed_length)
        test_set = Addtask(max_length, nb_test, fixed_length=fixed_length
                           )
        train_data = DataLoader(train_set, batch_size=None)
        test_data = DataLoader(test_set, batch_size=None)
    else:
        if nb_train is None:
            nb_train = nb_data_ref[dataset]
        nb_test = int(nb_train * test_set_proportion[dataset])
        assert dataset in {'MNIST', 'CIFAR', 'FashionMNIST'}
        nb_classes = nb_classes_ref[dataset]
        transform = ToTensor()
        if normalize:
            transform = Compose([transform, Normalize(mean_ref[dataset], std_ref[dataset])])
        if vectorize:
            transform = Compose([transform, Vectorize(size_chunk)])
            if with_permut:
                length_img = np.prod(size_img_ref[dataset])
                transform = Compose([transform, Permut(torch.randperm(length_img))])

        target_transform = None
        if make_one_hot:
            target_transform = MakeOneHot(nb_classes)

        train_set = data_loader[dataset](root='{0}/{1}'.format(data_folder, dataset), train=True,
                                         transform=transform, target_transform=target_transform, download=True)
        test_set = data_loader[dataset](root='{0}/{1}'.format(data_folder, dataset), train=False,
                                        transform=transform, target_transform=target_transform, download=True)

        train_subset = list(range(nb_train))
        test_subset = list(range(nb_test))
        train_sampler = SubsetRandomSampler(train_subset)
        test_sampler = SubsetRandomSampler(test_subset)
        if batch_size is None:
            batch_size = nb_train
        train_data = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
        test_data = DataLoader(test_set, batch_size=batch_size, sampler=test_sampler)

    return train_data, test_data


class TempOrdTask(Dataset):
    """
    Data set for temporal ordering problem.
    Code inspired from "Target Propagation in Recurrent Neural Networks" by N. Manchev, M. Spartling, JMLR 2020
    See e.g. explanations given in the above paper.
    """
    def __init__(self, max_seq_length, size_data_set, nb_bits=1, fixed_length=True, batch_size=20):
        super(TempOrdTask, self).__init__()
        self.dim_in = 6
        self.dim_out = 4 if nb_bits == 1 else 8
        self.max_length = max_seq_length
        self.batch_size = 20
        self.size_data_set = size_data_set
        self.nb_bits = nb_bits
        self.fixed_length = fixed_length
        self.batch_size = batch_size

    def __len__(self):
        return self.size_data_set

    def __getitem__(self, index):
        """
        Get a mini-batch of samples for the adding task
        the data sample is of size [batch_size, max_length, dim_in]
        the target sample is of size [batch_size, dim_out]
        """
        l = self.max_length if self.fixed_length else torch.randint(10, self.max_length, (1, 1)).item()
        p0 = torch.randint(int(l*.1), size=(self.batch_size,)) + int(l * .1)
        v0 = torch.randint(2, size=(self.batch_size,))
        p1 = torch.randint(int(l*.1), size=(self.batch_size,)) + int(l * .5)
        v1 = torch.randint(2, size=(self.batch_size,))
        targ = v0 + v1 * 2
        vals = torch.randint(self.dim_out, size=(l, self.batch_size)) + 2
        if self.nb_bits == 3:
            p2 = torch.randint(int(l * .1), size=(self.batch_size,)) + int(l * .6)
            v2 = torch.randint(2, size=(self.batch_size,))
            vals[p2, torch.arange(self.batch_size)] = v2
            targ = targ + v2*4
        vals[p0, torch.arange(self.batch_size)] = v0
        vals[p1, torch.arange(self.batch_size)] = v1
        data = torch.zeros((l, self.batch_size, 6))
        data.reshape((l * self.batch_size, 6))[torch.arange(l * self.batch_size), vals.flatten()] = 1.
        data.transpose_(0, 1)
        return data, targ


class Addtask(Dataset):
    """
        Data set for adding task.
        Code inspired from "Target Propagation in Recurrent Neural Networks" by N. Manchev, M. Spartling, JMLR 2020
        See e.g. explanations given in the above paper.
    """
    def __init__(self, max_seq_length, size_data_set, fixed_length=True, batch_size=20):
        self.max_length = max_seq_length
        self.size_data_set = size_data_set
        self.nin = 2
        self.nout = 1
        self.batch_size = batch_size
        self.fixed_length = fixed_length

    def __len__(self):
        return self.size_data_set

    def __getitem__(self, index):
        """
        Get a mini-batch of samples for the adding task
        the data sample is of size [batchsize, max_length, dim_in]
        the target sample is of size [batch_size, dim_out]

        """
        l = self.max_length if self.fixed_length else torch.randint(10, self.max_length, (1, 1)).item()

        p0 = torch.randint(int(l * .1), size=(self.batch_size,))
        p1 = torch.randint(int(l * .4), size=(self.batch_size,)) + int(l * .1)
        data = torch.rand(size=(l, self.batch_size, 2), )
        data[:, :, 0] = 0.
        # access array through a list of indexes
        data[p0, torch.arange(self.batch_size), torch.zeros(self.batch_size, dtype=int)] = 1.
        data[p1, torch.arange(self.batch_size), torch.zeros(self.batch_size, dtype=int)] = 1.

        targs = ((data[p0, torch.arange(self.batch_size),
                      torch.ones(self.batch_size, dtype=int)] +
                  data[p1, torch.arange(self.batch_size),
                      torch.ones(self.batch_size, dtype=int)]) / 2.)
        return data.transpose_(0, 1), targs.unsqueeze(-1)


class Vectorize(object):
    """
    Vectorize a tensor
    """

    def __init__(self, size_chunk=1):
        self.size_chunk = size_chunk

    def __call__(self, tensor):
        return tensor.view(-1, self.size_chunk)



class Permut():
    """
    Transform that permutes a vector
    (so if an image had been vectorized to be scanned by an RNN, this transform permutes the pixels,
     making the spatial dependency potentially more random)
    """
    def __init__(self, permut):
        self.permut = permut

    def __call__(self, tensor):
        return tensor[self.permut]


class MakeOneHot(object):
    """
    Transform ordinary output into corresponding binary vector
    """
    def __init__(self, nb_classes):
        self.nb_classes = nb_classes

    def __call__(self, target):
        out = torch.eye(self.nb_classes)
        return out[target].view(-1)

    def __repr__(self):
        return self.__class__.__name__ + '(nb_classes={0})'.format(self.nb_classes)


class SubsetSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)







