import torch

class SemiSupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, data, labeled_indices, unlabeled_indices, name, transform=None, transform_test=None, unlabeled_transform=None):
        self.data = data
        if name in ['cifar10', 'cifar100']:
            self.targets = data.targets
        elif name in ['svhn']:
            self.targets = data.labels.tolist()
        else:
            raise ValueError('SemiSupervisedDataset: dataset name must be cifar10, cifar100, or svhn')
        
        self.labeled_indices = labeled_indices
        self.unlabeled_indices = unlabeled_indices
        self.transform = transform
        self.unlabeled_transform = unlabeled_transform or transform
        self.transform_test = transform_test
        
        if name in ['cifar10', 'svhn']:
            self.num_classes = 10
        elif name in ['cifar100']:
            self.num_classes = 100
        else:
            raise ValueError('SemiSupervisedDataset: dataset name must be cifar10, cifar100, or svhn')
        
        self.train = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, _ = self.data[idx]
        label = self.targets[idx]

        if idx in self.labeled_indices:
            if self.transform:
                image = self.transform(image)

        elif idx in self.unlabeled_indices:
            if not self.train:
                image = self.transform_test(image)
            elif self.train and self.unlabeled_transform:
                image = self.unlabeled_transform(image)

        return image, label, idx

    def update_pseudo_labels(self, indices, pseudo_labels):
        self.targets[indices] = pseudo_labels

