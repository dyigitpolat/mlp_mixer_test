from machine_learning.data_loading.data_loader_factory import DataLoaderFactory
from machine_learning.data_loading.data_providers.cifar_10 import CIFAR10_DataProvider
from machine_learning.training.basic_trainer import BasicTrainer
from machine_learning.loss_functions.basic_classification_loss import BasicClassificationLoss
from machine_learning.reporting.wandb_reporter import WandB_Reporter

import torch
import torch.nn as nn

def main():
    device = init()
    model = get_model()

    training_loader, validation_loader, test_loader = get_data_loaders(
        training_batch_size=128, 
        validation_batch_size=2048, 
        test_batch_size=128)

    trainer = BasicTrainer(
        model, device, training_loader, validation_loader, test_loader, 
        BasicClassificationLoss(),
        WandB_Reporter("cifar10", "vgg19_bn").report)
    
    # trainer.train_n_epochs(0.0001, 2)
    print("Test accuracy: {}".format(trainer.test()))

def init():
    assert torch.cuda.is_available(), "CUDA is not available"
    device = torch.device("cuda")
    print("CUDA device name: {}".format(torch.cuda.get_device_name(device)))

    return device

def get_model():
    cifar10_vgg19 = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_vgg19_bn', pretrained=True, trust_repo=True)
    print(cifar10_vgg19)

    return cifar10_vgg19

def get_data_loaders(training_batch_size, validation_batch_size, test_batch_size):
    data_loader_factory = DataLoaderFactory(CIFAR10_DataProvider('datasets'))
    test_loader = data_loader_factory.create_test_loader(test_batch_size)
    training_loader = data_loader_factory.create_training_loader(training_batch_size)
    validation_loader = data_loader_factory.create_validation_loader(validation_batch_size)

    return training_loader, validation_loader, test_loader

if __name__ == '__main__':
    main()