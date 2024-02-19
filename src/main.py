from machine_learning.data_loading.data_loader_factory import DataLoaderFactory
from machine_learning.data_loading.data_providers.cifar_10 import CIFAR10_DataProvider
from machine_learning.training.basic_trainer import BasicTrainer
from machine_learning.loss_functions.basic_classification_loss import BasicClassificationLoss
from machine_learning.reporting.wandb_reporter import WandB_Reporter

from mlp_mixer import MLPMixer

import torch
import torch.nn as nn

def main():
    device = init()
    

    patch_size = 8
    num_layers = 4
    num_features = 96
    expansion_factor = 2
    dropout = 0.5

    model = get_model(patch_size, num_layers, num_features, expansion_factor, dropout)

    training_batch_size = 128
    training_loader, validation_loader, test_loader = get_data_loaders(
        training_batch_size=training_batch_size, 
        validation_batch_size=2048, 
        test_batch_size=128)
    
    trainer = BasicTrainer(
        model, device, training_loader, validation_loader, test_loader, 
        BasicClassificationLoss(),
        WandB_Reporter("cifar10_layers", f"MLPMIXER_{training_batch_size}:{patch_size}_{num_layers}_{num_features}_{expansion_factor}_{dropout}").report)
    
    trainer.train_n_epochs(0.001, 200)
    print("Test accuracy: {}".format(trainer.test()))

def init():
    assert torch.cuda.is_available(), "CUDA is not available"
    device = torch.device("cuda")
    print("CUDA device name: {}".format(torch.cuda.get_device_name(device)))

    return device

def get_model(
        patch_size, num_layers, num_features, expansion_factor,
        dropout):
    mlp_mixer = MLPMixer(
        image_size=32, in_channels=3, 
        patch_size=patch_size, num_layers=num_layers, num_features=num_features, expansion_factor=expansion_factor,
        dropout=dropout,
        num_classes=10)
    print(mlp_mixer)

    return mlp_mixer

def get_data_loaders(training_batch_size, validation_batch_size, test_batch_size):
    data_loader_factory = DataLoaderFactory(CIFAR10_DataProvider('datasets'))
    test_loader = data_loader_factory.create_test_loader(test_batch_size)
    training_loader = data_loader_factory.create_training_loader(training_batch_size)
    validation_loader = data_loader_factory.create_validation_loader(validation_batch_size)

    return training_loader, validation_loader, test_loader

if __name__ == '__main__':
    main()