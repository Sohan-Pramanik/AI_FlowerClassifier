import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from PIL import Image
import json

import torchvision
import torchvision.models as models
from torchvision import datasets, transforms, models

from collections import OrderedDict
import random, os
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', action='store', dest='arch', type=str, default='vgg16', choices=['vgg16', 'densenet121'])
    # parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--hidden_units', action='store', dest='hidden_units', type=int, default=512)
    parser.add_argument('--data_dir', metavar='data_dir', type=str)
    parser.add_argument('--save_dir', action='store', dest='save_dir', type=str, default='train_checkpoint.pth')
    parser.add_argument('--learning_rate', action='store', dest='learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', action='store', dest='epochs', type=int, default=5)

    return parser.parse_args()

def get_model(arch, hidden_units):

    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        feat = 25088
        hidden_units = 2048
    else:
        model = models.densenet121(pretrained=True)
        feat = 1024
        hidden_units = 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # if gpu:
    #     device = torch.device("cuda")
    #     print('gpu')
    # else:
    #     device = torch.device("cpu")
    #     print('cpu')

    for parameters in model.parameters():
        parameters.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                                    ('fc1', nn.Linear(feat, hidden_units)),
                                    ('relu', nn.ReLU()),
                                    ('fc2', nn.Linear(hidden_units, 256)),
                                    ('relu', nn.ReLU()),
                                    ('fc3', nn.Linear(256, 102)),
                                    ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    model.to(device)

    return model, device, feat

def train(epochs, trainloader, validloader, model, device, criterion, optimizer):

    steps = 0
    running_loss = 0
    print_every = 10

    start = time.time()
    print('Training Started:')

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation Loss: {valid_loss/len(validloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()

    time_elapsed = time.time() - start
    print("\nTime spent training: {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))

def save_data(file_path, model, image_datasets, epochs, optimizer, learning_rate, input_size, output_size, arch, hidden_units):

    model.class_to_idx = image_datasets[0].class_to_idx

    checkpoint = {'input_size': input_size,
                  'output_size': output_size,
                  'hidden_units': hidden_units,
                  'pretrained_model': arch,
                  'classifier' : model.classifier,
                  'learning_rate': learning_rate,
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                 }

    torch.save(checkpoint, file_path)
    print("Model Saved")

def main():

    torch.cuda.empty_cache()

    print("Program Started:")

    args = parse_args()
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    image_datasets = image_datasets = [datasets.ImageFolder(train_dir, transform=train_transforms),
                                       datasets.ImageFolder(valid_dir, transform=valid_transforms),
                                       datasets.ImageFolder(test_dir, transform=test_transforms)]

    dataloaders = [torch.utils.data.DataLoader(image_datasets[0], batch_size=32, shuffle=True),
                   torch.utils.data.DataLoader(image_datasets[1], batch_size=32, shuffle=True),
                   torch.utils.data.DataLoader(image_datasets[2], batch_size=32, shuffle=True)]

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    model, device, feat = get_model(args.arch, args.hidden_units)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    train(args.epochs, dataloaders[0], dataloaders[1], model, device, criterion, optimizer)

    file_path = args.save_dir

    output_size = 102
    save_data(file_path, model, image_datasets, args.epochs, optimizer, args.learning_rate,
                    feat, output_size, args.arch, args.hidden_units)

if __name__ == "__main__":
    main()
