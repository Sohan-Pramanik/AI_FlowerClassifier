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
    parser.add_argument('--gpu', action='store_true', default=True)
    parser.add_argument('--image_path', metavar='image_path', type=str, default='flowers/test/online_test/image_124640.jpg')
    parser.add_argument('--checkpoint', metavar='checkpoint', type=str, default='train_checkpoint.pth')
    parser.add_argument('--top_k', action='store', dest="top_k", type=int, default=5)
    parser.add_argument('--category_names', action='store', dest='category_names', type=str, default='cat_to_name.json')
    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    model = getattr(torchvision.models, checkpoint['pretrained_model'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.hidden_units = checkpoint['hidden_units']
    model.learning_rate = checkpoint['learning_rate']
    model.epochs = checkpoint['epochs']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.input_size = checkpoint['input_size']
    model.output_size = checkpoint['output_size']

    return model

def process_image(image):

    (w, h) = image.size

    if h > w:
        h = int(max(h * 256 / w, 1))
        w = int(256)
    else:
        w = int(max(w * 256 / h, 1))
        h = int(256)

    im = image.resize((w, h))

    left = (w - 224) / 2
    top = (h - 224) / 2
    right = left + 224
    bottom = top + 224

    im = im.crop((left, top, right, bottom))

    im = np.array(im)
    im = im / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std
    im = np.transpose(im, (2, 0, 1))

    return im

def imshow(image, ax=None, title=None):

    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

def predict(image_path, model, top_k):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model.to(device)

    img = Image.open(image_path)
    img = process_image(img)
    img = torch.from_numpy(img)
    img = img.unsqueeze_(0)
    img = img.float()

    with torch.no_grad():
        output = model.forward(img.cuda())

    p = F.softmax(output.data, dim = 1)

    top_p = np.array(p.cpu().topk(top_k)[0][0])

    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [np.int(index_to_class[each]) for each in np.array(p.cpu().topk(top_k)[1][0])]

    return top_p, top_classes

def load_names(category_names_file):
    with open(category_names_file) as file:
        category_names = json.load(file)
    return category_names

def main():

    args = parse_args()
    checkpoint = args.checkpoint
    top_k = args.top_k
    category_names = args.category_names
    gpu = args.gpu

    path = r"C:\Users\\surit\Udacity\ImageClassifier\flowers\test"
    folders = os.listdir(path)
    randFolder = random.choice(folders)

    folderPath = '{}\{}'.format(path, randFolder)
    # print(folderPath)

    images = os.listdir(folderPath)
    randImage = random.choice(images)

    image_path = '{}\{}'.format(folderPath, randImage)
    # print(image_path)

    model = load_checkpoint(checkpoint)
    top_p, classes = predict(image_path, model, top_k)
    category_names = load_names(category_names)

    labels = [category_names[str(index)] for index in classes]

    print(f"File: {image_path}")

    for i in range(len(labels)):
        print("{} - {} - probability: {:.5f}%".format((i+1), labels[i], top_p[i] * 100))

    maxI = classes[0]

    fig = plt.figure(figsize=(8,8))
    ax1 = plt.subplot2grid((15,9), (0,0), colspan=9, rowspan=9)
    ax2 = plt.subplot2grid((15,9), (9,2), colspan=5, rowspan=5)

    img = Image.open(image_path)
    ax1.axis('off')
    ax1.set_title(category_names[str(maxI)])
    ax1.imshow(img)
    labels = [category_names[str(index)] for index in classes]
    y_pos = np.arange(5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.invert_yaxis()
    ax2.set_xlabel('Probability')
    ax2.set_ylabel('Flower Types')
    ax2.barh(y_pos, top_p, xerr=0, align='center')

    plt.show()

if __name__ == "__main__":
    main()
