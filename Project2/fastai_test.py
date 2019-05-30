from fastai import *
# import fastai
from fastai.vision import *
from fastai.datasets import *
from fastai.basic_data import *
# import fastai.vision as fv
# import fastai.datasets as fd
# import fastai.basic_data as fb
import sys

from fastai.vision import models


def train():
    path = untar_data(URLs.MNIST_SAMPLE)
    train_data = ImageDataBunch.from_folder(path)
    learn = cnn_learner(train_data, models.resnet18(), metrics=accuracy)

    learn.lr_find()
    learn.recoder.plot
    learn.fit(1)
    # print(sys.modules['fastai'])


if __name__ == '__main__':
    train()
