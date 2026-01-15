from PIL import Image
import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms.v2 as tfs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR_WEIGTH = os.path.join(BASE_DIR, "..","Datasets","model_sun.tar")
DATASET_DIR_TEST = os.path.join(BASE_DIR, "..","Datasets","dataset_reg", "test")

model=nn.Sequential(
    nn.Conv2d(3, 32 , 3, padding='same'), # 3-число цветовых каналов, 32-число фильтров для обработки изображения, 3-размер каждого фильтра (batch,32,256,256)
    nn.ReLU(),
    nn.MaxPool2d(2), # 2-умешьнаются линейные размеры карты призванов в два раза (32, 128, 128) т.к входное 256*256, 32-количество фильтров (batch,32,128,128)
    nn.Conv2d(32, 8, 3, padding='same'), #(batch,8,128,128)
    nn.ReLU(),
    nn.MaxPool2d(2), #(batch,8,64,64)
    nn.Conv2d(8, 4, 3, padding='same'), # (batch,4, 64,64)
    nn.ReLU(),
    nn.MaxPool2d(2), # (batch,4, 32,32)
    nn.Flatten(), # преобразвание таблицы в одномерный тензор (batch, 4096)
    nn.Linear(4096, 128), # Первый слой нейроннов
    nn.ReLU(),
    nn.Linear(128,2) # Выходной слой нейронов выводит (x,y)
)

path=DATASET_DIR_TEST
num_img=11

st=torch.load(DATASET_DIR_WEIGTH, weights_only=False)
model.load_state_dict(st)

with open(os.path.join(path, "format.json"), "r") as fp:
    format=json.load(fp)

transforms=tfs.Compose([tfs.ToImage(), tfs.ToDtype(torch.float32, scale=True)])
img=Image.open(os.path.join(path, f'sun_reg_{num_img}.png')).convert('RGB')
img_t=transforms(img).unsqueeze(0)

model.eval()
predict=model(img_t)
print(predict)
print(tuple(format.values())[num_img-1])
p=predict.detach().squeeze().numpy()
plt.imshow(img)
plt.scatter(p[0], p[1], s=15, c='r')
plt.show()
