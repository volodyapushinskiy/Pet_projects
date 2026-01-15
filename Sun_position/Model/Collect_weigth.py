import torch
import os
import json
from PIL import Image
import torch.utils.data as data
import torchvision.transforms.v2 as tfs
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..","Datasets","dataset_reg")

class SunDataset(data.Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path=os.path.join(path, "train" if train else "test")
        self.transform=transform

        with open(os.path.join(self.path, "format.json"), "r") as fp:
            self.format=json.load(fp)
        
        self.lenght=len(self.format)
        self.files=tuple(self.format.keys())
        self.target=tuple(self.format.values())

    def __getitem__(self, item):
        path_file=os.path.join(self.path, self.files[item]) 
        img=Image.open(path_file).convert('RGB')

        if self.transform:
            img=self.transform(img) # tensor
        
        return img, torch.tensor(self.target[item], dtype=torch.float32)
    
    def __len__(self):
        return self.lenght
transform=tfs.Compose([tfs.ToImage(), tfs.ToDtype(torch.float32, scale=True)])
d_train=SunDataset(DATASET_DIR, train=True, transform=transform)
train_data=data.DataLoader(d_train, batch_size=32, shuffle=True)

# Сверточная нейронаая сеть прямого распространения
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
).to(device)

# Создаем оптимизатор и функцию ошибки
optimizer=optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.001) #  weight_decay=0.001 - L2 регуляризация 
loss_function=nn.MSELoss()

epoch=5
model.train()

for _e in range(epoch):
    loss_mean=0
    lm_count=0

    train_tqdm=tqdm(train_data)
    for x_train, y_train in train_tqdm:
        x_train=x_train.to(device)
        y_train=y_train.to(device)
        predict=model(x_train)
        loss=loss_function(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count+=1
        loss_mean=1/lm_count * loss.item() + (1-1/lm_count)*loss_mean
        train_tqdm.set_description(f"Epoch [{_e}/{epoch}], loss_mean={loss_mean:.3f}")

st=model.state_dict()
torch.save(st, 'model_sun.tar') # Сохранили веса обучнной модели

model.eval()

d_test=SunDataset(DATASET_DIR, train=False, transform=transform)
test_data=data.DataLoader(d_test, batch_size=32, shuffle=False)

Q=0
count=0
test_tqdm=tqdm(test_data)
for x_test, y_test in test_tqdm:
    x_test=x_test.to(device)
    y_test=y_test.to(device)
    with torch.no_grad():
        pred=model(x_test)
        Q+=loss_function(pred, y_test).item()
        count+=1
Q=Q/count
print(Q)






