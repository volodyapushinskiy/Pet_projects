import os
import numpy as np
import re

from navec import Navec
from tqdm import tqdm
import torch
import torch.utils.data as data
import torchvision
from torchvision import models
import torchvision.transforms.v2 as tfs_v2
import torch.nn as nn
import torch.optim as optim

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
DATASET_DIR_NAVEC=os.path.join(BASE_DIR, "..","Datasets","navec_hudlit_v1_12B_500K_300d_100q.tar")
DATASET_DIR_TEXT=os.path.join(BASE_DIR, "..","Datasets","text_2.txt")
class WordsDataset(data.Dataset):
    def __init__(self, path, navec_emb, prev_words=3):
        super().__init__()
        self.navec_emb=navec_emb
        self.prev_words=prev_words

        with open(path, 'r', encoding='utf-8') as fp:
            self.text=fp.read()
            self.text=self.text.replace('\ufeff', '')
            self.text=self.text.replace('\n', ' ')
            self.text=re.sub(r'^A-zА-я- ', '', self.text)

        self.words=self.text.split()
        self.words=[word for word in self.words if word in self.navec_emb]
        vocab=set(self.words)

        self.int_to_word=dict(enumerate(vocab))
        self.word_to_int={b:a for a,b in self.int_to_word.items()}
        self.vocab_size=len(vocab)

    def __getitem__(self, item):
        _data=torch.vstack([torch.tensor(self.navec_emb[self.words[x]]) for x in range(item, item+self.prev_words)])
        word=self.words[item+self.prev_words]
        t=self.word_to_int[word]
        return _data, t # _data - это набор предыдущих слов представленных в виде тензора, t нужное слово
    
    def __len__(self):
        return len(self.words)-1-self.prev_words
    

class WordsRNN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.hidden_size=64
        self.in_features=in_features
        self.out_features=out_features

        self.rnn=nn.RNN(self.in_features, self.hidden_size, batch_first=True)
        self.out=nn.Linear(self.hidden_size, self.out_features)

    def forward(self, x):
        x,h=self.rnn(x)
        y=self.out(h)
        return y
    
path=DATASET_DIR_NAVEC
navec=Navec.load(path)
    
d_train=WordsDataset(DATASET_DIR_TEXT, navec ,prev_words=3)
train_data=data.DataLoader(d_train, batch_size=8, shuffle=False)

model=WordsRNN(300, d_train.vocab_size).to(device)

optimazer=optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.0001)
loss_function=nn.CrossEntropyLoss()

model.train()
epochs=20
k=0

for _e in range(epochs):
    loss_mean=0
    lm_count=0
    train_tqdm=tqdm(train_data)
    for x_train, y_train in train_tqdm:
        
        x_train=x_train.to(device)
        y_train=y_train.to(device)
        # if k==0:
        #     print((x_train).shape)
        #     print(type(model(x_train)))
        #     print(y_train)
        #     print(type(y_train))
        # k+=1

        predict=model(x_train).squeeze(0)
        loss=loss_function(predict, y_train.long())

        optimazer.zero_grad()
        loss.backward()
        optimazer.step()

        lm_count+=1
        loss_mean=1/lm_count*loss.item()+ (1-1/lm_count)*loss_mean
        train_tqdm.set_description(f"Epoch [{_e+1}/{epochs}], loss_mean={loss_mean:.3f}")

# st=model.state_dict()
# torch.save(st, 'RNN_words.tar')


model.eval()
predict='Поел попил устал прилег'.lower().split()
total=10
for _ in range(total):
    _data=torch.vstack([torch.tensor(d_train.navec_emb[predict[-x]]) for x in range(d_train.prev_words,0,-1)]).to(device)
    pred=model(_data.unsqueeze(0)).squeeze(0).to(device)
    indx=torch.argmax(pred, dim=1)
    predict.append(d_train.int_to_word[indx.item()])

print(" ".join(predict))