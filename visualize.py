import pickle
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import sys
from models import CAE, MotionData




def main(dataname):

    model = CAE()
    name = "CAE"
    
    model_dict = torch.load('models/CAE_model', map_location='cpu')
    model.load_state_dict(model_dict)
    model.eval

    EPOCH = 1
    BATCH_SIZE_TRAIN = 49950

    # dataname = "demonstrations/demo_00_02.pkl"
    save_model_path = "models/" + name + "_model"
    best_model_path = "models/" + name + "_best_model"

    train_data = MotionData(dataname)
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    for epoch in range(EPOCH):
      epoch_loss = 0.0
      for batch, x in enumerate(train_set):
          # loss = model(x) 
          z = model.encoder(x).tolist()          
          print(len(z))
      # print(epoch, loss.item())
    data = np.asarray(z)
    return data 
    # pickle.dump(model.zl, open("z_dis.pkl", "wb"))
    





if __name__ == "__main__":
    data1 = main("demonstrations/demo_00_02.pkl")
    data2 = main("demonstrations/demo_01_02.pkl")
    print(len(data1))
    print(len(data2))
    # data1 = pickle.load(open('z_dis.pkl', "rb"))
    sns.distplot(data1,label='left')
    sns.distplot(data2,label='right')
    plt.legend()
    plt.show()
