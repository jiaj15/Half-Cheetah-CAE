import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import sys



class MotionData(Dataset):

  def __init__(self, filename):
    self.data = pickle.load(open(filename, "rb"))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return torch.FloatTensor(self.data[idx])



class CAE(nn.Module):

  def __init__(self):
    super(CAE, self).__init__()

    self.fc1 = nn.Linear(23,64)
    self.fc2 = nn.Linear(64,64)
    self.fc3 = nn.Linear(64,1)

    self.fc4 = nn.Linear(18,64)
    self.fc5 = nn.Linear(64,64)
    self.fc6 = nn.Linear(64,6)

    self.loss_func = nn.MSELoss()
    # self.zl = []

  def encoder(self, x):
    h1 = torch.tanh(self.fc1(x))
    h2 = torch.tanh(self.fc2(h1))
    return self.fc3(h2)

  def decoder(self, z_with_state):
    h4 = torch.tanh(self.fc4(z_with_state))
    h5 = torch.tanh(self.fc5(h4))
    h6 = self.fc6(h5)
    return h6

  def forward(self, x):
    a_target = x[:, :,0:6]
    s = x[:, :,6:]
    z = self.encoder(x)
    # print(z.shape,s.shape)
    # self.zl +=z.tolist()
    # print(a_target.shape)
    z_with_state = torch.cat((z, s), 2)
    # print(z_with_state.shape)
    a_decoded = self.decoder(z_with_state)
    loss = self.loss(a_decoded, a_target)
    return loss

  def loss(self, a_decoded, a_target):
    return self.loss_func(a_decoded, a_target)
  
  def evalulate(self, x):
    with torch.no_grad():
      a_target = x[:, :,0:6]
      s = x[:, :,6:]
      z = self.encoder(x)
      # print(z.shape,s.shape)
      self.zl +=z.tolist()
      z_with_state = torch.cat((z, s), 2)
      # print(z_with_state.shape)
      a_decoded = self.decoder(z_with_state)
      loss = self.loss(a_decoded, a_target)
    return a_decoded, loss




def main():

    model = CAE()
    name = "CAE"

    EPOCH = 600
    BATCH_SIZE_TRAIN = 10000
    LR = 0.01
    LR_STEP_SIZE = 280
    LR_GAMMA = 0.1

    dataname = 'datasets/traj.pkl'
    save_model_path = "models/" + name + "_model"
    best_model_path = "models/" + name + "_best_model"

    train_data = MotionData(dataname)
    train_set = DataLoader(dataset=train_data, shuffle=True, batch_size=BATCH_SIZE_TRAIN)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    best_loss = 1e7
    for epoch in range(EPOCH):
      epoch_loss = 0.0
      for batch, x in enumerate(train_set):
          optimizer.zero_grad()
          # a_target = x[:, 0:6]
          loss = model(x)
          loss.backward()
          optimizer.step()
          epoch_loss += loss.item()
      scheduler.step()
      print(epoch, loss.item())
      if epoch % 10 == 0:
        torch.save(model.state_dict(), save_model_path)
      if epoch_loss < best_loss:
        best_loss = epoch_loss
        print('best loss!!!----loss:{}'.format(best_loss))
        torch.save(model.state_dict(), best_model_path)




if __name__ == "__main__":
    main()
