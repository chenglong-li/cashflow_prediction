# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
from get_data import GetData



class CNN_Series(nn.Module):
    def __init__(self):
        super(CNN_Series, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=8,
                kernel_size=3
            ),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=8,
                out_channels=16,
                kernel_size=3
            ),
            nn.MaxPool1d(kernel_size=2)
        )
        self.fc = nn.Linear(384,1)

    def forward(self, indata):
        x = self.conv1(indata)
        print(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

    def to_variable(self, x):
        tmp = torch.Tensor(x)
        return Variable(tmp)


if __name__ == '__main__':
    model = CNN_Series()
    lossfunc = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    # y = model.series_gen()
    # In_data, Out_data = model.train_data_gen(y.numpy(), 10);

    In_data = []
    Out_data = []
    dataClass = GetData()
    day_balance= dataClass.get_day_balance()
    for idx in range(np.shape(day_balance)[0] - 5):
        In_data.append(np.array(day_balance['tBalance'][idx:idx+5]).tolist())
        Out_data.append(np.array(day_balance['tBalance'][idx+5]).tolist())


    for ite in range(5):
        print("Epoch: [{}/{}]".format(ite, 5))
        for batch_idx in range(10):
            seq = In_data[batch_idx * 30:(batch_idx + 1) * 30]
            out = Out_data[batch_idx * 30:(batch_idx + 1) * 30]
            seq = model.to_variable(np.array(seq, dtype="int64"))
            seq = seq.unsqueeze(1)
            out = model.to_variable(np.array(out, dtype="int64"))
            optimizer.zero_grad()
            modelout = model(seq)
            loss = lossfunc(modelout, out)
            print("Batch:[{}], Loss:{}".format(batch_idx, loss.data.cpu().numpy()[0]))
            loss.backward()
            optimizer.step()

    seq_test = In_data[400:]
    out_test = Out_data[400:]
    seq_test = model.to_variable(np.array(seq_test, dtype="int64"))
    seq_test = seq_test.unsqueeze(1)
    modelout_test = model(seq_test)
    loss_test = F.mse_loss(modelout_test, out_test)
    print("Test, Loss:{}".format(loss_test.data.cpu().numpy()[0]))

    fig = plt.figure()
    plt.plot(day_balance['report_date'], Out_data)
    plt.plot(day_balance['report_date'], modelout_test)
    plt.show()
