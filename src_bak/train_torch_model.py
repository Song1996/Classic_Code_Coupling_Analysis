from model_pytorch import symcnn_model
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from create_dataset import Data_gener
import torch.autograd as autograd
import torch.optim as optim

g = Data_gener('wine')
gg = g.gener('train')

xy = next(gg)

net = symcnn_model()

optimizer = optim.SGD(net.parameters(), lr=0.01)
   # zero the gradient buffers
criterion = nn.MSELoss()

for cnt in range(20000):
    #xy = next(gg)
    x = [autograd.Variable(i) for i in xy[:2]]
    label = xy[2]
    output = net(x)
    target = autograd.Variable(label).unsqueeze(1)
    loss = criterion(output, target)
    optimizer.zero_grad()

    if cnt%10 == 0:
        print(loss)
    loss.backward()
    optimizer.step()
