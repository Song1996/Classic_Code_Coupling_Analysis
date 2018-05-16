import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class symcnn_model(nn.Module):

    def __init__(self, vocab_size = 3500, embedding_dim = 128, context_size = 20, conv_num_kernel = 256, fc1_num = 128, fc2_num = 32, kernel_size = 5):
        super(symcnn_model, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        #self.left_conv  = nn.Conv1d(in_channels = embedding_dim, out_channels = conv_num_kernel, kernel_size = 5)
        self.conv = nn.Conv1d(in_channels = embedding_dim, out_channels = conv_num_kernel, kernel_size = kernel_size)
        self.fc1 =  nn.Linear(2*conv_num_kernel, fc1_num)
        self.fc2 =  nn.Linear(fc1_num, fc2_num)
        self.fc3 =  nn.Linear(fc2_num, 1)

    def forward(self, inputs):
        left_x = torch.transpose(self.embeddings(inputs[0]),1,2)
        left_x = self.conv(left_x)
        left_x = F.sigmoid(left_x)
        left_x = (F.max_pool1d(left_x, kernel_size = left_x.size()[2])).squeeze()
        left_x = F.dropout(left_x, p=0.5)

        right_x = torch.transpose(self.embeddings(inputs[1]),1,2)
        right_x = self.conv(right_x)
        right_x = F.sigmoid(right_x)
        right_x = (F.max_pool1d(right_x, kernel_size = right_x.size()[2])).squeeze()
        right_x = F.dropout(right_x, p=0.5)
        
        x = torch.cat([left_x, right_x], 1)
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = F.dropout(x, p = 0.5)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = F.dropout(x, p = 0.5)
        x = self.fc3(x)
        x = F.sigmoid(x)

        return x