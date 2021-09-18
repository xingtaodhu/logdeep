import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
from torch.utils.data import DataLoader
import logdeep.models.causal_convolution_layer


class Transformer(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Transformer, self).__init__()
        self.input_embedding = logdeep.models.causal_convolution_layer.context_embedding(input_size, hidden_size, 9)
        self.positional_embedding = torch.nn.Embedding(hidden_size * 2, hidden_size)
        self.decode_layer = torch.nn.TransformerEncoderLayer(d_model=hidden_size, nhead=1)
        self.transformer_decoder = torch.nn.TransformerEncoder(self.decode_layer, num_layers=num_layers)
        # self.fc1 = torch.nn.Linear(hidden_size, num_keys)

    def forward(self, z):
        # concatenate observed points and time covariate
        # (B*feature_size*n_time_points)
        # z = torch.cat((y.unsqueeze(1), x.unsqueeze(1)), 1)
        # input_embedding returns shape (Batch size,embedding size,sequence len) -> need (sequence len,Batch size,
        # embedding_size)
        z = z.permute(0, 2, 1)
        z_embedding = self.input_embedding(z).permute(2, 0, 1)  # 50 * 256 *128
        # get my positional embeddings (Batch size, sequence_len, embedding_size) -> need (sequence len,Batch size,
        # embedding_size)
        # 256 50 300
        # 2048 * 10 * 1
        # 256 300 50 128

        # if use robustlog
        # positional_embeddings = self.positional_embedding(
        #     torch.mean(z.squeeze(dim=1), dim=1).type(torch.long) + 1).permute(1, 0, 2)

        # use deeplog and loganormaly
        positional_embeddings = self.positional_embedding(z.squeeze(dim=1).type(torch.long)).permute(1, 0, 2)


        input_embedding = z_embedding + positional_embeddings
        transformer_embedding = self.transformer_decoder(input_embedding)
        output = transformer_embedding.permute(1, 0, 2)
        # output = self.fc1(transformer_embedding.permute(1, 0, 2)[:, -1, :])
        return output


class deeplog(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(deeplog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_keys = num_keys
        self.transformer = Transformer(input_size,
                                       hidden_size,
                                       num_layers)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, features, device):
        input0 = features[0]
        # h0 = torch.zeros(self.num_layers, input0.size(0),
        #                  self.hidden_size).to(device)
        # c0 = torch.zeros(self.num_layers, input0.size(0),
        #                  self.hidden_size).to(device)
        out = self.transformer(input0)  # 2048 * 10 * 1
        out = self.fc(out[:, -1, :])
        return out


class loganomaly(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(loganomaly, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_keys = num_keys
        self.transformer0 = Transformer(input_size,
                                        hidden_size,
                                        num_layers)
        self.transformer1 = Transformer(input_size,
                                        hidden_size,
                                        num_layers)
        self.fc = nn.Linear(2 * hidden_size, num_keys)
        self.attention_size = self.hidden_size

        self.w_omega = Variable(
            torch.zeros(self.hidden_size, self.attention_size))
        self.u_omega = Variable(torch.zeros(self.attention_size))

        self.sequence_length = 28

    def attention_net(self, lstm_output):
        output_reshape = torch.Tensor.reshape(lstm_output,
                                              [-1, self.hidden_size])
        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        attn_hidden_layer = torch.mm(
            attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer),
                                    [-1, self.sequence_length])
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        alphas_reshape = torch.Tensor.reshape(alphas,
                                              [-1, self.sequence_length, 1])
        state = lstm_output
        attn_output = torch.sum(state * alphas_reshape, 1)
        return attn_output

    def forward(self, features, device):
        input0, input1 = features[0], features[1]

        out0 = self.transformer0(input0)
        out1 = self.transformer1(input1)

        multi_out = torch.cat((out0[:, -1, :], out1[:, -1, :]), -1)
        out = self.fc(multi_out)

        # h0_0 = torch.zeros(self.num_layers, input0.size(0),
        #                    self.hidden_size).to(device)
        # c0_0 = torch.zeros(self.num_layers, input0.size(0),
        #                    self.hidden_size).to(device)
        #
        # out0, _ = self.lstm0(input0, (h0_0, c0_0))
        #
        # h0_1 = torch.zeros(self.num_layers, input1.size(0),
        #                    self.hidden_size).to(device)
        # c0_1 = torch.zeros(self.num_layers, input1.size(0),
        #                    self.hidden_size).to(device)
        #
        # out1, _ = self.lstm1(input1, (h0_1, c0_1))
        # multi_out = torch.cat((out0[:, -1, :], out1[:, -1, :]), -1)
        # out = self.fc(multi_out)
        return out


class robustlog(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(robustlog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_keys = num_keys
        self.transformer = Transformer(input_size,
                                       hidden_size,
                                       num_layers)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, features, device):
        input0 = features[0]
        # h0 = torch.zeros(self.num_layers, input0.size(0),
        #                  self.hidden_size).to(device)
        # c0 = torch.zeros(self.num_layers, input0.size(0),
        #                  self.hidden_size).to(device)
        # out, _ = self.lstm(input0, (h0, c0))
        out = self.transformer(input0)
        out = self.fc(out[:, -1, :])
        return out
