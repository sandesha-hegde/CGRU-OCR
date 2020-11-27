import torch.nn as nn
from torchvision.models import resnet18

import configparser
import global_config


class CGRU(nn.Module):

    def __init__(self, number_chars):
        super().__init__()
        cfg = configparser.ConfigParser()
        cfg.read(global_config.model_config)

        self.cfg = cfg

        resnet = resnet18(pretrained=True)
        self.number_chars = number_chars
        self.rnn_hidden_units = self.cfg['MODEL PARAM'].getint('rnn_hidden_units')
        self.dropout = self.cfg['MODEL PARAM'].getfloat('dropout')
        self.add_output = self.cfg['MODEL PARAM'].getboolean('add_output')

        resnet_modules = list(resnet.children())[:-3]

        self.cnn_part1 = nn.Sequential(*resnet_modules)

        self.cnn_part2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True))

        self.linear1 = nn.Linear(1024, 256)

        self.gru1 = nn.GRU(input_size=self.rnn_hidden_units,
                           hidden_size=self.rnn_hidden_units,
                           bidirectional=True,
                           batch_first=True)

        self.gru2 = nn.GRU(input_size=self.rnn_hidden_units,
                           hidden_size=self.rnn_hidden_units,
                           bidirectional=True,
                           batch_first=True)

        self.logits = nn.Linear(self.rnn_hidden_units * 2, number_chars)

    def forward(self, data):
        # print('IN ', data.size())
        conv1 = self.cnn_part1(data)
        # print('conv1',conv1.size())

        conv2 = self.cnn_part2(conv1)

        # print('before ', conv2.size())

        conv2 = conv2.permute(0, 3, 1, 2)
        # print('permute ',conv2.size())

        flatten = conv2.view(conv2.size(0), conv2.size(1), -1)
        # print('after flat ',flatten.size())

        flatten = self.linear1(flatten)
        # print('IN GRU ', flatten.size())

        out_feat, hidden_state = self.gru1(flatten)
        # print('out gru1 ', out_feat.size())

        feature_size = out_feat.size(2)

        if self.add_output:
            out_feat = out_feat[:, :, :feature_size // 2] + out_feat[:, :, feature_size // 2:]

        out_feat, hidden_state = self.gru2(out_feat)

        out = self.logits(out_feat)

        out = out.permute(1, 0, 2)

        return out
