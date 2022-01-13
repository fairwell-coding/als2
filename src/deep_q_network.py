from collections import OrderedDict

from torch import nn


class DeepQNet(nn.Module):
    def __init__(self, h, w, image_stack, num_actions):
        super(DeepQNet, self).__init__()

        num_output_filters = 64

        self.conv_net = nn.Sequential(OrderedDict([
            ('conv2d_1',
             nn.Conv2d(image_stack, 32, kernel_size=(3, 3), stride=(1, 1), padding='same', padding_mode='zeros')),
            ('relu_1', nn.ReLU()),
            ('max2d_pooling_1', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))),
            ('conv2d_2', nn.Conv2d(32, num_output_filters, kernel_size=(3, 3), stride=(1, 1), padding='same',
                                   padding_mode='zeros')),
            ('relu_2', nn.ReLU()),
            ('max2d_pooling_2', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))),
            ('flatten', nn.Flatten())
        ]))

        num_hidden = 128

        self.dense_net = nn.Sequential(OrderedDict([
            ('dense_1', nn.Linear(int(num_output_filters * (h / 4) * (w / 4)), num_hidden)),
            ('dense_2', nn.Linear(num_hidden, num_actions))
        ]))

    def forward(self, x):
        return self.dense_net(self.conv_net(x))
