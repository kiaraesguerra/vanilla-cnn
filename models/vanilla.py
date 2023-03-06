import torch.nn as nn
import torch
import sys
sys.path.append("../")
import _ext.nn as enn


__all__ = ['van32', 'van128', 'van200', 'van256', 'van300', 'van512', 'van2048',
           'van4096', 'van8192']


class Vanilla(nn.Module):
    def __init__(self, base, c, num_classes=10,  conv_init='conv_delta_orthogonal', gain=1):
        super(Vanilla, self).__init__()
        self.init_supported = ['conv_delta_orthogonal', 'conv_delta_orthogonal_relu', 'kaiming_normal']
        self.gain = gain
        if conv_init in self.init_supported:
            self.conv_init = conv_init
        else:
            print('{} is not supported'.format(conv_init))
            self.conv_init = 'kaiming_normal'
        print('initialize conv by {}'.format(conv_init))
        print(f'The gain used is {gain}')
        self.base = base
        self.fc = nn.Linear(c, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.conv_init == self.init_supported[0]:
                    enn.init.conv_delta_orthogonal_(m.weight, gain)
                elif self.conv_init == self.init_supported[1]:
                    if m.in_channels == 3:
                        enn.init.conv_delta_orthogonal_(m.weight, gain)
                    else:
                        enn.init_relu.conv_delta_orthogonal_relu_(m.weight, gain)      
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain)

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def make_layers(depth, c, activation):
    assert isinstance(depth, int)
    
    if activation == 'tanh':
        act = nn.Tanh()
    elif activation == 'relu':
        act = nn.ReLU()
    
    layers = []
    in_channels = 3
    for stride in [1, 2, 2]:
        conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1, stride=stride)
        layers += [conv2d, act]
        in_channels = c
    for _ in range(depth):
        conv2d = nn.Conv2d(c, c, kernel_size=3, padding=1)
        layers += [conv2d, act]
    layers += [nn.AvgPool2d(8)] # For mnist is 7
    return nn.Sequential(*layers), c


def van32(c, activation, **kwargs):
    """Constructs a 32 layers vanilla model.
    """
    model = Vanilla(*make_layers(32, c, activation), **kwargs)
    return model


def van128(c,activation,**kwargs):
    """Constructs a 128 layers vanilla model.
    """
    model = Vanilla(*make_layers(128, c, activation), **kwargs)
    return model

def van200(c,activation,**kwargs):
    """Constructs a 200 layers vanilla model.
    """
    model = Vanilla(*make_layers(200, c, activation), **kwargs)
    return model


def van256(c,activation,**kwargs):
    """Constructs a 256 layers vanilla model.
    """
    model = Vanilla(*make_layers(256, c, activation), **kwargs)
    return model

def van300(c,activation,**kwargs):
    """Constructs a 300 layers vanilla model.
    """
    model = Vanilla(*make_layers(300, c, activation), **kwargs)
    return model


def van512(c,activation,**kwargs):
    """Constructs a 512 layers vanilla model.
    """
    model = Vanilla(*make_layers(512, c, activation), **kwargs)
    return model


def van2048(c,activation,*kwargs):
    """Constructs a 2048 layers vanilla model.
    """
    model = Vanilla(*make_layers(2048, c, activation), **kwargs)
    return model


def van4096(c,activation,**kwargs):
    """Constructs a 4096 layers vanilla model.
    """
    model = Vanilla(*make_layers(4096, c, activation), **kwargs)
    return model


def van8192(c,activation,**kwargs):
    """Constructs a 8192 layers vanilla model.
    """
    model = Vanilla(*make_layers(8192, c, activation), **kwargs)
    return model