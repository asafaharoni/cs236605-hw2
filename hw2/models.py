import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Block, Linear, ReLU, Sigmoid, Dropout, Sequential


class MLP(Block):
    """
    A simple multilayer perceptron model based on our custom Blocks.
    Architecture is (with ReLU activation):

        FC(in, h1) -> ReLU -> FC(h1,h2) -> ReLU -> ... -> FC(hn, num_classes)

    Where FC is a fully-connected layer and h1,...,hn are the hidden layer
    dimensions.
    If dropout is used, a dropout layer is added after every activation
    function.
    """
    def __init__(self, in_features, num_classes, hidden_features=(),
                 activation='relu', dropout=0, **kw):
        super().__init__()
        """
        Create an MLP model Block.
        :param in_features: Number of features of the input of the first layer.
        :param num_classes: Number of features of the output of the last layer.
        :param hidden_features: A sequence of hidden layer dimensions.
        :param activation: Either 'relu' or 'sigmoid', specifying which 
        activation function to use between linear layers.
        :param: Dropout probability. Zero means no dropout.
        """
        blocks = []

        # TODO: Build the MLP architecture as described.
        # ====== YOUR CODE: ======
        activation_layers = {'relu': ReLU, 'sigmoid': Sigmoid}
        activation = activation_layers[activation]
        aux_list = [in_features] + list(hidden_features)
        nn_architecture = [(aux_list[i], aux_list[i+1]) for i in range(len(aux_list)-1)]
        for din, dout in nn_architecture:
            blocks.append(Linear(din, dout))
            blocks.append(activation())
            if dropout != 0:
                blocks.append(Dropout(dropout))
        blocks.append(Linear(aux_list[-1], num_classes))

        # ========================

        self.sequence = Sequential(*blocks)

    def forward(self, x, **kw):
        return self.sequence(x, **kw)

    def backward(self, dout):
        return self.sequence.backward(dout)

    def params(self):
        return self.sequence.params()

    def train(self, training_mode=True):
        self.sequence.train(training_mode)

    def __repr__(self):
        return f'MLP, {self.sequence}'


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(Conv -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param filters: A list of of length N containing the number of
            filters in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        # ====== YOUR CODE: ======
        self.padding = 1
        self.kernels = {
            'Conv': 3,
            'Pool': 2
        }
        # ========================

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        # Pooling to reduce dimensions.
        # ====== YOUR CODE: ======

        for i, filter in enumerate(self.filters):
            layers.append(nn.Conv2d(in_channels, filter, self.kernels['Conv'], padding=self.padding))
            layers.append(nn.ReLU())
            if (i + 1) % self.pool_every == 0:
                layers.append(nn.MaxPool2d(self.kernels['Pool']))
            in_channels = filter
        if len(self.filters) % self.pool_every != 0:
            layers.append(nn.MaxPool2d(self.kernels['Pool']))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # ====== YOUR CODE: ======

        # calculate in_channels
        for i, filter in enumerate(self.filters):
            in_channels = filter

            #conv layers:
            in_h = int((in_h + 2 * self.padding - (self.kernels['Conv'] - 1) - 1) / 1 + 1)
            in_w = int((in_w + 2 * self.padding - (self.kernels['Conv'] - 1) - 1) / 1 + 1)
            if (i + 1) % self.pool_every == 0:
                #maxpool layers
                in_h = int((in_h - (self.kernels['Pool'] - 1) - 1) / self.kernels['Pool'] + 1)
                in_w = int((in_w - (self.kernels['Pool'] - 1) - 1) / self.kernels['Pool'] + 1)
        if len(self.filters) % self.pool_every != 0:
            # maxpool layers
            in_h = int((in_h - (self.kernels['Pool'] - 1) - 1) / self.kernels['Pool'] + 1)
            in_w = int((in_w - (self.kernels['Pool'] - 1) - 1) / self.kernels['Pool'] + 1)
        in_channels = int(in_w * in_h * in_channels)

        # append linear layers
        for i, dim in enumerate(self.hidden_dims):
            layers.append(nn.Linear(in_channels, dim))
            layers.append(nn.ReLU())
            in_channels = dim
        layers.append(nn.Linear(in_channels, self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input, run the classifier on them and
        # return class scores.
        # ====== YOUR CODE: ======
        features = self.feature_extractor(x)
        out = self.classifier(features.view(features.shape[0], -1))
        # ========================
        return out


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        super().__init__(in_size, out_classes, filters, pool_every, hidden_dims)
        self.dropout = nn.Dropout(0.2)

    # TODO: Change whatever you want about the ConvClassifier to try to
    # improve it's results on CIFAR-10.
    # For example, add batchnorm, dropout, skip connections, change conv
    # filter sizes etc.
    # ====== YOUR CODE: ======
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features.view(features.shape[0], -1))

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        for i, filter in enumerate(self.filters):
            layers.append(nn.Conv2d(in_channels, filter, self.kernels['Conv'], padding=self.padding))
            layers.append(nn.ReLU())
            if (i + 1) % self.pool_every == 0:
                layers.append(nn.MaxPool2d(self.kernels['Pool']))
                layers.append(nn.BatchNorm2d(filter))
            in_channels = filter
        if len(self.filters) % self.pool_every != 0:
            layers.append(nn.MaxPool2d(self.kernels['Pool']))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = self._get_size_at_depth(len(self.filters))
        layers = []
        in_channels = int(in_w * in_h * in_channels)

        # append linear layers
        for i, dim in enumerate(self.hidden_dims):
            layers.append(nn.Linear(in_channels, dim))
            layers.append(nn.ReLU())
            in_channels = dim
        layers.append(nn.Linear(in_channels, self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _get_size_at_depth(self, d):
        in_channels, in_h, in_w, = tuple(self.in_size)
        for i, filter in enumerate(self.filters):
            if d == i:
                return in_channels, in_h, in_w
            in_channels = filter
            #conv layers:
            in_h = int((in_h + 2 * self.padding - (self.kernels['Conv'] - 1) - 1) / 1 + 1)
            in_w = int((in_w + 2 * self.padding - (self.kernels['Conv'] - 1) - 1) / 1 + 1)
            if (i + 1) % self.pool_every == 0:
                #maxpool layers
                in_h = int((in_h - (self.kernels['Pool'] - 1) - 1) / self.kernels['Pool'] + 1)
                in_w = int((in_w - (self.kernels['Pool'] - 1) - 1) / self.kernels['Pool'] + 1)
        if len(self.filters) % self.pool_every != 0:
            # maxpool layers
            in_h = int((in_h - (self.kernels['Pool'] - 1) - 1) / self.kernels['Pool'] + 1)
            in_w = int((in_w - (self.kernels['Pool'] - 1) - 1) / self.kernels['Pool'] + 1)
        return in_channels, in_h, in_w

    # ========================

