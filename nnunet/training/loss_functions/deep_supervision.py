#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from torch import nn


class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        # x and y are the lists/tuples of predictions and targets
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i])
        return l

class DualTasksMultipleOutputLoss(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)

        NOTE: this function only support segmentation and classification loss

        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x_seg, x_class, y_seg, y_class):

        # x and y are the lists/tuples of predictions and targets
        assert isinstance(x_seg, (tuple, list)), "x_seg must be either tuple or list"
        assert isinstance(x_class, (tuple, list)), "x_class must be either tuple or list"
        assert isinstance(y_seg, (tuple, list)), "y_seg must be either tuple or list"
        assert isinstance(y_class, (tuple, list)), "y_class must be either tuple or list"

        # make sure they are of the same length
        assert len(x_seg) == len(y_seg), 'x_seg has different length than y_seg'
        assert len(x_class) == len(y_class), 'x_class has different length than y_class'

        # the weighting factors are shared across the two tasks
        if self.weight_factors is None:
            weights = [1] * len(x_seg)
        else:
            weights = self.weight_factors

        # this is the final prediction loss
        l = weights[0] * self.loss(x_seg[0], x_class[0], y_seg[0], x_class[0])
        for i in range(1, len(x_seg)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x_seg[i], x_class[i], y_seg[i], x_class[i])
        return l
