# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from .qemb import emulate_int8_ct2

class IntLinear(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        p=0,
        update_step=3000,
        bits=8,
        method="histogram",
    ):
        super(IntLinear, self).__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.chosen_bias = bias
        if self.chosen_bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        # quantization parameters
        self.p = p
        self.bits = bits
        self.method = method
        self.update_step = update_step
        self.counter = 0

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.chosen_bias:
            nn.init.constant_(self.bias, 0.0)
        return

    def forward(self, input):
        if self.training:
            p = self.p * (1-torch.exp(torch.tensor(-self.counter/self.update_step)))
            self.counter += 1
            # quantize input
            input_quantized = emulate_int8_ct2(input, -1).type_as(input)
            # mask to apply noise
            input_mask = torch.zeros_like(input)
            input_mask.bernoulli_(1 - p)
            input_noise = (input_quantized  - input).masked_fill(input_mask.bool(), 0)
            # using straight-through estimator (STE)
            input = input + input_noise.detach()
            # quantize weight
            weight_quantized = emulate_int8_ct2(self.weight, 1).type_as(self.weight)
            # mask to apply noise
            mask = torch.zeros_like(self.weight)
            mask.bernoulli_(1 - p)
            noise = (weight_quantized - self.weight).masked_fill(mask.bool(), 0)
            # using straight-through estimator (STE)
            weight = (
                self.weight + noise.detach()
            )
            # return output
            output = F.linear(input, weight, self.bias)
        else:
            weight = self.weight
            output = F.linear(input, weight, self.bias)
        return output

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}, quant_noise={}, bits={}, method={}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
            self.p,
            self.bits,
            self.method,
        )
