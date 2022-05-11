# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.quantization import ObserverBase
from torch.quantization import PerChannelMinMaxObserver

def quantize(w, ch_axis, scale):
    # In the default behavior, max_val = 255.
    scale = torch.unsqueeze(scale, ch_axis)
    return (
        torch.clamp(torch.round(w / scale), -128, 127)
    ) * scale

class CT2_quantization_observer(ObserverBase):
    #### symmetric signed int8 quantization
    min_vals: torch.Tensor
    max_vals: torch.Tensor
    def __init__(self, ch_axis=0, dtype=torch.qint8):
        super(CT2_quantization_observer, self).__init__(dtype)
        self.ch_axis = ch_axis
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float32).eps]))
        self.register_buffer('min_vals', torch.tensor([]))
        self.register_buffer('max_vals', torch.tensor([]))

    def forward(self, x_orig):
        x = x_orig.detach()
        min_vals = self.min_vals
        max_vals = self.max_vals
        x = x.to(self.min_vals.dtype)
        max_vals = torch.amax(torch.absolute(x), axis=self.ch_axis)
        self.max_vals.resize_(max_vals.shape)
        self.max_vals.copy_(max_vals)
        return x_orig
    
    def _calculate_qparams(self, min_val: torch.Tensor, max_val: torch.Tensor):
        scale = 2 * max_val / (127+128)
        scale = torch.where(scale > self.eps, scale, torch.ones_like(scale))  
        return scale

    @torch.jit.export
    def calculate_qparams(self):
        return self._calculate_qparams(self.min_vals, self.max_vals)

def emulate_int8_ct2(w, ch_axis, scale=None, bits=8):

    ### only work for 2 dimension matrix

    obs = CT2_quantization_observer(ch_axis=ch_axis)
    # obs = PerChannelMinMaxObserver(ch_axis=ch_axis, dtype=torch.qint8,
    #              qscheme=torch.per_channel_symmetric, reduce_range=True, 
    #              quant_min=None, quant_max=None, factory_kwargs=None)
    obs.to(device=w.device)
    #print("device: ", w.device)
    _ = obs(w)
    #scale, zero_point = obs.calculate_qparams()
    scale = obs.calculate_qparams()
    # print("quant_min:", obs.quant_min, "quant_max: ", obs.quant_max)
    # print("zero_point", zero_point)
    # print("scale size", scale.size())
    scale = scale.cuda().type_as(w)
    return quantize(w, ch_axis, scale)

class IntEmbedding(nn.Module):
    """
    Quantized counterpart of the nn.Embedding module that applies QuantNoise during training.

    Args:
        - num_embeddings: number of tokens
        - embedding_dim: embedding dimension
        - p: amount of noise to inject (0 = no quantization, 1 = quantize all the weights)
        - bits: number of bits
        - method: choose among {"tensor", "histogram", "channel"}
        - update_step: recompute scale and zero_point every update_steps iterations

    Remarks:
        - We use the straight-through estimator so that the gradients
          back-propagate nicely in the network, this is implemented with
          the detach() trick
        - Parameters scale and zero_point are recomputed every update_step
          forward pass to reduce the overhead
        - At test time, the weights are fully quantized
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        p=0,
        update_step=1000,
        bits=8,
        method="histogram",
    ):
        super(IntEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert (
                    padding_idx < self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
            elif padding_idx < 0:
                assert (
                    padding_idx >= -self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [
                num_embeddings,
                embedding_dim,
            ], "Shape of weight does not match num_embeddings and embedding_dim"
            self.weight = nn.Parameter(_weight)
        self.sparse = sparse

        # quantization parameters
        self.p = p
        self.bits = bits
        self.method = method
        self.update_step = update_step
        self.counter = 0

    def reset_parameters(self):
        nn.init.normal_(self.weight)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input):

        weight = self.weight

        if self.training:
            p = self.p * (1-torch.exp(torch.tensor(-self.counter/self.update_step)))
            self.counter += 1
            # quantize weight
            weight_quantized = emulate_int8_ct2(self.weight, 1).type_as(self.weight)
            # mask to apply noise
            mask = torch.zeros_like(self.weight)
            mask.bernoulli_(1 - p)
            noise = (weight_quantized - self.weight).masked_fill(mask.bool(), 0)
            # using straight-through estimator (STE)
            weight = (self.weight + noise.detach())
            
        # return output
        output = F.embedding(
            input,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return output

    def extra_repr(self):
        s = "{num_embeddings}, {embedding_dim}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        if self.max_norm is not None:
            s += ", max_norm={max_norm}"
        if self.norm_type != 2:
            s += ", norm_type={norm_type}"
        if self.scale_grad_by_freq is not False:
            s += ", scale_grad_by_freq={scale_grad_by_freq}"
        if self.sparse is not False:
            s += ", sparse=True"
        s += "quant_noise={p}, bits={bits}, method={method}"
        return s.format(**self.__dict__)
