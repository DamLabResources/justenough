# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/01_models.glp.ipynb (unless otherwise specified).

__all__ = ['masked_concat_pool', 'MaskedConcatPooling', 'model_mask_pooling']

# Cell
#export

import pandas as pd
import numpy as np

from fastai.text.all import *

# Cell

def masked_concat_pool(output, mask, bptt):
    "Pool `MultiBatchEncoder` outputs into one vector [last_hidden, max_pool, avg_pool]"
    # True in mask implies MASKED and will be hidden!

    lens = output.shape[1] - mask.long().sum(dim=1)
    last_lens = mask[:,-bptt:].long().sum(dim=1)
    avg_pool = output.masked_fill(mask[:, :, None], 0).sum(dim=1)
    avg_pool.div_(lens.type(avg_pool.dtype)[:,None])
    max_pool = output.masked_fill(mask[:,:,None], -float('inf')).max(dim=1)[0]

    last_hidden = output[torch.arange(0, output.size(0)),-last_lens-1]
    x = torch.cat([last_hidden,
                   max_pool, avg_pool], 1) #Concat pooling.
    x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)

    return x

# Cell

class MaskedConcatPooling(Module):

    def __init__(self, seq_len = None, mask_is_attention = False):

        if seq_len is None:
            self.bptt = None
        else:
            self.bptt = seq_len - 1
        self.mask_is_attention = mask_is_attention


    def forward(self, x):

        x, mask = x

        if self.bptt is None:
            bptt = mask.shape[1]
        else:
            bptt = self.bptt

        if self.mask_is_attention:
            return masked_concat_pool(x, mask==False, bptt)
        else:
            return masked_concat_pool(x, mask, bptt)


# Cell

def model_mask_pooling(input_ids, attention_mask, model, bs = 32):

    with torch.no_grad():

        if bs is not None:
            out = []
            for start in range(0, input_ids.shape[0], bs):
                res = model(input_ids = input_ids[start:start+bs],
                            attention_mask = attention_mask[start:start+bs])
                out.append(masked_concat_pool(res[0], attention_mask[start:start+bs], input_ids.shape[1]-1))
            return torch.vstack(out)
        else:
            res = self.model(input_ids = input_ids,
                             attention_mask = attention)
            return masked_concat_pool(res[0], attention,
                                      input_ids.shape[1]-1)