# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/11_transforms.sequence.ipynb (unless otherwise specified).

__all__ = ['space_adder', 'SpaceTransform', 'HFTokenizerWrapper', 'HFPoolingTransform']

# Cell
#export

from itertools import islice
import pandas as pd
import numpy as np

from fastai.text.all import *


# Cell

def space_adder(seq):

    return ' '.join(seq)


class SpaceTransform(Transform):
    """Adds spaces between AAs for HuggingFace"""

    def encodes(self, x):
        if type(x) == str:
            return space_adder(x)

        return L(space_adder(seq) for seq in x)

    def decodes(self, x):

        if type(x) == str:
            return

        return [seq.replace(' ', '') for seq in x]


# Cell

class HFTokenizerWrapper(Transform):

    def __init__(self, tokenizer, tokens_only = True,
                 truncation = True, max_length = 128,
                 padding = 'max_length',
                 skip_special_tokens = True,
                 device = 'cuda'):
        self.tokenizer = tokenizer
        self.tokens_only = tokens_only
        self.truncation = truncation
        self.max_length = max_length
        self.padding = padding
        self.skip_special_tokens = skip_special_tokens
        self.device = device

    def encodes(self, x):

        if type(x) == str:
            x = [x]

        tokenized = self.tokenizer(list(x),
                                   return_tensors='pt',
                                   padding=self.padding,
                                   truncation = self.truncation,
                                   max_length = self.max_length)
        tokenized = tokenized.to(self.device)



        if self.tokens_only:
            return tokenized['input_ids']
        else:
            return [fastuple(tokenized['input_ids'][i], tokenized['attention_mask'][i]) for i in range(len(x))]


    def decodes(self, x):

        return self.tokenizer.batch_decode(x, skip_special_tokens = self.skip_special_tokens)

# Cell

from ..models.glp import model_mask_pooling

class HFPoolingTransform(Transform):

    def __init__(self, model, bs = 32):

        self.model = model
        self.bs = bs

    def encodes(self, x):

        if type(x[0]) == fastuple:
            input_ids, attention = zip(*x)
            input_ids = torch.vstack(input_ids)
            attention = torch.vstack(attention).type(torch.bool)
        else:
            input_ids = x
            attention = x == 0

        out = model_mask_pooling(input_ids, attention, self.model, bs = self.bs)
        #print(out.shape)
        if out.shape[0] == 1:
            out = torch.squeeze(out, 0)

        return out
