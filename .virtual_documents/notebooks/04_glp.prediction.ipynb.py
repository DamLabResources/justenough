# default_exp glp.prediction

get_ipython().run_line_magic("reload_ext", " autoreload")
get_ipython().run_line_magic("autoreload", " 2")


import sys
sys.path.append('../')


#hide
#export

from itertools import islice
import pandas as pd
import numpy as np

from transformers import AutoModel, AutoTokenizer
from umap import UMAP
from fastai.text.all import *
import hdbscan

from justenough.nlp.core import *
from justenough.explain.core import *


df = pd.read_csv('../tutorials/HIV_tat_example.csv').dropna(subset = ['sample_tissue'])
df['sequence'] = df['sequence'].str.strip('*')
df.head()





model_name = 'Rostlab/prot_bert'
device = 'cuda'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)


#export


def space_adder(seq):
    
    return ' '.join(seq)


class SpaceTransform(Transform):
    """Adds spaces between AAs for HuggingFace"""
    
    def encodes(self, x):
        if type(x) == str:
            x = [x]
        return L(space_adder(seq) for seq in x)
    
    def decodes(self, x):
        
        return [seq.replace(' ', '') for seq in x]
        
        
    


test_eq(space_adder('MIVLR'), 'M I V L R')



space_tfm = SpaceTransform()

pipe = Pipeline([space_tfm])

tst = ['MIVLR', 'AAR']
cor = ['M I V L R', 'A A R']

test_eq(pipe(tst), cor)


#export

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
        
        



space_tfm = SpaceTransform()
token_tfm = HFTokenizerWrapper(tokenizer, max_length=7, device = 'cpu')
pipe = Pipeline([space_tfm, token_tfm])

tst = ['MIVLR', 'AAR']
cor = [[2, 21, 11,  8, 5, 13, 3], 
       [2,  6,  6, 13, 3,  0, 0]]

test_eq(pipe(tst), tensor(cor))


test_eq(pipe.decode(tensor(cor)), tst)


token_tfm = HFTokenizerWrapper(tokenizer, max_length=6, tokens_only=False, device = 'cpu')
pipe = Pipeline([space_tfm, token_tfm])
pipe(tst)


# export
from fastprogress.fastprogress import master_bar, progress_bar


class HFPoolingTransform(Transform):
    
    def __init__(self, model, batch_size = 32, progress = False):
        
        self.model = model
        self.batch_size = batch_size
        self.progress = progress
    
    def encodes(self, x):
        
        if type(x[0]) == fastuple:
            input_ids, attention = zip(*x)
            input_ids = torch.vstack(input_ids)
            attention = torch.vstack(attention).type(torch.bool)
        else:
            input_ids = x
            attention = x get_ipython().getoutput("= 0")
            
        with torch.no_grad():
            
            if self.batch_size is not None:
                #print(input_ids.shape)
                out = []
                if self.progress:
                    it = progress_bar(range(0, input_ids.shape[0], self.batch_size))
                else:
                    it = range(0, input_ids.shape[0], self.batch_size)
                for start in it:            
                    res = self.model(input_ids = input_ids[start:start+self.batch_size],
                                     attention_mask = attention[start:start+self.batch_size])
                    out.append(masked_concat_pool(res[0], attention[start:start+self.batch_size], input_ids.shape[1]-1))
                return torch.vstack(out)
            else:
                res = self.model(input_ids = input_ids,
                                 attention_mask = attention)
                return masked_concat_pool(res[0], attention, 
                                          input_ids.shape[1]-1)
        




token_tfm = HFTokenizerWrapper(tokenizer, max_length=6, tokens_only=True, device = 'cuda')
bert_pool_tfm = HFPoolingTransform(model)
pipe = Pipeline([space_tfm, token_tfm, bert_pool_tfm])

encoded = pipe(tst*100)

test_eq(encoded.shape, (200, 3072))


class HFBertDataLoaders(DataLoaders):
    
    @staticmethod
    def from_df(frame, tokenizer, model, sequence_col = 'sequence', label_col = None, vocab=None,
                max_length = 128, device = 'cuda', bs = 32, precompute = True,
                splitter = None, num_workers = 0):
        
        if splitter is None:
            splitter = RandomSplitter()
            
            
        seq_tfms = [ColReader(sequence_col),
                    SpaceTransform(),
                    HFTokenizerWrapper(tokenizer, 
                                       max_length=max_length, 
                                       tokens_only=False, 
                                       device = device),
                    HFPoolingTransform(model,batch_size=bs)]
        if label_col is None:
            label_tfms = seq_tfms
        else:
            label_tfms = [ColReader(label_col), Categorize(vocab=vocab)]
            
        
        if precompute:
            
            seq_pipe = Pipeline(seq_tfms)
            seq_tls = seq_pipe(frame)
            
            if label_col is None:
                label_tls = seq_tls
            else:
                label_tls = TfmdLists(frame, label_tfms)
                
            tls = TfmdLists(zip(seq_tls, label_tls), [])
            train, test = splitter(tls)
            
            return DataLoaders.from_dsets(tls[train], tls[test], num_workers=0).to(device)
            
            
        else:
            
            train, test = splitter(frame)
            feat_tls = Datasets(frame, [seq_tfms, label_tfms],
                               splits = (train, test))
            
            dls = feat_tls.dataloaders(num_workers=0).to(device)
            
            return dls


dls = HFBertDataLoaders.from_df(df, tokenizer, model, 
                                label_col=None, precompute=True)
dls.one_batch()


# export

def create_bert_head(nf, n_out, lin_ftrs=None, ps=0.5, concat_pool=True, first_bn=True, bn_final=False,
                     lin_first=False, y_range=None):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and out `n_out` classes."
    lin_ftrs = [nf, 512, n_out] if lin_ftrs is None else [nf] + lin_ftrs + [n_out]
    bns = [first_bn] + [True]*len(lin_ftrs[1:])
    ps = L(ps)
    if len(ps) == 1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    #pool = AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1)
    layers = [Flatten()]
    if lin_first: layers.append(nn.Dropout(ps.pop(0)))
    for ni,no,bn,p,actn in zip(lin_ftrs[:-1], lin_ftrs[1:], bns, ps, actns):
        layers += LinBnDrop(ni, no, bn=bn, p=p, act=actn, lin_first=lin_first)
    if lin_first: layers.append(nn.Linear(lin_ftrs[-2], n_out))
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    if y_range is not None: layers.append(SigmoidRange(*y_range))
    return nn.Sequential(*layers)


# export


class ProtBertHead(Module):
    
    def __init__(self, 
                 in_features = 3072, 
                 hidden_dim = 128,
                 out_features = 'autoencoder', 
                 encoder = None,
                 lin_ftrs = [1024], ps = 0.25):
        
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        
        if out_features == 'autoencoder':
            self.out_features = in_features
        else:
            self.out_features = out_features
        
        if encoder is None:
            self.encoder = create_bert_head(in_features, hidden_dim, lin_ftrs = lin_ftrs, ps=ps)
        else:
            self.encoder = encoder
        
        self.decoder = create_bert_head(self.hidden_dim, self.out_features, lin_ftrs = lin_ftrs, ps = ps)
        
    def re_head(self, new_out_features, lin_ftrs = [1024], ps = 0.25):
        
        
        return ProtBertHead(in_features=self.in_features,
                             hidden_dim=self.hidden_dim,
                             out_features = new_out_features,
                             encoder = self.encoder,
                             lin_ftrs = lin_ftrs, ps = ps)
        
        
    def forward(self, x):
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return decoded
        
                


# export

#@delegates(Learner.__init__)
def protbert_classifier_learner(dls, in_features, 
                                model = None,
                                n_out=None,
                                lin_ftrs=None, 
                                max_len=128, hidden_dim = 128,
                                ps = 0.25,
                                y_range=None, **kwargs):
    "Create a `Learner` with a ProtBert classifier from `dls` and `arch`."
        
    if model is None:
        if n_out == 'autoencoder': n_out = in_features
        if n_out is None: n_out = get_c(dls)
        if n_out is None: n_out = in_features
            
        model = ProtBertHead(in_features = in_features,
                              out_features= n_out,
                              hidden_dim = hidden_dim,
                              lin_ftrs = lin_ftrs, ps = ps)
        
    learn = Learner(dls, model, **kwargs)
    
    return learn



df = pd.read_csv('../tutorials/HIV_tat_example.csv').dropna(subset = ['sample_tissue'])
df['sequence'] = df['sequence'].str.strip('*')
df.head()


model_name = 'Rostlab/prot_bert_bfd'
device = 'cuda'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)


auto_dls = HFBertDataLoaders.from_df(df, tokenizer, model, 
                                     label_col=None, precompute=True)





auto_learner = protbert_classifier_learner(dls, model.config.hidden_size*3, 
                                           n_out = 'autoencoder',
                                           loss_func = nn.MSELoss())
auto_learner.fit_one_cycle(50, lr_max = 0.01, cbs = [EarlyStoppingCallback(patience=1)])
auto_learner.freeze()


label_dls = HFBertDataLoaders.from_df(df, tokenizer, model, 
                                      label_col='sample_tissue', precompute=True)


vocab = df['sample_tissue'].dropna().unique()
label_learner = protbert_classifier_learner(label_dls, None,
                                            model= auto_learner.model.re_head(len(vocab), lin_ftrs=[64, 32]),
                                            loss_func = nn.CrossEntropyLoss(),  metrics=[accuracy])


label_learner.fit_one_cycle(50, lr_max = 0.001, cbs = [EarlyStoppingCallback(patience=2)])


label_learner.unfreeze()
label_learner.fit_one_cycle(50, lr_max = 0.001, cbs = [EarlyStoppingCallback(patience=2)])



