# default_exp glp.clustering

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


df = pd.read_csv('../tutorials/HIV_tat_example.csv')
df['sequence'] = df['sequence'].str.strip('*')
df.head()


tmi = TopicModelingInterface(model_name = 'Rostlab/prot_bert')


cluster_info, emb_data = tmi.process_df(df, col = 'sequence')
cluster_info.head()


full_df = pd.concat([cluster_info, df], axis=1)
full_df.head()


from justenough.explain.core import *


from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.transform import factor_cmap
from bokeh.palettes import viridis
    
    
class BokehFigureExplanation(DataFrameExplanation):
    
    bokeh_source = None
    
    tooltips = None
    tip_cols = None
    
    factor_col = None
    factor_cmap = None
    factors = None
    
    pallette = viridis
    
    fig = None
    
    # For reference, do not overload
    def plot(self, **polish_kwargs):
        
        #Setup the parent class info if needed.
        self.setup() #which calls self._setup
        
        self.generate_bokeh()
        
        self.polish_bokeh(**polish_kwargs)
        
        return self.for_show()
        
                
    def _setup(self):
        """Subclass to setup explaination specific processes.
        
        Should at the very least set self.bokeh_source"""
        raise NotImplementedError
        
    def generate_bokeh(self):
        """Subclass to setup specific plotting"""
        raise NotImplementedError
        
                
    def polish_bokeh(self):
        """Useful to subclass for adding labels, annotations, etc. to the plot.
        
        Assume all setup, setup_bokeh, and generate_bokeh have been done.
        """
        pass
    
    
    def for_show(self):
        """Retuns an object ready to show with Bokeh.
        
        Useful to subclass if there's anything beyond returning self.fig"""
        
        return self.fig
    
    
    
    ##### Below this are utility functions that should be left alone.
    
    
    def setup(self):
        
        super().setup()
        self._setup()
    
    
    def setup_bokeh(self):
        
        triggers = [
            (self.factor_col, self._build_factors),
            (self.tip_cols, self._build_tips)
                    ]
        
        for trig, func in triggers:
            if trig is not None:
                func()
                        
        self.setup_source()
        assert self.bokeh_source is not None
    
    
    def save_png(self, path, fig_kw = {}, export_kw={}):
        
        fig = self.generate_figure(**fig_kw)
        export_png(fig, path, **export_kw)
            
    
def _build_tips(tip_cols):

    
    return [(col, '@' + col) for col in tip_cols if ' ' not in col]


def _build_factor_cmap(data, factor_col, pallette):

    factors = data[factor_col].map(str).unique().tolist()
    cmap = factor_cmap(factor_col, 
                       pallette(len(factors)),  
                       factors)
    return factors, cmap
    
    


from bokeh.plotting import output_notebook, show
output_notebook()    



#export

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CDSView, BooleanFilter
from bokeh.transform import factor_cmap
from bokeh.palettes import viridis


class ClusteringBokehExplanation(BokehFigureExplanation):
    
    
    def __init__(self, cluster_data, 
                 xy_cols = ('X', 'Y'),
                 factor_col = 'label',
                 tooltips = None,
                 tip_cols = None,
                 pallette = viridis,
                 plot_missing = False,
                 extra_figure_args = None):
        
        self.cluster_data = cluster_data
        self.x, self.y = xy_cols
        self.factor_col = factor_col
        
        if tip_cols is None:
            self.tip_cols = ['Cluster']
        else:
            self.tip_cols = tip_cols
    
        self.pallette = pallette
        self.plot_missing = plot_missing
        self.extra_figure_args = {} if extra_figure_args is None else extra_figure_args

    
    def generate_bokeh(self):
                
        if self.plot_missing:
            mask = self.data[self.factor_col].notnull()
            view = CDSView(source=self.bokeh_source, filters=[BooleanFilter(mask.tolist())])
        else:
            view = CDSView(source=self.bokeh_source)

        fig = figure(tooltips = self.tooltips, **self.extra_figure_args)
        
        color = 'black' if self.factor_cmap is None else self.factor_cmap
        
        fig.scatter(x = self.x, y = self.y,
                    source = self.bokeh_source, view = view,
                    #legend_group = self.factor_col,
                    legend_field = self.factor_col,
                    color = color)
        
        fig.legend.click_policy="hide"
        
        self.fig = fig
        
    def _setup(self):
        
        self.bokeh_source = ColumnDataSource(self.cluster_data)
        self.factors, self.factor_cmap = _build_factor_cmap(self.cluster_data,
                                                            self.factor_col,
                                                            self.pallette)
        self.tooltips = _build_tips(self.tip_cols)
        


cl_exp = ClusteringBokehExplanation(full_df,
                                    factor_col= 'sample_tissue',
                                    tip_cols= ['accession', 'cluster', 
                                               'coreceptor', 'sample_tissue'])

fig = cl_exp.plot()
show(fig)


from sklearn.metrics import silhouette_samples

from bokeh.models import FactorRange

class SilhoutteBokehExplanation(BokehFigureExplanation):
    
    def __init__(self, cluster_data, feature_cols, label_cols):
        
        self.cluster_data = cluster_data
        self.feature_cols = feature_cols
        self.label_cols = label_cols

    def _setup(self):
        
        self._build_tips()
        
        self.calc_silhouttes()
        
        self.x_range = FactorRange(*self.silhoutte_means.index)
        source = {'feature_col': list(self.silhoutte_means.index),
                  'silhoutte_mean': self.silhoutte_means.tolist(),
                  'color': viridis(len(self.silhoutte_means))}
        self.bokeh_source = ColumnDataSource(data=source)
        
        
    def calc_silhouttes(self):
        
        out = {}
        for col in self.label_cols:
            mask = self.cluster_data[col].notnull()
            sample_scores = silhouette_samples(self.cluster_data.loc[mask, self.feature_cols], 
                                               self.cluster_data.loc[mask, col])
            out[col] = pd.Series(sample_scores, index = self.cluster_data.index[mask])
        
        odf = pd.DataFrame(out)
        self.silhoutte_scores = odf.reindex(self.cluster_data.index, axis=0)
        self.silhoutte_means = self.silhoutte_scores.mean()
        self.silhoutte_means.name = 'silhoutte_mean'
        self.silhoutte_means.index.name = 'feature_col'
        
        return self.silhoutte_scores
    
        
    def generate_bokeh(self):
                
        fig = figure(x_range = self.x_range, 
                     plot_height=250, tooltips = self.tooltips)
        fig.vbar(x = 'feature_col', top = 'silhoutte_mean', 
                 source = self.bokeh_source, color = 'color',
                 width=0.3)
        fig.y_range.start = -1
        fig.y_range.end = 1
        fig.xgrid.grid_line_color = None
        fig.yaxis.axis_label = 'Silhoutte Score'
        
        self.fig = fig
    
    
    def _build_tips(self):
        
        self.tooltips = [('Feature', '@feature_col'),
                         ('Score', '@silhoutte_mean')]
    
    


full_df['seq_len'] = full_df['sequence'].map(len)
feats = [col for col in full_df.columns if col.startswith('d')]
sil_exp = SilhoutteBokehExplanation(full_df, feats, 
                                    ['coreceptor', 'sample_tissue', 'seq_len'])

fig = sil_exp.plot()

show(fig)





# export

class AutoPipeline(object):
    
    
    
    def save(self):
        pass
    
    def load(self):
        pass
    
    def fit(self, data):
        """Fits this data to the model"""
        pass
    
    def fit_transform(self, data):
        """Sometimes its best to do both at once."""
        pass
    
    def transform(self, data):
        """Returns AutoPipelineResult"""
        pass
    
    
    
class AutoResult(object):
    
    
    def explain(self, explanations = 'all'):
        
        pass
    
    def save(self):
        pass
    
    def load(self):
        pass
    
    











#export

class GLPClusteringPipeline(AutoPipeline):
    
    full_embedding = None
    cluster_embedding = None
    vis_embessing = None
    
    
    def __init__(self, tmi = None, 
                 tokenizer = None, model = None, 
                 model_name = None, bs=8,
                 cluster_dim = 10, viz_dim = 2, 
                 device = 'cuda',
                 min_cluster_size = 5,
                 defaults = None):
        
        if tmi is not None:
            self.tmi = tmi
            
        else:
            self.tmi = TopicModelingInterface(tokenizer = tokenizer, model = model, model_name = model_name, bs=bs,
                                              cluster_dim = cluster_dim, viz_dim = viz_dim, device = device,
                                              min_cluster_size = min_cluster_size)
            
        self.model = self.tmi.model
            
        if defaults is None:
            self.defaults = {}
        else:
            self.defaults = defaults
            
            
    def fit(self, glp_data):
        self.fit_transform(glp_data, fit = True)
        
    
    def fit_transform(self, glp_data, 
                      sequence_col = None,
                      feature_cols = None,
                      fit = True):
        
        if sequence_col is None:
            try:
                sequence_col = self.defaults['sequence_col']
            except KeyError:
                sequence_col = 'sequence'
                
        if feature_cols is None:
            try:
                feature_cols = self.defaults['feature_cols']
            except KeyError:
                feature_cols = []
            
        
        seqs = glp_data[sequence_col].dropna()
        cluster_data, raw_embedding = self.tmi.process_df(glp_data, col = sequence_col, fit = fit)
        feature_data = glp_data.loc[seqs.index, feature_cols]
        
        return GLPClusteringResult(feature_data, raw_embedding, cluster_data)
        
    
    def transform(self, glp_data):
        return self.fit_transform(glp_data, fit=False)
    
    def save(self, path):
        pass
    
    @staticmethod
    def load(path):
        pass
             
    
class GLPClusteringResult(AutoResult):
    
    
    raw_embedding = None
    cluster_data = None
    feature_data = None
    explanations = ['cluster_figure', 'silhoutte_figure']
    
    
    def __init__(self, feature_data, raw_embedding, cluster_data):
        self.feature_data = feature_data
        self.raw_embedding = raw_embedding
        self.cluster_data = cluster_data
    
    
    def explain(self, factor_col = 'cluster', tip_cols = None):
        
        if tip_cols is None:
            tip_cols = list(self.feature_data.columns)
        
        
        # Pull out the cluster-feature data for later
        clust_feats = [col for col in self.cluster_data.columns if col.startswith('d')]
        
        full_df = pd.concat([self.cluster_data, self.feature_data], axis=1)
        
        sil_exp = SilhoutteBokehExplanation(full_df, clust_feats, 
                                            self.feature_data.columns)
        
        
        cl_exp = ClusteringBokehExplanation(full_df,
                                            factor_col = factor_col,
                                            tip_cols = tip_cols)
        
        return {'cluster_figure': cl_exp, 'silhoutte_figure': sil_exp}
        
    


df = pd.read_csv('../tutorials/HIV_tat_example.csv')
df['sequence'] = df['sequence'].str.strip('*')
df['seq_length'] = df['sequence'].map(len)
df.head()


ANALYSIS_DEFAULTS = {'feature_cols': ['coreceptor', 'sample_tissue', 'seq_length']}
clustering_pipeline = GLPClusteringPipeline(model_name = 'Rostlab/prot_bert',
                                            defaults = ANALYSIS_DEFAULTS)

train_data = df.sample(500)

clustering_pipeline.fit(train_data)


result = clustering_pipeline.transform(df)
result


len(result.cluster_data)


explanation = result.explain(factor_col = 'coreceptor')
explanation


fig = explanation['cluster_figure'].generate_figure(skip_missing = True)
fig


show(fig)



