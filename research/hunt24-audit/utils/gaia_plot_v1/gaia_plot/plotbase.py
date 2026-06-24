
import matplotlib.pyplot as plt
from . import config

class PlotBase:
    def __init__(self,df):
        self.df=df
    def figax(self):
        fig,ax=plt.subplots(figsize=config.FIGSIZE,dpi=config.DPI)
        if config.GRID:
            ax.grid(alpha=.3)
        return fig,ax
    def _split(self):
        m=(1==1)
        return self.df,self.df
