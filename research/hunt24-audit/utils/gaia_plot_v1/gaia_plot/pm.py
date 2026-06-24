
from .plotbase import PlotBase
from . import config
class PMPlot(PlotBase):
    def plot(self):
        fig,ax=self.figax()
        mem,fld=self._split()
        ax.scatter(fld.pmra,fld.pmdec,s=config.FIELD_SIZE,c=config.FIELD_COLOR,alpha=config.ALPHA_FIELD)
        ax.scatter(mem.pmra,mem.pmdec,s=config.MEMBER_SIZE,c=config.MEMBER_COLOR)
        ax.set_xlabel("pmRA"); ax.set_ylabel("pmDec")
        return fig,ax
