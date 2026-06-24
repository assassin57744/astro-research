
from .plotbase import PlotBase
from . import config
class SkyPlot(PlotBase):
    def plot(self):
        fig,ax=self.figax()
        mem,fld=self._split()
        ax.scatter(fld.ra,fld.dec,s=config.FIELD_SIZE,c=config.FIELD_COLOR,alpha=config.ALPHA_FIELD)
        ax.scatter(mem.ra,mem.dec,s=config.MEMBER_SIZE,c=config.MEMBER_COLOR)
        ax.invert_xaxis()
        ax.set_aspect("equal","box")
        ax.set_xlabel("RA"); ax.set_ylabel("Dec")
        return fig,ax
