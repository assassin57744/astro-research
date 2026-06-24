
from .plotbase import PlotBase
from . import config
class CMDPlot(PlotBase):
    def plot(self):
        fig,ax=self.figax()
        mem,fld=self._split()
        ax.scatter(fld.bp_rp,fld.phot_g_mean_mag,s=config.FIELD_SIZE,c=config.FIELD_COLOR,alpha=config.ALPHA_FIELD,label="Field")
        ax.scatter(mem.bp_rp,mem.phot_g_mean_mag,s=config.MEMBER_SIZE,c=config.MEMBER_COLOR,alpha=config.ALPHA_MEMBER,label="Member")
        ax.invert_yaxis()
        ax.set_xlabel("BP-RP")
        ax.set_ylabel("G")
        ax.legend()
        return fig,ax
