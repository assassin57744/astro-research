
import pandas as pd
from gaia_plot import CMDPlot,PMPlot,SkyPlot

df=pd.read_csv("cluster.csv")
CMDPlot(df).plot()[0].savefig("cmd.pdf")
PMPlot(df).plot()[0].savefig("pm.pdf")
SkyPlot(df).plot()[0].savefig("sky.pdf")
print("Done")
