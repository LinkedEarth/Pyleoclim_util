import pyleoclim as pyleo
import pandas as pd
data=pd.read_csv('https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',skiprows=0,header=1)
time=data.iloc[:,1]
value=data.iloc[:,2]
ts=pyleo.Series(time=time,value=value,time_name='Year C.E', value_name='SOI', label='SOI')
fig,ax = ts.plot()
pyleo.showfig(fig)
