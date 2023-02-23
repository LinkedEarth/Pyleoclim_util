''' For examples in documentation
'''
import pandas as pd

def load_dataset(name, **kws):
    url_dict = {
        'soi': 'https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',
    }

    df = pd.read_csv(url_dict[name], **kws)

    return df



# resample example
import pyleoclim as pyleo
ts = pyleo.utils.load_dataset('LR04')
ts5k = ts.resample('5ka').mean()
fig, ax = ts.plot(invert_yaxis='True')
ts5k.plot(ax=ax,color='C1')
pyleo.closefig(fig)