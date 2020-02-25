''' For examples in documentation
'''
import pandas as pd

def load_dataset(name, **kws):
    url_dict = {
        'soi': 'https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/soi_data.csv',
    }

    df = pd.read_csv(url_dict[name], **kws)

    return df
