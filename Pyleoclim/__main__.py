# -*- coding: utf-8 -*-
"""
Created on Fri May 13 11:36:41 2016

@author: deborahkhider
"""

import os
import sys

def main():
    """
    Wrapper for Jupyter. Keep directory for lipd notebooks under root folders
    """
    
    dir_nb = os.path.join(os.path.expanduser('~'),'Pyleoclim_Notebooks')
    if not os.path.isdir(dir_nb):
        os.mkdir(dir_nb)

    print("Retrieving notebooks..")
    os.chdir(dir_nb)

    try:
        print("Starting Jupyter...")
        os.system('jupyter notebook')
    except:
        print("Problem launching jupyter!")
        sys.exit(1)

    return


if __name__ == '__main__':
    main()
        
        