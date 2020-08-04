##############################################################################
#
# IMPORT LIBRARIES
#
##############################################################################

from shutil import copy, rmtree
import pandas as pd
import numpy as np
import sqlite3
import os, sys

from scipy.interpolate import griddata
from scipy import ndimage

import matplotlib.pyplot as plt
import matplotlib as mpl

import argparse

##############################################################################
#
# TUNABLE PARAMETERS
#
##############################################################################
parser = argparse.ArgumentParser(description='Process user inputs')

today = pd.Timestamp.today()
minusyr = pd.Timestamp.today() - pd.DateOffset(days=365)

parser.add_argument('-freq', '--freq', action='store', default='30T')
parser.add_argument('-median', '--blur', default='True')
parser.add_argument('-median_size', '--blur_size', action='store', default=9)
parser.add_argument('-start', '--start_date', action='store', default= minusyr)
parser.add_argument('-end', '--end_date', action='store', default=today)

args = parser.parse_args()

freq = args.freq
blur = args.blur
median_size = args.blur_size
start_date = args.start_date
end_date = args.end_date

##############################################################################
#
# FUNCTIONS
#
##############################################################################

# used to make temporary folder with history files
def copy_address(fname, src, dst_folder='temp_history'): 
    
    os.makedirs(dst_folder, exist_ok=True)
    
    dst = os.path.join(dst_folder, fname)
    
    try:
        copy(src, dst)
        return dst
    
    except FileNotFoundError: 
        print('The file \'' + fname + '\' could not be found.')
        return 0
    
    except: 
        print('Something went wrong, the file \'' + fname + '\' not loaded.')
        return 0

# used to import the history file databases

def import_history(file_name, command_str):
    
    cnx = sqlite3.connect(file_name)
    
    df = pd.read_sql_query(command_str,cnx); cnx.commit(); cnx.close()
    
    df.rename(inplace=True, columns={ df.columns[0]: 'visit_time'})
    df = pd.to_datetime(df['visit_time'],errors='coerce').dropna()

    return df    

##############################################################################
#
# COPY HISTORY FILES TO TEMP STORAGE FOLDER
#
##############################################################################

mydir = os.path.join(os.getcwd()) # get the current working directory
home = os.path.expanduser("~") # set the path to home directory

# Define the platform specific path to history files
if sys.platform == "darwin": # Darwin == OSX
    safari_src = home + '/Library/Safari/History.db'
    chrome_src = home + '/Library/Application Support/Google/Chrome/Default/History'

elif sys.platform == "win32": 
    safari_src = None # will not attempt importing safari in windows
    chrome_src = home + '/AppData/Local/Google/Chrome/User Data/Default/History'

else: 
    print('Sorry, I''m having trouble with your operating system.')
    sys.exit()

safari_file = copy_address('History.db', safari_src)
chrome_file = copy_address('History', chrome_src)

##############################################################################
#
# IMPORT HISTORY FILES and DELETE TEMP STORAGE FOLDER
#
##############################################################################

df = pd.DataFrame()

if safari_file: 
    command_str = 'SELECT datetime(visit_time+978307200, "unixepoch",\
                  "localtime") FROM history_visits ORDER BY visit_time DESC;'
    
    df_safari = import_history(safari_file, command_str)
    df = pd.concat([df, df_safari])
    
if chrome_file: 
    command_str = "SELECT datetime(last_visit_time/1000000-11644473600,\
    'unixepoch','localtime'), url FROM urls ORDER BY last_visit_time DESC;"
    
    df_chrome = import_history(chrome_file, command_str)
    df = pd.concat([df, df_chrome])
    
if not(any([chrome_file,safari_file])):
    print('\nError: No database(s) imported! Everything in working directory?')
    sys.exit()
    
if os.path.isdir('temp_history'): rmtree('temp_history')

##############################################################################
#
# PROCESS DATAFRAME
#
##############################################################################
df.rename(inplace=True, columns={0: "visit_time"}) # rename first column

date_rng  = pd.date_range(df.visit_time.min().replace(hour=0, minute=0, second=0),
                          df.visit_time.max().replace(hour=0, minute=0, second=0),
                          freq=freq) # get rid of extraneous hour/second info

df.visit_time = pd.to_datetime(df.visit_time)
df.set_index('visit_time', inplace=True)

# count the number of searches within the specified date range w/ freq granularity
df['sum'] = 1; searches = df.resample(freq).agg({'sum':'sum'})
df = pd.DataFrame({'z':searches['sum']},index=pd.to_datetime(date_rng))

df['x'], df['y'] = df.index.date, df.index.hour

df = df[df.index >= start_date] # set oldest time to plot from
df = df[df.index <= end_date] # set most recent time to plot until
##############################################################################
#
# SETUP PLOTTING DATA 
#
##############################################################################

# Define the granularity of the y axis (e.g. 15 minute, 2h, 3 day increments)
if freq[-1] == 'T':
    freq_no = int(24*60/float(freq[:-1]))
elif freq[-1] == 'H':
    freq_no = int(24*float(freq[:-1]))    
    
# Setup the data for pcolor
xi = pd.date_range(df.index.min(), df.index.max()).to_julian_date().tolist()
yi = np.linspace(df.y.min(), df.y.max(), freq_no)
zi = griddata((df.index.to_julian_date(),df.index.hour),
              df.z,(xi[:],yi[:,None]),method='nearest')

xid = pd.to_datetime(xi,unit='D',origin='julian').tolist()

# Apply a median_filter blur if specified
if blur: zi = ndimage.median_filter(zi,size=median_size)
zi[zi>0] = 1 # apply binary thresholding (i.e. awake or asleep)

##############################################################################
#
# PLOTTING INFORMATION
#
##############################################################################
""" Explanation: We're going to make two pcolor plots and smush them against
each other, that way we get the 'double-plotted' effect to better visualize
a late-night periods of activity """

plt.close('all')

colors, labels = ['midnightblue','cornsilk'], ['Asleep','Awake']
cmap = mpl.colors.ListedColormap(colors)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 8))
fig.suptitle('Double-Plotted Online Actogram',fontsize=14, y=0.92)
fig.subplots_adjust(hspace=0, wspace=0)

ax1.pcolor(yi,xid,zi.T,cmap=cmap,shading='auto')
container = ax2.pcolor(yi,xid,zi.T,cmap=cmap,shading='auto')

ax2.tick_params(axis='y', which='both', length=0)
plt.setp(ax2.get_yticklabels(), visible=False)

ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)

ax1.set(ylabel='Time of Year')
fig.text(0.5, 0.04, 'Hour of Day', ha='center', va='center')

cb = plt.colorbar(container, ax=ax2, fraction =0.064,pad=0.04)
cb.set_ticks([0.25,0.75]); cb.set_ticklabels(labels);
cb.ax.set_yticklabels(labels, rotation='vertical')
cb.ax.tick_params(size=0, labelsize=12)

ax1.set_xticks([0,6,12,18]); ax1.set_xticklabels([0,6,12,18])
ax2.set_xticks([0,6,12,18]); ax2.set_xticklabels([0,6,12,18])

ax1.set_ylim(ax1.get_ylim()[::-1])
ax2.set_ylim(ax2.get_ylim()[::-1])

plt.show()