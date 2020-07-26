##############################################################################
#
# IMPORT LIBRARIES
#
##############################################################################

import sys
import pandas as pd
import numpy as np
import sqlite3

from scipy.interpolate import griddata
from scipy import ndimage

import matplotlib.pyplot as plt
import matplotlib

##############################################################################
#
# DEFINE FILE NAMES and TIMEZONE
#
##############################################################################

safari_file = 'History.db' # set to zero if unused # 0
chrome_file = 'History' # set to 0 if unused # 0

##############################################################################
#
# SET TUNABLE PARAMETERS
#
##############################################################################

# only change these if you want ----------------------------------------------
freq = '30T' # sample every 30 minutes, use '1H' for every hour
freq_no  = 48 # how many freq periods are in 24h? 24 for 1H, 48 for 30 min

median_filter_size = 9 # set to one to keep all data, increase to smooth
paint_over_camping_trips = 1 # set to 0 to keep all-nighters and days offline

##############################################################################
#
# Functions
#
##############################################################################

def import_history(file_name, command_str):
    
    cnx = sqlite3.connect(file_name)
    df = pd.read_sql_query(command_str,cnx); cnx.commit(); cnx.close()
    
    df.rename(inplace=True, columns={ df.columns[0]: 'visit_time'})
    df = pd.to_datetime(df['visit_time'],errors='coerce').dropna()

    return df    

##############################################################################
#
# IMPORT DATABASE FILES
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
    
df.rename(inplace=True, columns={0: "visit_time"})

if not(any([chrome_file,safari_file])):
    print('Error: No database(s) imported! Everything in working directory?')
    sys.exit()

##############################################################################
#
# PROCESS DATAFRAME
#
##############################################################################

date_rng  = pd.date_range(df.visit_time.min().replace(hour=0, minute=0, second=0),
                          df.visit_time.max().replace(hour=0, minute=0, second=0),
                          freq=freq)

df.visit_time = pd.to_datetime(df.visit_time)
df.set_index('visit_time', inplace=True)

df['sum'] = 1; searches = df.resample(freq).agg({'sum':'sum'})
df = pd.DataFrame({'z':searches['sum']},index=pd.to_datetime(date_rng))

df['x'], df['y'] = df.index.date, df.index.hour
df.head()

dist = df.max()-df.min()
dist = dist['x'].days

##############################################################################
#
# SETUP PLOTTING DATA 
#
##############################################################################

xi = pd.date_range(df.index.min(), df.index.max()).to_julian_date().tolist()
yi = np.linspace(df.y.min(), df.y.max(), freq_no)
zi = griddata((df.index.to_julian_date(),df.index.hour),
              df.z,(xi[:],yi[:,None]),method='nearest')

xid = pd.to_datetime(xi,unit='D',origin='julian').tolist()

# course correct for camping trips and one-off all-nighters ------------------
if paint_over_camping_trips: 
    for idx, val in enumerate(zi.T): 
        if len(set(val)) == 1: zi.T[idx,:] = zi.T[idx-1,:]

# median filter and set to binary---------------------------------------------
zi = ndimage.median_filter(zi,size=median_filter_size)
zi[zi>0] = 1

##############################################################################
#
# PLOTTING INFORMATION
#
##############################################################################
""" Explanation: We're going to make two pcolor plots and smush them against
each other, that way we get the 'double-plotted' effect to better visualize
a late-night periods of activity """

plt.close('all')

colors = ['midnightblue','cornsilk']
labels = ['Asleep','Awake']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 12))
fig.suptitle('Double-Plotted Online Actogram',fontsize=14, y=0.92)
fig.subplots_adjust(hspace=0, wspace=0)

ax1.pcolor(yi,xid,zi.T,cmap=matplotlib.colors.ListedColormap(colors),shading='auto')
container = ax2.pcolor(yi,xid,zi.T,cmap=matplotlib.colors.ListedColormap(colors),shading='auto')

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
