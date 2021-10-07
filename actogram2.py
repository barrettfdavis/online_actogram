##############################################################################
#
# IMPORT LIBRARIES
#
##############################################################################

import os
import sys
import sqlite3
from   shutil import copy, rmtree

import numpy as np
import pandas as pd

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from scipy import ndimage
from scipy.interpolate import griddata

##############################################################################
#
# FUNCTIONS
#
##############################################################################

""" this doesn't need to be a class,it's really not even best practice,
 but this is a hobby project and I wanted to practice class syntax. It''s spaghetti code
 not high-caliber, performance-driven stuff"""

class actography:
    
    def __init__(self):
        
        self.freq = '60T'
        
        self.hourly_blur = False# median filter size
        self.daily_blur = 5 # median filter size
        
        self.end = pd.Timestamp.today()
        self.start = pd.Timestamp.today() - pd.DateOffset(days=90)
        
        self.safari_file = 0
        self.chrome_file = 0
        self.df = pd.DataFrame()
        
        self.HH, self.DD, self.ZZ = None, None, None
        
        self.generate()
        
    def generate(self):
    
        self._history_to_temp_folder()
        self._history_to_working_memory()
        self._process_df()
        self._preplot_process()

    def _copy_address(self, fname, src, dst_folder='temp_history'): 
    
        os.makedirs(dst_folder, exist_ok=True)
        dst = os.path.join(dst_folder, fname)
        
        try:
            copy(src, dst); return dst
        except IOError as e:
            print("Unable to copy file. %s" % e); return 0
        except FileNotFoundError: 
            print('The file \'' + fname + '\' could not be found.'); return 0
        except: 
            print('Something went wrong, the file \'' + fname + '\' was not loaded.')
            return 0
        return 0

    def _history_to_temp_folder(self):
            
            home = os.path.expanduser("~") # set the path to home directory
            
            # Define the platform specific path to history files
            if sys.platform == "darwin": # Darwin == OSX
            
                safari_src = os.path.join(home, 'Library/Safari/History.db')
                chrome_src = os.path.join(home, 'Library/Application Support/Google/Chrome/Default/History')
            
            elif sys.platform == "win32": 
                safari_src = None # will not attempt importing safari in windows
                chrome_src = home + '/AppData/Local/Google/Chrome/User Data/Default/History'
            
            else: 
                print('Sorry, I''m having trouble with your operating system.')
                sys.exit()
            
            self.safari_file = self._copy_address('History.db', safari_src)
            self.chrome_file = self._copy_address('History', chrome_src)
            
    
    def _import_history(self, file_name, command_str):
        
        cnx = sqlite3.connect(file_name)
        
        df = pd.read_sql_query(command_str,cnx); cnx.commit(); cnx.close()
        
        df.rename(inplace=True, columns={ df.columns[0]: 'visit_time'})
        df = pd.to_datetime(df['visit_time'],errors='coerce').dropna()
    
        return df    
        
            
    def _history_to_working_memory(self):
        
        if self.safari_file: 
            command_str = 'SELECT datetime(visit_time+978307200, "unixepoch",\
                          "localtime") FROM history_visits ORDER BY visit_time DESC;'
            
            df_safari = self._import_history(self.safari_file, command_str)
            self.df = pd.concat([self.df, df_safari])
            
        if self.chrome_file: 
            command_str = "SELECT datetime(last_visit_time/1000000-11644473600,\
            'unixepoch','localtime'), url FROM urls ORDER BY last_visit_time DESC;"
            
            df_chrome = self._import_history(self.chrome_file, command_str)
            self.df = pd.concat([self.df, df_chrome])
            
        if not(any([self.chrome_file,self.safari_file])):
            print('\nError: No database(s) imported! Everything in working directory?')
            sys.exit()
            
        if os.path.isdir('temp_history'): rmtree('temp_history')
    
    
    """ spam redirects can blow up time-windowed search history
    this gets rid of that issue by capping to some max value in time window"""
    def _normalize(self, z, intv):
    
        z = z.T
        
        for row, zhour in enumerate(z):
            z[row][z[row] > intv ] = 1
            z[row][z[row] <= 0] = 0
    
        return z.T

    """if you googled a bunch of stuff, ran to make lunch and then came back to 
    google a bunch more right away, this filters out the gap in the middle.
    median filter means that longer periods of inactivity aren't falsely counted"""
    def _hourly_blur(self, z):
        
        z = z.T
        
        for row, zhour in enumerate(z): 
    
            z[row] = ndimage.median_filter(zhour, size=self.hourly_blur, mode='wrap')
        
        z = z.round()
        
        return z.T
    
    """this denoises small trends in sleep pattern across some number of days
    using this filter highlites changes in sleep/wake times at the expense
    of day-to-day actographic resolution"""
    def _daily_blur(self, z):
        
        for row, zday in enumerate(z):
            
            z[row] = ndimage.median_filter(zday, size=self.daily_blur, mode='reflect')
    
        return z
    
        
    def _process_df(self):
        
        self.df.rename(inplace=True, columns={0: "visit_time"}) # rename first column

        date_rng  = pd.date_range(self.df.visit_time.min().replace(hour=0, minute=0, second=0),
                                  self.df.visit_time.max().replace(hour=0, minute=0, second=0),
                                  freq=self.freq) # get rid of extraneous hour/second info
        
        self.df.visit_time = pd.to_datetime(self.df.visit_time)
        self.df.set_index('visit_time', inplace=True)
        
        # count the number of searches within the specified date range w/ freq granularity
        self.df['sum'] = 1; searches = self.df.resample(self.freq).agg({'sum':'sum'})
        self.df = pd.DataFrame({'z':searches['sum']},index=pd.to_datetime(date_rng))
        
        self.df['x'], self.df['y'] = self.df.index.date, self.df.index.hour
        self.df = self.df[self.df.index >= self.start] # set oldest time to plot from
        self.df = self.df[self.df.index <= self.end] # set most recent time to plot until
        
    def _preplot_process(self):
        # Define the granularity of the y axis (e.g. 15 minute, 2h, 3 day increments)
        
        if self.freq[-1] == 'T':
            self.freq_no = int(24*60/float(self.freq[:-1]))
            self.freq_intv = float(self.freq[:-1])/60
            
        elif self.freq[-1] == 'H':
            self.freq_no = int(24*float(self.freq[:-1]))    
            self.freq_intv = float(self.freq[:-1])
        
        
        # Setup the data for pcolor
        xx = pd.date_range(self.df.index.min(), self.df.index.max()).to_julian_date().tolist()
        
    
        yy = np.arange(self.df.y.min(), self.df.y.max()+1, self.freq_intv)
        zz = griddata((self.df.index.to_julian_date(),self.df.index.hour),
                      self.df.z,(xx[:],yy[:,None]),method='nearest')
        
        
        if self.hourly_blur: zz = self._hourly_blur(zz)
        if self.daily_blur: zz = self._daily_blur(zz)
        
        zz = self._normalize(zz, intv=1)
            
        self.DD = pd.to_datetime(xx,unit='D',origin='julian').tolist()
        self.HH = np.arange(self.df.y.min(), 2*(self.df.y.max()+1), self.freq_intv)
        self.ZZ = np.tile(zz,(2,1)).T
        
        

    def plot_the_actogram(self):
        
        fig, ax = plt.subplots(1, 1, figsize=(4, 6))
        plt.title('Double-Plotted Online Actogram',fontsize=12)
        fig.subplots_adjust(hspace=0, wspace=0, left=0.2, right=0.9)
         
        ax.pcolormesh(self.HH, self.DD, self.ZZ, cmap='Greys_r',shading='auto',snap=True)

        ax.set_xticks(np.arange(0,48,6))
        ax.set_xticklabels([0,6,12,18,]*2)
        ax.set_ylim(ax.get_ylim()[::-1])
        
        ax.yaxis.set_major_formatter(mdates.DateFormatter('%b-%d-%Y'))
        
        plt.savefig('actograms/actogram'+str(pd.to_datetime('today'))[0:10] + '.png',dpi=600)
        
        return fig



plt.close('all')

act = actography()
figgy = act.plot_the_actogram()
