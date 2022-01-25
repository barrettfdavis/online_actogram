##############################################################################
#
# IMPORT LIBRARIES
#
##############################################################################

import os
import sys
import sqlite3
from shutil import copy, rmtree

import numpy as np
import pandas as pd
from scipy import ndimage

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec


class actography:
    def __init__(self):

        self.freq ='30T'

        self.hourly_blur = False  # median filter size
        self.daily_blur = False  # median filter size
        self.normalize = True
        self.weekend_color_support = False

        self.end = pd.Timestamp.today() - pd.DateOffset(days=1)
        self.start = pd.Timestamp.today() - pd.DateOffset(days=365)

        self.safari_file = 0
        self.chrome_file = 0
        self.df = pd.DataFrame()

        self.HH, self.DD, self.ZZ = None, None, None

        self.createFolders()
        self.generate()

    def createFolders(self):
        os.makedirs('actograms/', exist_ok=True)

    def generate(self):

        self._history_to_temp_folder()
        self._history_to_working_memory()
        self._process_df()
        self._preplot_process()

    def _copy_address(self, fname, src, dst_folder='temp_history'):

        os.makedirs(dst_folder, exist_ok=True)
        dst = os.path.join(dst_folder, fname)

        try:
            copy(src, dst)
            return dst
        except IOError as e:
            print("Unable to copy file. %s" % e)
            return 0
        except FileNotFoundError:
            print('The file \'' + fname + '\' could not be found.')
            return 0
        except Exception:
            print('Something went wrong, the file \'' +
                  fname + '\' was not loaded.')
            return 0
        return 0

    def _history_to_temp_folder(self):

        home = os.path.expanduser("~")  # set the path to home directory

        # Define the platform specific path to history files
        if sys.platform == "darwin":  # Darwin == OSX

            safari_src = os.path.join(home, 'Library/Safari/History.db')
            chrome_src = os.path.join(home, 'Library/Application Support/Google/Chrome/Default/History')

        elif sys.platform == "win32":
            safari_src = None  # will not attempt importing safari in windows
            chrome_src = home + '/AppData/Local/Google/Chrome/User Data/Default/History'

        else:
            print('Sorry, I''m having trouble with your operating system.')
            sys.exit()

        self.safari_file = self._copy_address('History.db', safari_src)
        self.chrome_file = self._copy_address('History', chrome_src)

    def _import_history(self, file_name, command_str):

        cnx = sqlite3.connect(file_name)

        df = pd.read_sql_query(command_str, cnx)
        cnx.commit()
        cnx.close()

        df.rename(inplace=True, columns={df.columns[0]: 'visit_time'})
        df = pd.to_datetime(df['visit_time'], errors='coerce').dropna()

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

        if not(any([self.chrome_file, self.safari_file])):
            print('\nError: No database(s) imported, in working directory?')
            sys.exit()

        if os.path.isdir('temp_history'):
            rmtree('temp_history')

    """ spam redirects can blow up time-windowed search history
    this gets rid of that issue by capping to some max value in time window"""

    def _normalize(self, z, intv):

        z = z.T

        for row, zhour in enumerate(z):
            z[row][z[row] >= intv] = 1
            z[row][z[row] <= 0] = 0

        return z.T

    """if you googled a bunch of stuff, ran to make lunch and then came back to
    google a bunch more right away, this filters out the gap in the middle.
    median filter means longer periods of inactivity aren't falsely counted"""

    def _hourly_blur(self, z):

        z = z.T
        for row, zhour in enumerate(z):
            z[row] = ndimage.median_filter(
                zhour, size=self.hourly_blur, mode='wrap')
        z = z.round()

        return z.T

    """this denoises small trends in sleep pattern across some number of days
    using this filter highlites changes in sleep/wake times at the expense
    of day-to-day actographic resolution"""

    def _daily_blur(self, z):

        for row, zday in enumerate(z):

            z[row] = ndimage.median_filter(
                zday, size=self.daily_blur, mode='reflect')

        return z

    def _process_df(self):

        df = self.df.copy()
        # rename first column
        df.rename(inplace=True, columns={0: "visit_time"})

        date_rng = pd.date_range(
                   df.visit_time.min().replace(hour=0, minute=0, second=0),
                   df.visit_time.max().replace(hour=0, minute=0, second=0),
                   freq=self.freq)  # get rid of extraneous hour/second info

        df.visit_time = pd.to_datetime(df.visit_time)
        df.set_index('visit_time', inplace=True)

        # count number of searches w/in specified date range w/ freq granularity
        df['sum'] = 1
        queries = df.resample(self.freq).agg({'sum': 'sum'})
        df = pd.DataFrame({'z': queries['sum']}, index=pd.to_datetime(date_rng))

        df['x'], df['y'] = df.index.date, df.index.hour

        # if start time was set too far in past, reset to match available data
        earliest_available_start = pd.Timestamp(df[df['z'] > 0].min()['x'])
        if earliest_available_start > self.start:
            self.start = earliest_available_start

        df = df[df.index >= self.start]
        df = df[df.index <= self.end]

        self.df = df

    def _preplot_process(self):
        # Define the granularity of the x axis (e.g. 15 minute, 1h increments)
        df = self.df # cutting down on self calls...
    
        if self.freq[-1] == 'T':
            self.freq_no = int(24*60/float(self.freq[:-1]))
            self.freq_intv = float(self.freq[:-1])/60

        elif self.freq[-1] == 'H':
            self.freq_no = int(24*float(self.freq[:-1]))
            self.freq_intv = float(self.freq[:-1])

        # Setup the data for pcolor
        xx = pd.date_range(df.index.min(),
                           df.index.max()).to_julian_date().tolist()
        yy = np.arange(df.y.min(), df.y.max()+1, self.freq_intv)

        app = len(xx)*len(yy) - len(df)
        df = df.append(pd.DataFrame(
            [[np.nan] * len(self.df.columns)]*app, columns=df.columns))
        zz = df.z.values.reshape(len(yy), len(xx), order='F')

        if self.hourly_blur:
            zz = self._hourly_blur(zz)
        if self.daily_blur:
            zz = self._daily_blur(zz)
        if self.normalize:
            zz = self._normalize(zz, intv=1)

        self.DD = np.array(pd.to_datetime(xx, unit='D', origin='julian').to_list())
        self.HH = np.arange(df.y.min(), 2 * (df.y.max()+1), self.freq_intv)

        self.ZZ = np.tile(zz, (2, 1)).T

        if self.weekend_color_support:
            week = np.array([i.weekday() < 5 for i in act.DD], dtype=int) + 1
            self.ZZ *= week[:, None]

        self.df = df  # pass the dataframe back out

    def plot_the_actogram(self, ax, printer_friendly=False, landscape=False):

        cmap = 'binary' if printer_friendly else 'binary_r'

        if landscape:

            ax.pcolormesh(self.DD, self.HH, self.ZZ.T, shading='auto',
                          cmap=cmap, vmin=0, rasterized=True)

            ax.set_yticks(np.arange(0, 48+6, 6)[::-1])
            ax.set_yticklabels([0, 6, 12, 18, ] * 2 + [24, ])
            ax.set_xlim(ax.get_xlim())  # [::-1])
            ax.set_ylim(0, 47)

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d-%Y'))
            ax.set_xticks(ax.get_xticks()[1:])
            plt.xticks(rotation=30, ha='right')

            ax.tick_params(axis='y', which='major', pad=15)

        else:

            ax.pcolormesh(self.HH, self.DD, self.ZZ, shading='auto',
                          cmap=cmap, vmin=0, rasterized=True)

            ax.set_xlim(0, 47)
            ax.set_xticks(np.arange(0, 48+6, 6))
            ax.set_xticklabels([0, 6, 12, 18, ] * 2 + [24, ])

            ax.set_ylim(ax.get_ylim()[::-1])

            ax.yaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
            ax.yaxis.set_label_position("right")
            ax.set_yticks(ax.get_yticks()[1:])
            ax.yaxis.tick_right()
            plt.yticks(va='center')

            ax.tick_params(axis='x', which='major', pad=10)

        return ax

    def plot_the_cdf(self, ax, ref_ax):

        ax.fill_between(act.HH,
                        np.nansum(act.ZZ, axis=0) /
                        np.nansum(act.ZZ, axis=0).max(),
                        color='grey', alpha=0.2)

        ax.xaxis.tick_top()

        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        ax.set_ylim([0, 1])
        ax.set_xlim(act.HH.min(), act.HH.max())

        ax.set_xticks(ref_ax.get_xticks())
        ax.set_xticklabels([])

        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Activity CDF', 1])

        ax.yaxis.tick_right()
        
        """
        ax.yaxis.set_label_position("right")
        
        ax.set_ylabel('Activity CDF',
                      x=0, y=1,
                      rotation=0,
                      ha='left', va='top')
                      #transform=ax.transAxes)"""
        ax.invert_yaxis()

        return ax

    def plot_the_timeshare(self, ax, ref_ax):

        offline = 24 - np.nansum(self.ZZ * self.freq_intv/2, axis=1)
        offline_avg = pd.Series(offline).rolling(window=7, min_periods=0).mean()

        ax.fill_betweenx(self.DD, offline_avg, color='grey', alpha=0.2)
        ax.axes.axvline(8, color='k', linestyle='--',lw=0.75)

        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.set_yticks([])
        ax.set_xticks([0, 8, 24])
        ax.set_xlabel('Offline (h)')

        ax.set_xlim(0, 24)
        ax.set_ylim(bottom=act.end - pd.DateOffset(days=7), top=act.start)
        ax.tick_params(axis='x', which='major', pad=10)
        ax.invert_xaxis()

        return ax

    def saveit(self, fig, orientation):

        plt.savefig('actograms/actogram' + orientation +
                    str(pd.to_datetime('today'))[0:10] + '.png', dpi=600)

    def plotter(self, printer_friendly=False):

        fig, ax = plt.subplots(figsize=DIMS)
        plt.subplots_adjust(left=0.1, right=0.75,
                            bottom=0.05, top=0.85,
                            wspace=0.1, hspace=0.2)

        plt.text(x=0, y=1.1,
                 s='Double-Plotted Online Actogram',
                 ha='left', va='bottom', fontweight='bold', wrap=True)

        plt.text(x=0, y=1.08,
                 s="Approximate sleep-wake periods, generated"
                 " from time stamped internet browser searches."
                 " Increments of {} minutes."
                 " Last updated {:%b-%d-%Y}.".format(
                    int(60/(self.freq_no/(24))),
                    pd.Timestamp.today()),
                 ha='left', va='top', wrap=True)

        ax.axis('off')

        spec = gridspec.GridSpec(ncols=2, nrows=2,
                                 height_ratios=[0.9, 0.1],
                                 width_ratios=[0.2, 0.8])

        ax_timeshare = fig.add_subplot(spec[0, 0])
        ax_actogram = fig.add_subplot(spec[0, 1])
        ax_dummy = fig.add_subplot(spec[1, 0])
        ax_cdf = fig.add_subplot(spec[1, 1])

        self.plot_the_actogram(ax_actogram, printer_friendly)
        self.plot_the_timeshare(ax_timeshare, ax_actogram)
        self.plot_the_cdf(ax_cdf, ax_actogram)

        ax_dummy.axis('off')


plt.style.use('default')
for tick in ['xtick.minor.visible', 'ytick.minor.visible']:
    plt.rcParams[tick] = False

DIMS = (6,8)
act = actography()
figgy = act.plotter(printer_friendly=False)
