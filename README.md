# online_actogram
Actogram from browsers history, to screen sleep-wake patterns and sleep disorders.

This repository and tool will be renamed.

Forked from Barrett F. Davis excellent [online_actogram](https://github.com/barrettfdavis/online_actogram) script.

## Description
Script to generate an 'online actogram' from google chrome or safari history files

Currently set to import Safari's History.db file AND chrome's History file where applicable on MacOS and Windows. History files are copied from their home directories to a temporary location in the working directory. The temporary copies are then deleted after the script has executed. 

Plot can most simply be generated from the command line with:

```python actogram.py```


Plots will be saved in a new sub-folder called "actograms" with appropriate timestamp and description. 


Script now supports command line arguments (optional) for additional customizability, e.g.: 

```python actogram.py --freq '15T' --daily_blur 3 --start '2020-01-01' ```

```python actogram.py --freq '30T' --printer_friendly True```

```python actogram.py --dims (8,8)```

Where: 

--freq determines the granularity of binned online/offline periods (default is 15 minutes increments, ex.  --freq '15T')

--start_date sets initial date to plot from, default is 180 days ago (ex. --start_date '2022-01-01')

—-daily_blur applies median filtering between days (off by default, ex. --daily_blur 3)  

--period_blur applies median filtering between binned time periods (off by default, ex. --period_blur 5)

--normalize normalizes search frequency against max, then applies binary mask (plot shows periods of some search history vs. none, on by default)

--dims sets the relative dimensions of generated actogram plot (ex. --dims (4, 6))

--printer_friendly sets whether activity is shown in black on white (friendly) or vice versa (False by default, ex. --printer_friendly True)


_______________

Fixes: 

[Bug fix] Previously there was an artificially low minimum window for all generated plots. Plots can now be shown with minutes resolution 

[Feature] Added "activity CDF" subplot to gauge periods of minimum and maximum activity 

[Feature] Added cumulative "offline hours" subplot to estimate sleep per 24h period (NB: this yields artificially high results with high freq values)








## Authors

This tool originated from an idea and script by [Barrett F. Davis](https://github.com/barrettfdavis/online_actogram) in August 2020.

## License

MIT Public License.
