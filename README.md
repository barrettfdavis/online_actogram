# online_actogram
Actogram from browsers history, to screen sleep-wake patterns and sleep disorders.

This repository and tool will be renamed.

Forked from Barrett F. Davis excellent [online_actogram](https://github.com/barrettfdavis/online_actogram) script.

## Description
Script to generate an 'online actogram' from google chrome or safari history files

Currently set to import Safari's History.db file AND chrome's History file where applicable on MacOS and Windows. History files are copied from their home directories to a temporary location in the working directory. The temporary copies are then deleted after the script has executed. 

Now supports command line arguments, e.g.: 

```python actogram.py --freq '30T' --blur True --blursize 9 --start '2020-01-01' --end '2020-08-04'```

--freq determines the binning frequency of online/offline hours, default is 30 minutes ('30T')

--blur sets whether or not a median_filter is applied over data, default is True

--blur_size determines how large median_filter kernel is, default is 9

--start_date sets the oldest time to plot from, default is 1 year prior to today's date

--end_date sets the newest time to plot until, default is today's date

Plot can most simply be generated from the command line as:

```python actogram.py```

## Authors

This tool originated from an idea and script by [Barrett F. Davis](https://github.com/barrettfdavis/online_actogram) in August 2020.

## License

MIT Public License.
