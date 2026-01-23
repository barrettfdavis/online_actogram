    class ImportData:

        def __init__(self, act):
            super().__init__()
            self.act = act

            self.__main__()

        def __main__(self):

            self.lookup_history_filepaths()
            self.copy_history_to_temp_folder()
            self.import_history_to_working_memory()
            self.delete_temporary_history_folder()

        def lookup_history_filepaths(self):
                """ check which OS user is running script from, then 
                check typical file paths for popular browser history files """

                home = os.path.expanduser("~")

                if sys.platform == "darwin":  # Darwin == OSX
                    safari_src = os.path.join(home, 'Library/Safari/History.db')
                    chrome_src = os.path.join(home, 'Library/Application Support/Google/Chrome/Default/History')
                    firefox_src = None # TODO
                    edge_src = None # TODO

                elif sys.platform == "win32":
                    safari_src = None
                    chrome_src = home + '/AppData/Local/Google/Chrome/User Data/Default/History'
                    firefox_src = None # TODO
                    edge_src = None # TODO

                else:
                    print('Sorry, having trouble with your operating system.')
                    sys.exit()

                self.history_loc_dict = {'safari':   [safari_src, 'History.db'],
                                         'chrome':   [chrome_src, 'History'],
                                         'firefox':  [firefox_src, 'History'],
                                         'edge':     [edge_src, 'History']
                                         }

        def copy_history_to_temp_folder(self):
            """ Iterate through each file referenced in the history_loc_dict 
            and copy to some temporary folder. This avoids direclty operating 
            on the user's broswers' history files. """

            for key, value in self.history_loc_dict.items():
                src, fname = value

                if src is not None and os.path.exists(src):
                    print(key, value, fname, src)
                    self.copy_history_func(src, fname)


        def copy_history_func(self, src, fname, dst_folder='temp_history'):
            """ function to copy file at given file location to temporary folder"""

            os.makedirs(dst_folder, exist_ok=True)
            dst = os.path.join(dst_folder, fname)
    

            try:
                copy(src, dst)
                return dst

            except IOError as e:
                print("Unable to copy file, bro. %s" % e)

            except FileNotFoundError:
                print('The file \'' + fname + '\' could not be found.')

            except Exception:
                print('Something went wrong, the file \'' +
                      fname + '\' was not loaded.')

        def import_history_to_working_memory(self):
            """ Imports all of the files in the temporary folder into working
                memory. Each browser's particular history file format is 
                standardized before concatenating to an overarching df"""

            for key, value in self.history_loc_dict.items():
                src, fname = value
                
                #if src is not None:
                    
                if src is not None and os.path.exists(src):
                    if key == 'safari':
                        command_str = 'SELECT datetime(visit_time+978307200, "unixepoch",\
                                      "localtime") FROM history_visits ORDER BY visit_time DESC;'
                        command_str = 'SELECT * FROM history_visits'
                        
                    elif key == 'chrome':
                        command_str = "SELECT datetime(last_visit_time/1000000-11644473600,\
                        'unixepoch','localtime'), url FROM urls ORDER BY last_visit_time DESC;"

                    elif key == 'firefox':
                        pass

                    elif key == 'edge':
                        pass

                    df = self._import_history_func(src, command_str)
                    self.act.df = pd.concat([self.act.df, df])

        def delete_temporary_history_folder(self):
            """ Delete the temporary folder after files are copied into working 
            memory. No need to cache this temporary folder, unless looking to backup
            browser history data (in which case there are better alternatives) """

            if os.path.isdir('temp_history'):
                rmtree('temp_history')

        def _import_history_func(self, file_name, command_str):
            """ Function to open SQL styled history files and convert to a pandas
            DataFrame type. SQL objects are closed after copying to Pandas DF. """
            
            cnx = sqlite3.connect(f'file:{file_name}?immutable=1', uri=True)
            df = pd.read_sql_query(command_str, cnx)
            df.to_csv('safari_db.csv')

            cnx.commit()
            cnx.close()

            df.rename(inplace=True, columns={df.columns[0]: 'visit_time'})
            df = pd.to_datetime(df['visit_time'], errors='coerce').dropna()

            return df