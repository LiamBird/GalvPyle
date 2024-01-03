import re
import numpy as np
import pandas as pd

def mpt_to_df(filename, eis=False):
    with open(filename) as f:
        content = f.readlines()
  
    ## Changed 02/12/2022 for compatibility with Biologic outputs without long header
    if content[0].split("\t")[0] == "mode":
        n_header_lines = 1
    else:
    ## Changed 20/12/2023 to warn about empty data files without terminating notebook cell
        try:
            n_header_lines = int(re.findall("\d+", content[1])[0])
        except:
            print ("Failed to load, please check data file")
            return
    column_headers = content[n_header_lines-1].strip("\n").split("\t")[:-1]
    
    ## Changed 20/12/2023 for datafile with incomplete last line (unknown reason?)
    data_list = []
    for nline, line in enumerate(content[n_header_lines:]):
        line_entries = line.strip("\n").split("\t")
        if len(line_entries) == len(column_headers):
            data_list.append(line_entries)
        else:
            print("Omitting line {} of {}".format(nline, len(content[n_header_lines:])))
    data = np.array(data_list, dtype=float)
    
    if eis == False:
        return pd.DataFrame(data, columns=column_headers)
    else:
        frequencies = np.unique(data[:, 0])
        cycle_id = np.array([item for sublist in [[n]*frequencies.shape[0] for n in range(int(data.shape[0]/frequencies.shape[0]))] 
                             for item in sublist]).reshape(data.shape[0], 1)

        EIS_data_labelled = np.hstack((cycle_id, data))

        column_headers.insert(0, "Cycle id")

        return pd.DataFrame(EIS_data_labelled, columns=column_headers)