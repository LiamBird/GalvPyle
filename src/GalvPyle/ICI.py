import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import re
from tqdm import notebook
import time
import warnings

def _dataread(filename, fix_cycle_numbers=False):
    data_raw = []
    
    ## For fixing cycle numbers
    limits = {}
    target = "none_selected_"

    with open(filename) as f:
        print("Reading .mpt file")
        for line in notebook.tqdm(f.readlines()):
            if "Nb header lines" in line:
                headers = int(re.findall(r"\d+", line)[0])

            if "rec" in line and "type" in line and "Time" in line:
                target = line.split("\t")[0].split("_")[0]

            if target in line:
                parts = re.split(r"\s{2,}", line.strip())
                limits.update([(parts[0], parts[1:])])

            data_raw.append(line.strip("\n").split("\t"))

    columns = data_raw[headers-1]
    try:
        data_read = np.array([line for line in data_raw[headers:] if "mode" not in line], dtype=float) ## changed 13/04/2026
    except:
        data_read = np.array([line for line in data_raw[headers:] if "mode" not in line and len(line)>1], dtype=float) ## Fix if possible
    
    
    df = pd.DataFrame(data_read, columns=columns[:data_read.shape[1]])
    # self._df = df ## keeping raw df available

    if fix_cycle_numbers==True:
        limits = pd.DataFrame(limits)
        limits.columns = ["_".join((column.split("_")[1:])) for column in limits.columns]
        limits["value"] = limits["value"].astype(float)

        time = df["time/s"].to_numpy()
        cycle_breaks = np.nonzero(time[1:]-time[:-1]>2*max(limits["value"]))[0]
        cycle_breaks = np.insert(cycle_breaks, 0, 0)
        cycle_breaks = np.append(cycle_breaks, len(df))

        for nidx, idx in enumerate(cycle_breaks[:-1]):
            df.loc[cycle_breaks[nidx]: cycle_breaks[nidx+1], "cycle number"] = nidx
            
            
        ## Added 29/04/2026
        df_adj = df.copy() ##self._df.copy()

        for cycn in np.unique(df["cycle number"]):
            cycle_df = df.loc[df["cycle number"]==cycn]

            ## Adjusting Q discharge/mA.h
            original_Qdischarge = cycle_df["Q discharge/mA.h"].to_numpy()
            min_Qdischarge = np.nanmin(original_Qdischarge)
            adjusted_Qdischarge = original_Qdischarge-min_Qdischarge
            df_adj.loc[cycle_df.index, "Q discharge/mA.h"] = adjusted_Qdischarge


            ## Adjusting Q charge/mA.h
            original_Qcharge = cycle_df["Q charge/mA.h"].to_numpy()
            min_Qcharge = np.nanmin(original_Qcharge)
            adjusted_Qcharge = cycle_df["Q charge/mA.h"]-min_Qcharge
            df_adj.loc[cycle_df.index, "Q charge/mA.h"] = adjusted_Qcharge

            ## Adjusting Q charge/discharge/mA.h
            original_Qchargedischarge = cycle_df["Q charge/discharge/mA.h"].to_numpy()
            min_Qchargedischarge = np.nanmin(original_Qchargedischarge)
            adjusted_Qchargedischarge = original_Qchargedischarge-min_Qchargedischarge
            df_adj.loc[cycle_df.index, "Q charge/discharge/mA.h"] = adjusted_Qchargedischarge 

            ## Adjusting Capacity/mA.h
            original_capacity = cycle_df["Capacity/mA.h"].to_numpy()
            min_capacity = np.nanmin(original_capacity)
            adjusted_capacity = original_capacity-min_capacity
            df_adj.loc[cycle_df.index, "Capacity/mA.h"] = adjusted_capacity
            
        df = df_adj

    return df
    
def _annotate_raw(self):
    raw = self._df[["time/s", "cycle number", "I/mA", "Ecell/V", "Q charge/mA.h", "Q discharge/mA.h"]]
    raw.columns = ["t", "cycn", "I", "E", "cQ", "dQ"]

    raw = raw.copy()
    raw.loc[:, "cQ"] = raw["cQ"]
    raw.loc[:, "dQ"] = raw["dQ"]
    raw.loc[:, "I"] = raw["I"]*1e-3

    # 1) compute state column: "R" if I==0, "D" if I<0, "C" if I>0
    def compute_state(i_val):
        if pd.isna(i_val):
            return np.nan
        if i_val == 0:
            return "R"
        if i_val < 0:
            return "D"
        if i_val > 0:
            return "C"
    
    raw.loc[:, "state"] = raw["I"].apply(compute_state)
    
    # 2) compute rests: increment counter each time we observe a transition into "R"
    rests = []
    count_rest = 0
    states = raw["state"].tolist()

    for idx, st in enumerate(states):
        if idx == 0:
            # first row: count_rest stays 0
            rests.append(count_rest)
            continue

        prev_st = states[idx - 1]
        # transition into R from non-R
        if st == "R" and prev_st != "R":
            count_rest += 1
        rests.append(count_rest)

    raw = raw.copy()
    raw.loc[:, "rests"] = rests

    # 3) adjQ = dQ - cQ
    raw.loc[:, "adjQ"] = raw["dQ"] - raw["cQ"]

    return raw
    
    
def f_R(self, rest_index):
    """
    rest_index here corresponds to the integer i used in the R code.
    We'll implement:
      numerator = last E where (rests == rest_index) & (state == "R") 
                  minus last E where rests == rest_index-1
      denominator = -1 * last I where rests == rest_index-1
    Return numerator / denominator or np.nan if not computable.
    """
    prev = rest_index - 1

    # rows for current rest_index with state == "R"
    sel_current_R = self.raw[(self.raw["rests"] == rest_index) & (self.raw["state"] == "R")]
    # rows for previous rest index (no state restriction)
    sel_prev = self.raw[self.raw["rests"] == prev]

    try:
        E_current = sel_current_R["E"].iloc[-1]  # tail(...,1)
    except (IndexError, KeyError):
        E_current = np.nan

    try:
        E_prev = sel_prev["E"].iloc[-1]
    except (IndexError, KeyError):
        E_prev = np.nan

    try:
        I_prev = sel_prev["I"].iloc[-1]
    except (IndexError, KeyError):
        I_prev = np.nan

    # denominator = -1 * I_prev
    denom = -1.0 * I_prev if pd.notna(I_prev) else np.nan

    # if denom is zero or any required value is nan, return NaN
    if denom == 0 or not np.isfinite(denom) or not np.isfinite(E_current) or not np.isfinite(E_prev):
        return np.nan

    return (E_current - E_prev) / denom

def _proc_data(self):
    max_rest = int(self.raw["rests"].max()) if not self.raw["rests"].isnull().all() else 0
    rest_list = list(range(1, max_rest + 1))
    proc_columns = ["rest", "state", "cycn", "Q", "E", "R"]
    proc_rows = dict([(keys, []) for keys in proc_columns])
    
    print("ICI calculation")
    for r in notebook.tqdm(rest_list):
        prev = r - 1
        sel_prev = self.raw[self.raw["rests"] == prev]

        # get tail values from previous rest (if available), otherwise NaN
        if not sel_prev.empty:
            last_state = sel_prev["state"].iloc[-1]
            last_cycn = sel_prev["cycn"].iloc[-1] if "cycn" in sel_prev else np.nan
            last_adjQ = sel_prev["adjQ"].iloc[-1] if "adjQ" in sel_prev else np.nan
            last_E = sel_prev["E"].iloc[-1] if "E" in sel_prev else np.nan
        else:
            last_state = np.nan
            last_cycn = np.nan
            last_adjQ = np.nan
            last_E = np.nan

        proc_rows["rest"].append(r)
        proc_rows["state"].append(last_state)
        proc_rows["cycn"].append(last_cycn)
        proc_rows["Q"].append(last_adjQ)
        proc_rows["E"].append(last_E)
        proc_rows["R"].append(f_R(self, r))
        
    proc = pd.DataFrame(proc_rows)

    # 6) replace non-finite proc.R with NaN (matches R's is.finite/NA behavior)
    proc.loc[:, "R"] = proc["R"].apply(lambda v: v if np.isfinite(v) else np.nan)

    return proc
    
class ICI(object):
    """
        Version = 10/06/2026
    """
    def __init__(self, filename, verbose=False, fix_cycle_numbers=False, test_var=None, load_raw_df=False):
        
        self._version = "2026.06.10"
        self._change_log = {"2026.03.14": "Added line in data read to ensure repeated headers are not included",
                            "2026.04.23": "Split out functions for ease of diagnostics",
                            "2026.05.17": "Added save_path",
                            "2026.06.03": "Restructured for reloading legacy files and dealing with 'CROP_' filenames",
                            "2026.06.10": "Added load_raw_df for use with concatenating files, changed previous 'strip .mpt' to [-4:] to make usable for both .mpt and .txt files"}

        
        self.filename_norm = os.path.normpath(filename)
        self.filepath = os.path.split(self.filename_norm)[0]
        self.file_ext = self.filename_norm.split(".")[-1]
        self.file_label = os.path.split(self.filename_norm)[-1].strip("CROP_")[:-len(self.file_ext)-1]##.strip(".mpt")        
        self.cropped_filename = os.path.join(self.filepath, "CROP_"+self.file_label+"."+self.file_ext)
        self.uncropped_filename = os.path.join(self.filepath, self.file_label+"."+self.file_ext)

        processed_fname = self.file_label+"_processed.csv"
        charge_discharge_fname = self.file_label+"_raw_chargedischarge.csv"
        
        ## 1. Has the file previously been cropped and is this data available?            
        if os.path.isfile(self.cropped_filename):
            self.filename = self.cropped_filename
            if verbose == True:
                print("Exists as cropped: switching to cropped_filename")
        elif os.path.isfile(self.uncropped_filename):
            self.filename = self.uncropped_filename
            print("Exists as uncropped")
        else:
            print(f"File not found: {self.file_label}")
            print(f"self.uncropped_filename = {self.uncropped_filename}")
            print(f"self.cropped_filename = {self.cropped_filename}")

        ## 2. Has the data previously been processed to make a charge/ discharge file and a summary file?
        self.filename_dir = os.path.split(self.filename_norm)[0]
        self.processed_dir = os.path.join(self.filename_dir, "processed")

        if os.path.isdir(self.processed_dir):
            if verbose==True:
                print(f"Processed dir exists: {self.processed_dir}")
            if os.path.isfile(os.path.join(self.processed_dir, "CROP_"+processed_fname)):
                if os.path.isfile(os.path.join(self.processed_dir, processed_fname)):
                    os.remove(os.path.join(self.processed_dir, processed_fname))
                os.rename(os.path.join(self.processed_dir, "CROP_"+processed_fname),
                          os.path.join(self.processed_dir, processed_fname))

            incorrect_mpt_names = [file for file in os.listdir(self.processed_dir) if ".mpt_raw_chargedischarge" in file]
            for file in incorrect_mpt_names:
                os.rename(os.path.join(self.processed_dir, file),
                          os.path.join(self.processed_dir, file.replace(".mpt_raw_chargedischarge", "_raw_chargedischarge")))
            
            if os.path.isfile(os.path.join(self.processed_dir, "CROP_"+charge_discharge_fname)):
                if os.path.isfile(os.path.join(self.processed_dir, charge_discharge_fname)):
                    os.remove(os.path.join(self.processed_dir, charge_discharge_fname))
                os.rename(os.path.join(self.processed_dir, "CROP_"+charge_discharge_fname),
                          os.path.join(self.processed_dir, charge_discharge_fname))

            
            ## Case 1: the files exist and have the expected names
            if os.path.isfile(os.path.join(self.processed_dir, processed_fname)):
                raw_data_update_time = os.path.getmtime(self.filename)
                processed_update_time = os.path.getmtime(os.path.join(self.processed_dir, processed_fname))

                if processed_update_time > raw_data_update_time:
                    self.proc = pd.read_csv(os.path.join(self.processed_dir, processed_fname), index_col=0)
                    self.raw = pd.read_csv(os.path.join(self.processed_dir, charge_discharge_fname), index_col=0)

                if load_raw_df == True:
                    self._df = _dataread(filename=self.filename, 
                                    fix_cycle_numbers=fix_cycle_numbers) ## returns self._df

            ## Case 2: The files do not exist
            elif not os.path.isfile(os.path.join(self.processed_dir, processed_fname)):
                if verbose == True:
                    print("Files do not yet exist")
                self._df = _dataread(filename=self.filename, 
                      fix_cycle_numbers=fix_cycle_numbers)
                self.raw = _annotate_raw(self) ## returns self.raw
                self.proc = _proc_data(self) ## returns self.proc

                self.raw.to_csv(os.path.join(self.processed_dir, charge_discharge_fname))
                self.proc.to_csv(os.path.join(self.processed_dir, processed_fname))

        ## 3. If the data has not previously been processed, do so:
        elif not os.path.isdir(self.processed_dir):
            if verbose == True:
                print("Processed data directory does not yet exist")
            # print(self.filename)
            # print(os.path.isfile(self.filename))
            self._df = _dataread(filename=self.filename, 
                      fix_cycle_numbers=fix_cycle_numbers)
            self.raw = _annotate_raw(self)
            self.proc = _proc_data(self)

            ## Added 04/06/2026
            os.makedirs(self.processed_dir)
            self.raw.to_csv(os.path.join(self.processed_dir, charge_discharge_fname))
            self.proc.to_csv(os.path.join(self.processed_dir, processed_fname))            

            
                        