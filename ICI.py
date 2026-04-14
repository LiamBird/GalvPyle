import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import re
from tqdm import notebook
import time

import warnings

## Converted from R code by Lacey 2015 using ChatGPT
## Original R code: M. J. Lacey et al., Chem. Comm., 51(92), 16502, (2015) 
## DOI: 10.1039/C5CC07167D

class ICI(object):
    def __init__(self, filename, reload=True, verbose=True, reload_check=False, save_csv=True, raw_only=False):
        
        self._version = "2026.03.14"
        self._change_log = {"2026.03.14": "Added line in data read to ensure repeated headers are not included"}
        
        warnings.filterwarnings(action="ignore", message="SettingWithCopyWarning")
        
        self.filename = os.path.split(filename)[-1].strip(".mpt") ## used for labelling later
        self.path = os.path.split(filename)[0]
        creation_time = os.path.getmtime(filename)
        time_str = "%d/%m/%Y %H:%M:%S"
        self.data_last_updated = time.strftime(time_str, time.localtime(creation_time))
        
        if os.path.isfile(f"{self.filename}_rawchargedischarge.csv"):
            if verbose==True:
                print("Previous data available")
            self.raw = pd.read_csv(os.path.join(f"{self.filename}_rawchargedischarge.csv"))
            if verbose == True:
                print("Reloaded data from charge/ discharge only, omitting rest steps")
            
        else:
            self.raw = self._dataload(filename)
        
        if reload == True and reload_check==False:
            if verbose==True:
                print("Reloading processed and raw data")
            if raw_only==False:
                if verbose == True:
                    print("reload == True and reload_check==False")
#                 print(os.path.join(self.path, self.filename+r"_rawchargedischarge.csv")
                self.proc = pd.read_csv(os.path.join(self.path, self.filename+".csv"), index_col=0)
            
        
        if reload == True and reload_check==True and raw_only==False:
            if verbose == True:
                print("Checking whether processed data up to date")
                print("To skip this step, use reload_check=False")
            if os.path.isfile(os.path.join(self.path, self.filename)+".csv"):
                proc_df = pd.read_csv(os.path.join(self.path, self.filename+".csv"), index_col=0)

                # Finding the last cycle reported in the existing data:
                f = open(filename)
                line = f.readline()

                header_idx = None
                while line and type(header_idx)==type(None):
                    if "Nb header lines" in line:
                        header_idx = int(re.findall(r"\d+", line)[0])
                    line = f.readline()
                f.close()
                f = open(filename)
                headers = f.readlines()[header_idx-1].strip("\n").split("\t")
                f.close()

                f = open(filename)
                last_cycle = float(f.readlines()[-1].strip("\n").split("\t")[headers.index("cycle number")])
                f.close()

                if last_cycle == proc_df["cycn"].max():
                    print("Running line 66")
                    self.proc = proc_df 
                    self.raw = self._dataload(filename)
                    if verbose==True:
                        
                        print("Reloaded previously processed data (up to date)")
                else:
                    print("Running line 69")
                    print("Calculating and saving new values")
                    self.proc = self._proc_data()
                    
        if reload == False:
            print("Calculating and saving new values")
            self.raw = self._dataload(filename)
            if raw_only==False:
                self.proc = self._proc_data()
            
        if save_csv == True and raw_only==False:
            self.proc.to_csv(os.path.join(self.path, self.filename)+".csv")
            
            raw_charge_discharge = self.raw.loc[self.raw["state"]!="R"]
            raw_charge_discharge.to_csv(os.path.join(os.path.join(self.path, self.filename+r"_rawchargedischarge.csv")))

            
    ## Loading raw data
    def _dataload(self, filename):
        data_raw = []
        print(filename)

        with open(filename) as f:
            for line in notebook.tqdm(f.readlines()):
                if "Nb header lines" in line:
                    headers = int(re.findall(r"\d+", line)[0])
                    
                data_raw.append(line.strip("\n").split("\t"))

        columns = data_raw[headers-1][:-1]
        data_read = np.array([line for line in data_raw[headers:] if "mode" not in line], dtype=float) ## changed 13/04/2026
        

        df = pd.DataFrame(data_read, columns=columns)

        ## Below adapted from Lacey 
        raw = df[["time/s", "cycle number", "I/mA", "Ecell/V", "Q charge/mA.h", "Q discharge/mA.h"]]
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
            return "D" if i_val < 0 else "C"

        raw = raw.copy()
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

    ## Processing raw data

    def _proc_data(self):
        # 4) define f.R equivalent in python
        def f_R(rest_index):
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

        # 5) build proc DataFrame
        max_rest = int(self.raw["rests"].max()) if not self.raw["rests"].isnull().all() else 0
        rest_list = list(range(1, max_rest + 1))

        proc_rows = {
            "rest": [],
            "state": [],
            "cycn": [],
            "Q": [],
            "E": [],
            "R": []
        }

        print("Calculating resistance")

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
            proc_rows["R"].append(f_R(r))

        proc = pd.DataFrame(proc_rows)

        # 6) replace non-finite proc.R with NaN (matches R's is.finite/NA behavior)
        proc.loc[:, "R"] = proc["R"].apply(lambda v: v if np.isfinite(v) else np.nan)

        return proc

        # Now 'proc' is equivalent to the R 'proc' data.frame
        # and 'raw' has the new columns 'state', 'rests', and 'adjQ'.

        # Optional: show quick summaries
#         print(proc.head())
#         print(raw[["t", "cycn", "I", "E", "cQ", "dQ", "adjQ", "state", "rests"]].head())
#             self.proc = proc
#             self.raw = raw

            