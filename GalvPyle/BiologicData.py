class BiologicData(object):
    def __init__(self, filename):
        
        import re
        import pandas as pd
        import numpy as np
               
        data_read = []
        
        self.version = "2.1.0"
        self._change_log = ["2.0.1    251204    added self._current_column and self._voltage_column for outputs from EC lab vs BT lab, added np.nan for empty cycles",
                            "2.0.2     260413    added 'if 'mode' not in line' when making self._df to avoid files with repeated headers",
                            "2.1.0     260414    added removing skipped cycles in init. Previously, some files returned to cycle 0 midway, unclear why"]

        with open(filename) as f:
            for line in f.readlines():
                if "Nb header lines" in line:
                    header_lines = int(re.findall("\d+", line)[0])
                data_read.append(line.strip("\n").split("\t"))
                
        header_labels = data_read[header_lines-1]
        
        df = pd.DataFrame(np.array([line for line in data_read[header_lines:] if "mode" not in line], ## Added 13/04/2026 ## removed self._df 14/04/2026
                                         dtype=float), columns=header_labels[:-1])
        
        from copy import deepcopy
        self._df = df ## added 14/04/2026
        
        ## Moved without modification 14/04/2026 so that filtering by self._current_column works
        if "I/mA" in self._df.columns:
            self._current_column = "I/mA"
        elif "<I>/mA" in self._df.columns:
            self._current_column = "<I>/mA"
            
        if "Ecell/V" in self._df.columns:
            self._voltage_column = "Ecell/V"
        elif "Ewe/V" in self._df.columns:
            self._voltage_column = "Ewe/V"
        
        ## Removing skipped cycles: added 14/04/2026        
        self.df = df.loc[df[self._current_column]!=0].reset_index(drop=True)

        cycle_numbers = self.df["cycle number"].to_numpy()
        skip_points = cycle_numbers[1:]<cycle_numbers[:-1]

        if any(skip_points==True) == True:

            cycle_number_breaks = np.argwhere(skip_points)[0]+1 ## +1 because of offset

            cycle_number_breaks = [n for n in cycle_number_breaks if n>1]

            for break_point in cycle_number_breaks:
                previous_cycle_number = self.df["cycle number"].iloc[break_point-1]
                apparent_cycle_number = self.df["cycle number"].iloc[break_point]
                number_gap = previous_cycle_number-apparent_cycle_number
                self.df = self.df.copy()

                previous_time = self.df["time/s"].iloc[break_point-1]
                apparent_time = self.df["time/s"].iloc[break_point]
                time_gap = previous_time-apparent_time

                self.df.loc[break_point:, "cycle number"] = self.df.loc[break_point:, "cycle number"]+1+number_gap
                self.df.loc[break_point:, "time/s"] = self.df.loc[break_point:, "time/s"]+time_gap
        
        
        self.number_cycles = int(self.df["cycle number"].max())
        

        
        class _Cycle(object):
            def __init__(cycle_self):
                cycle_self_keys = ["capacity", "voltage", "summary_capacity"]
                cycle_self.__dict__.update([(keys, []) for keys in cycle_self_keys])
            
        cycle_types = {"charge": 1,
                       "discharge": 0}
        
        data_types = {"capacity": "Capacity/mA.h",
                      "voltage": self._voltage_column, 
                      }
        
        for cycle_type in cycle_types.keys():
            setattr(self, cycle_type, _Cycle()) 
        
        ## Changed output to numpy rather than pd.Series
        ## 14.04.2026: changed references from self._df to self.df to use processed table!!
        for ncyc in range(self.number_cycles):
            cycle_df = self.df.loc[(self.df["cycle number"]==ncyc) & (abs(self.df[self._current_column])>0)]
            
            for cycle_keys, cycle_values in cycle_types.items():
                cycle_capacity = cycle_df.loc[cycle_df["ox/red"]==cycle_values][data_types["capacity"]]
                vars(self)[cycle_keys].capacity.append((cycle_capacity-cycle_capacity.min()).to_numpy())
                
                vars(self)[cycle_keys].voltage.append(cycle_df.loc[cycle_df["ox/red"]==cycle_values][data_types["voltage"]].to_numpy())           
                if len(vars(self)[cycle_keys].capacity[-1]) > 0:
                    vars(self)[cycle_keys].summary_capacity.append(max(vars(self)[cycle_keys].capacity[-1]))
                else:
                    vars(self)[cycle_keys].summary_capacity.append(np.nan)
                
    def save_to_excel(self, save_file_name):
        
        import numpy as np
        import pandas as pd
        import os
        
        if save_file_name[-5:] != ".xlsx" :
            save_file_name += ".xlsx"
        
        max_cycle_length = max([max([len(data) for data in vars(self)[cycle_keys].capacity]) for cycle_keys in ["discharge", "charge"]])

        data_arrs = dict([(cycle_type, np.full((max_cycle_length, 2*self.number_cycles), np.nan))
                          for cycle_type in ["discharge", "charge"]])

        for cycle_type in ["discharge", "charge"]:
            for ncyc in range(self.number_cycles):
                data_arrs[cycle_type][:len(vars(self)[cycle_type].capacity[ncyc]), 2*ncyc] = vars(self)[cycle_type].capacity[ncyc]
                data_arrs[cycle_type][:len(vars(self)[cycle_type].voltage[ncyc]), 2*ncyc+1] = vars(self)[cycle_type].voltage[ncyc]
                
        column_headers = [item for sublist in [["capacity{}".format(n),
                                                  "voltage{}".format(n)] for n in range(self.number_cycles)] for item in sublist]
        
        discharge_df = pd.DataFrame(data_arrs["discharge"], columns=column_headers)
        charge_df = pd.DataFrame(data_arrs["charge"], columns=column_headers)
        
        summary_capacity_df = pd.DataFrame({"Cycle number": np.arange(self.number_cycles), 
              "Discharge capacity": self.discharge.summary_capacity,
              "Charge capacity": self.charge.summary_capacity})
        
        print("Writing discharge data")
        
        if os.path.isfile(save_file_name):
            with pd.ExcelWriter(save_file_name, engine='openpyxl', mode="a", if_sheet_exists="replace") as writer:
                discharge_df.to_excel(writer,
                                sheet_name="discharge",
                                index=False)
        else:
            with pd.ExcelWriter(save_file_name, engine='openpyxl') as writer:
                discharge_df.to_excel(writer,
                                sheet_name="discharge",
                                index=False)
            
        print("Writing charge data")
        with pd.ExcelWriter(save_file_name, engine='openpyxl', mode="a", if_sheet_exists="replace") as writer:
            charge_df.to_excel(writer,
                                    sheet_name="charge",
                                    index=False)
            
        print("Writing summary capacity data")
        with pd.ExcelWriter(save_file_name, engine='openpyxl', mode="a", if_sheet_exists="replace") as writer:
            summary_capacity_df.to_excel(writer,
                                 sheet_name="summary_capacity",
                                 index=False)
            
    def plot_summary_capacity(self):
        
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams.update({"xtick.direction": "in"})
        plt.rcParams.update({"ytick.direction": "in"})
        plt.rcParams.update({"xtick.top": True})
        plt.rcParams.update({"ytick.right": True})
        
        fig, ax = plt.subplots()
        ax.plot(self.discharge.summary_capacity, "o")
        ax.set_ylim([0, None])
        ax.set_xlabel("Cycle number")
        ax.set_ylabel("Capacity (mAh)")
        
        return fig, ax
    
    def plot_capacity_voltage(self, cycle_interval=None):
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams.update({"xtick.direction": "in"})
        plt.rcParams.update({"ytick.direction": "in"})
        plt.rcParams.update({"xtick.top": True})
        plt.rcParams.update({"ytick.right": True})
        
        
        if type(cycle_interval) == type(None):
            cycles_to_plot = np.arange(0, self.number_cycles)
        else:
            cycles_to_plot = np.arange(0, self.number_cycles, cycle_interval, dtype=int)
            
        fig, ax = plt.subplots()

        for cyc in cycles_to_plot:    
            d, = ax.plot(self.discharge.capacity[cyc],
                    self.discharge.voltage[cyc], label=cyc)
            c = ax.plot(self.charge.capacity[cyc],
                    self.charge.voltage[cyc], color=d.get_color())

        ax.legend(title="Cycle number", loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_ylabel("Voltage vs. Li/ Li$^{+}$ (V)")
        ax.set_xlabel("Capacity (mAh)")
        ax.set_xlim([0, None])
        plt.tight_layout()
        
        return fig, ax
            
    