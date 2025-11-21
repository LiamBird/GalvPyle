class BiologicNew(object):
    def __init__(self, filename):
        from mpt_to_df import mpt_to_df
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        
        self.df = mpt_to_df(filename)

        if "Ecell/V" in self.df.columns:
            volt_col = "Ecell/V"
        elif "Ewe/V" in self.df.columns:
            volt_col = "Ewe/V"
        
        self.n_cycles = int(np.max(self.df["cycle number"]))
        cycle_split = [self.df.loc[self.df["cycle number"]==n] for n in range(self.n_cycles)]

        class _CycleData(object):
            def __init__(cycle_self, n_cycles):
                cycle_self.capacity = np.full((self.n_cycles), np.nan, dtype=object)
                cycle_self.voltage = np.full((self.n_cycles), np.nan, dtype=object)
                cycle_self._time = np.full((self.n_cycles), np.nan, dtype=object)
                cycle_self._current = np.full((self.n_cycles), np.nan, dtype=object)

                cycle_self.summary_capacity = np.full((self.n_cycles), np.nan)
                
        self.discharge = _CycleData(self.n_cycles)
        self.charge = _CycleData(self.n_cycles)
        
        for cyc in range(self.n_cycles):
            cycle_discharge = cycle_split[cyc].loc[np.sign(cycle_split[cyc]["I/mA"])==-1]
            self.discharge.capacity[cyc] = cycle_discharge["Capacity/mA.h"].to_numpy()
            self.discharge.voltage[cyc] = cycle_discharge[volt_col].to_numpy()
            self.discharge._time[cyc] = cycle_discharge["time/s"].to_numpy()
            self.discharge._current[cyc] = cycle_discharge["I/mA"].to_numpy()

            self.discharge.summary_capacity[cyc] = np.nanmax(self.discharge.capacity[cyc])
            
            cycle_charge = cycle_split[cyc].loc[np.sign(cycle_split[cyc]["I/mA"])==1]
            self.charge.capacity[cyc] = cycle_charge["Capacity/mA.h"].to_numpy()
            self.charge.voltage[cyc] = cycle_charge[volt_col].to_numpy() ## fixed Monday!
            self.charge._time[cyc] = cycle_charge["time/s"].to_numpy()
            self.charge._current[cyc] = cycle_charge["I/mA"].to_numpy()
            try:
                self.charge.summary_capacity[cyc] = np.nanmax(self.charge.capacity[cyc])
            except:
                pass
            
    def plot_cycle_capacity(self):
        import matplotlib.pyplot as plt
        self.fig, self.ax = plt.subplots()
        self.ax.plot(self.discharge.summary_capacity, "o")
        
        return self.fig, self.ax
    
    def plot_capacity_voltage(self, cycles_to_plot = "all", 
                              palette="crest_r", current=True, 
                              electrolyte=167, separator="Celgard+GF+Celgard",
                              annotation_pos="default", rate_change=False):
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np
        
        if rate_change==True:
            fig, ax = plt.subplots()

            for cyc in [4, 14, 24, 34, 44]:
                if cyc in range(low_high_data.n_cycles):
                    current = abs(np.nanmedian(low_high_data.discharge._current[cyc]))
                    d, = ax.plot(low_high_data.discharge.capacity[cyc], low_high_data.discharge.voltage[cyc], label="{:.2f} mA".format(current))
                    ax.plot(low_high_data.charge.capacity[cyc], low_high_data.charge.voltage[cyc], color=d.get_color())

            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            ax.set_xlabel("Capacity (mAh)")
            ax.set_ylabel("Voltage vs. Li/ Li$^{+}$ (V)")
            plt.tight_layout()
            ax.set_xlim([0, None])
            
            return fig, ax
        
        else:
            fig, ax = plt.subplots()
            if cycles_to_plot == "all":
                cycles_to_plot = np.arange(self.n_cycles, dtype=int)

            cmap = sns.color_palette(palette=palette, as_cmap=True)
            colors = [cmap(i) for i in np.linspace(0, 1, len(cycles_to_plot))]
            for ncyc, cyc in enumerate(cycles_to_plot):
                d, = ax.plot(self.discharge.capacity[cyc], self.discharge.voltage[cyc], color=colors[ncyc], label=cyc+1)
                ax.plot(self.charge.capacity[cyc], self.charge.voltage[cyc], color=colors[ncyc])
            ax.set_xlabel("Capacity (mAh)")
            ax.set_ylabel("Voltage vs Li/ Li$^{+}$ (V)")
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

            annotation_string = ""

            if current == True:
                annotation_string += "Current: {:.2f} mA".format(abs(np.nanmedian(self.discharge._current[-1])))
            if electrolyte != False:
                annotation_string += "\nElectrolyte: {} $\mu$L".format(electrolyte)
            if separator != False:
                annotation_string += "\nSeparator: {}".format(separator)

            if annotation_pos == "default":
                annotation_pos = (0.1, 0.1)

            ax.annotate(annotation_string,
                        annotation_pos, xycoords=ax.transAxes, va="bottom")


            return fig, ax
        
    def rate_change_capacity_voltage(self, cycles_per_rate=10, area_type="coin_cell", show_units=False, show_last=False,
                                     figsize=(6, 6)):
        
        import matplotlib.pyplot as plt
        import numpy as np
        def _show_current(xy, rate_current):
            
            ax = plt.gca()

            if show_units == False:
                    annotate_str = "{:.2f}".format(rate_current)
            else:
                annotate_str = "{:.2f} mA/cm$^{}$".format(rate_current, 2)
            ax.annotate(annotate_str, 
                        (max(xy.get_xdata()), max(xy.get_ydata())), ha="center")


        rate_change_colours = {"initial_01": plt.cm.Paired(1),
                   "_02": plt.cm.Paired(3),
                   "_05": plt.cm.Paired(5),
                  "_1": plt.cm.Paired(7),
                  "final_01": plt.cm.Paired(0)}
        fig, ax = plt.subplots(figsize=figsize)
        if area_type == "coin_cell":
            coin_cell_diameter = 14 ## mm
            cell_area = (coin_cell_diameter/20)**2/np.pi

        elif area_type == "swagelok":
            swagelok_diamter = 3 ## mm
            cell_area = (swagelok_diamter/2)**np.pi

        ## Plotting first cycle
        ax.plot(self.discharge.capacity[0], self.discharge.voltage[0], color=rate_change_colours["initial_01"])
        c01_i, = ax.plot(self.charge.capacity[1], self.charge.voltage[1], color=rate_change_colours["initial_01"])

        ## Plotting subsequent cycles
        for nvalue, value in enumerate([*rate_change_colours.values()]):
            ax.plot(self.discharge.capacity[int(nvalue*cycles_per_rate+cycles_per_rate/2)], 
                    self.discharge.voltage[int(nvalue*cycles_per_rate+cycles_per_rate/2)], color=value)
            xy, = ax.plot(self.charge.capacity[int(nvalue*cycles_per_rate+cycles_per_rate/2)], 
                    self.charge.voltage[int(nvalue*cycles_per_rate+cycles_per_rate/2)], color=value)
            rate_current = abs(np.nanmedian(self.discharge._current[nvalue*cycles_per_rate+int(cycles_per_rate/2)]))/cell_area

            if show_last == False and nvalue != len(rate_change_colours)-1:  
                _show_current(xy, rate_current)

            elif show_last == True:
                _show_current(xy, rate_current)
        coin_cell_area = 1.4**2/4*np.pi

        ax2 = ax.twiny()
        ax2.set_xlim([min(ax.get_xlim())/coin_cell_area,
                      max(ax.get_xlim())/coin_cell_area])
        ax2.set_xlabel("Capacity (mAh cm$^{-2}$)")

        ax.set_ylabel("Voltage vs. Li/ Li$^{+}$ (V)")
        ax.set_xlabel("Capacity (mAh)")
        plt.tight_layout()     
        return fig, ax
    
    def rate_change_capacity_plot(self, area_type="coin_cell", figsize=(6, 6), cycles_per_rate=10):
        import matplotlib.pyplot as plt
        import numpy as np
        
        
        rate_change_colours = {"initial_01": plt.cm.Paired(1),
                           "_02": plt.cm.Paired(3),
                           "_05": plt.cm.Paired(5),
                          "_1": plt.cm.Paired(7),
                          "final_01": plt.cm.Paired(0)}

        all_cyc_idx = np.arange(self.n_cycles)
        n_rates = int(self.n_cycles/cycles_per_rate)
        idx_by_rate = [all_cyc_idx[n*cycles_per_rate: (n+1)*cycles_per_rate] for n in range(n_rates)]

        fig, ax = plt.subplots(figsize=figsize)

        if area_type == "coin_cell":
            coin_cell_diameter = 14 ## mm
            cell_area = (coin_cell_diameter/20)**2/np.pi

        elif area_type == "swagelok":
            swagelok_diamter = 3 ## mm
            cell_area = (swagelok_diamter/2)**np.pi
            
        for n in range(n_rates):
            data, = ax.plot(idx_by_rate[n],
                [self.discharge.summary_capacity[idx] for idx in idx_by_rate[n]], "o", 
                   color=[*rate_change_colours.values()][n])

            current = abs(np.nanmedian(self.discharge._current[idx_by_rate[n][0]]))/cell_area

            ax.annotate("{:.2f} mA/cm$^{}$".format(current, 2), (data.get_xdata()[0], data.get_ydata()[0]), va="bottom")
        ax.set_ylim([0, None])
        ax2 = ax.twinx()
        ax2.set_ylim([ax.get_ylim()[0]/cell_area, 
                      ax.get_ylim()[1]/cell_area])

        ax.set_xlabel("Cycle number")
        ax.set_ylabel("Capacity (mAh)")
        ax2.set_ylabel("Areal capacity (mAh cm$^{-2}$)")

        return fig, ax