def _find_cycle_plateaus(cell_data, cycle):
    import numpy as np
    
    x = cell_data.discharge.capacity[cycle]
    y = cell_data.discharge.voltage[cycle]

    straight_grad = (y[-1]-y[0])/(x[-1]-x[0])
    straight_intc = y[0]-(x[0]*straight_grad)

    straight_y = straight_grad*x+straight_intc

    yD = straight_y-y

    if int(y.shape[0]/100) >= 1:
        step = int(y.shape[0]/100)
        window = 10
    else:
        step = 1
        window = 2

    dydx = (yD[step:]-yD[:-step])/(x[step:]-x[:-step])
    dx = x[int(step/2): int(step/2)+dydx.shape[0]]

    ## General: window -> -window
    ## Seeking positions where NEG before and POS after
    pos_after = np.argwhere([np.count_nonzero(np.sign(dydx[i: i+window])==1)>0.5*window for i in range(0, dydx.shape[0]-window)]).flatten()-int(window/2)
    neg_before = np.argwhere([np.count_nonzero(np.sign(dydx[i-window: i])==-1)>0.5*window for i in range(window, dydx.shape[0])]).flatten()+int(window/2)

    neg_after = np.argwhere([np.count_nonzero(np.sign(dydx[i: i+window])==-1)>0.5*window for i in range(0, dydx.shape[0]-window)]).flatten()-int(window/2)
    pos_before = np.argwhere([np.count_nonzero(np.sign(dydx[i-window: i])==1)>0.5*window for i in range(window, dydx.shape[0])]).flatten()+int(window/2)

    plateau_end = np.array([xi for xi in pos_after if xi in neg_before])
    plateau_start = np.array([xi for xi in pos_before if xi in neg_after])
    
    break_points = [0]+[nxi for nxi, x in enumerate(plateau_start[1:]) if plateau_start[nxi]-plateau_start[nxi-1]!=1 ]+[len(plateau_start)]
    break_points_starts = [int(np.nanmedian(item)) for item in [plateau_start[break_points[n]:break_points[n+1]]  for n in range(len(break_points)-1)] if len(item)>0]

    break_points = [0]+[nxi for nxi, x in enumerate(plateau_end[1:]) if plateau_end[nxi]-plateau_end[nxi-1]!=1]+[len(plateau_end)]
    break_points_ends = [int(np.nanmedian(item)) for item in [plateau_end[break_points[n]:break_points[n+1]] for n in range(len(break_points)-1)] if len(item)>0]

    break_points_starts, break_points_ends

    start_end_dict = {"EH_start": None,
                      "EH_end": None,
                      "EL_start": None,
                      "EL_end": None}

    ## General case: 2 starts + 2 ends
    if len(break_points_starts)==2 and len(break_points_ends)==2:
        if np.min(break_points_starts) < np.min(break_points_ends):
            start_end_dict["EH_start"] = np.min(break_points_starts)
            start_end_dict["EH_end"] = np.min(break_points_ends)

        if np.max(break_points_starts) < np.max(break_points_ends):
            start_end_dict["EL_start"] = np.max(break_points_starts)
            start_end_dict["EL_end"] = np.max(break_points_ends)

    ## No EL_end point
    if len(break_points_starts) == 2 and len(break_points_ends)==1:
        if np.min(break_points_starts) < break_points_ends[0]:
            start_end_dict["EH_start"] = np.min(break_points_starts)
            start_end_dict["EH_end"] = break_points_ends[0]

    ## No EH start point detected
    if len(break_points_starts) == 1 and len(break_points_ends)==2:
        if np.min(break_points_ends) < break_points_starts[0]:
            start_end_dict["EH_start"] = 0
            start_end_dict["EH_end"] = np.min(break_points_ends)
            start_end_dict["EL_start"] = break_points_starts[0]
            start_end_dict["EL_end"] = np.max(break_points_ends)
            
    return start_end_dict


class Plateaus(object):
    
    
    
    def __init__(self, cell_data):

        self._version = "15.10.2024"
        self._change_log = ["15.10.2024: Overhauled from old version to accommodate different capacity axes for variable rate data"]

        import warnings
        warnings.filterwarnings(action="ignore", category=RuntimeWarning)

        import numpy as np
        import matplotlib.pyplot as plt
        
        keys = set(["EH_capacity_mAh", "EL_capacity_mAh", "EH_capacity_pc", "EL_capacity_pc",
                    "EH_voltage", 
                    "EL_voltage"])
        self.n_cycles = cell_data.n_cycles

        self.__dict__.update([(key, np.full((self.n_cycles), np.nan)) for key in keys])
        setattr(self, "EH_voltage_minmax", np.full((2, self.n_cycles), np.nan))
        setattr(self, "EL_voltage_minmax", np.full((2, self.n_cycles), np.nan))
        
        for cycle in range(self.n_cycles):
            start_end_dict = _find_cycle_plateaus(cell_data, cycle)
            self.start_end_dict = start_end_dict
            
            if type(start_end_dict["EH_start"]) != type(None) and type(start_end_dict["EH_end"]) != type(None):
                self.EH_capacity_mAh[cycle] = cell_data.discharge.capacity[cycle][start_end_dict["EH_end"]]-cell_data.discharge.capacity[cycle][start_end_dict["EH_start"]]
                self.EH_capacity_pc[cycle] = self.EH_capacity_mAh[cycle]/cell_data.discharge.summary_capacity[cycle]
                self.EH_voltage[cycle] = np.nanmedian(cell_data.discharge.voltage[cycle][start_end_dict["EH_start"]:start_end_dict["EH_end"]])
                self.EH_voltage_minmax[0, cycle] = self.EH_voltage[cycle]-np.nanmin(cell_data.discharge.voltage[cycle][start_end_dict["EH_start"]:start_end_dict["EH_end"]])
                self.EH_voltage_minmax[1, cycle] = self.EH_voltage[cycle]-np.nanmin(cell_data.discharge.voltage[cycle][start_end_dict["EH_start"]:start_end_dict["EH_end"]])
                                   
                
            if type(start_end_dict["EL_start"]) != type(None) and type(start_end_dict["EL_end"]) != type(None):
                self.EL_capacity_mAh[cycle] = cell_data.discharge.capacity[cycle][start_end_dict["EL_end"]]-cell_data.discharge.capacity[cycle][start_end_dict["EL_start"]]
                self.EL_capacity_pc[cycle] = self.EL_capacity_mAh[cycle]/cell_data.discharge.summary_capacity[cycle]
                self.EL_voltage[cycle] = np.nanmedian(cell_data.discharge.voltage[cycle][start_end_dict["EL_start"]: start_end_dict["EL_end"]])
                self.EL_voltage_minmax[0, cycle] = self.EL_voltage[cycle]-np.nanmin(cell_data.discharge.voltage[cycle][start_end_dict["EL_start"]: start_end_dict["EL_end"]])
                self.EL_voltage_minmax[1, cycle] = self.EL_voltage[cycle]-np.nanmin(cell_data.discharge.voltage[cycle][start_end_dict["EL_start"]: start_end_dict["EL_end"]])
                
           
    def plot_capacity_share(self):
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, ax = plt.subplots()
        ax.plot(np.arange(self.n_cycles), self.EH_capacity_pc, "o", color="black", label="E$_{H}$ 2.4V")
        ax.plot(np.arange(self.n_cycles), self.EL_capacity_pc, "s", mfc="white", mec="black", label="E$_{L}$ 2.1V")
        ax.set_ylim([0, 1])
        ax.axhline(0.25, color="k", ls="-")
        ax.axhline(0.75, color="k", ls=":")
        ax.set_ylabel("Capacity share (%)")
        ax.set_xlabel("Cycle number")
        ax.legend()
        return fig, ax
    
    def plot_plateau_voltages(self):
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, ax = plt.subplots()
        ax.errorbar(x=np.arange(self.n_cycles), y=self.EH_voltage, yerr=self.EH_voltage_minmax, 
                    ls="None", color="k", lw=2, label="E$_{H}$ 2.4V")
        ax.errorbar(x=np.arange(self.n_cycles), y=self.EL_voltage, yerr=self.EL_voltage_minmax,
                    ls="None", color="gray", label="E$_{L}$ 2.1V")
        ax.legend()

        ax.set_ylim([1.8, 2.6])
        ax.axhline(2.4, color="k", lw=0.5)
        ax.axhline(2.1, color="gray", lw=0.5)
        ax.set_xlabel("Cycle number")
        ax.set_ylabel("Voltage range at plateau vs. Li/ Li$^{+}$ (V)")
        
        return fig, ax
