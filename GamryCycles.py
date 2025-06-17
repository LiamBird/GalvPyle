class GamryCycles(object):
    def __init__(self, path):
        import numpy as np
        import re
        from GamryData import GamryData
        import os
        
        gamry_data_1 = dict([(keys, 
                            {}) 
                           for keys in np.unique(["_".join((name.split("_")[1:-1])) for name in os.listdir(path) if "CHARGE" in name])]
                         )

        for name in os.listdir(path):
            for keys in gamry_data_1.keys():
                if "CHARGE" in name and keys in name:
                    try:
                        gamry_data_1[keys].update([(name,
                                              GamryData(os.path.join(path, name), mass=None, CRate=None))])
                    except:
                        pass

        discharge_keys = dict([(keys, [name for name in values.keys() if "PWRDISCHARGE" in name]) for keys, values in gamry_data_1.items()])
        charge_keys = dict([(keys, [name for name in values.keys() if "PWRCHARGE" in name]) for keys, values in gamry_data_1.items()])


        discharge_order = dict([(keys, np.argsort([[int(re.findall("\d+", subseg)[0]) for subseg in seg if "#" in subseg][0] 
                 for seg in [item.split("_") for item in values]])) 
                for keys, values in discharge_keys.items()])

        charge_order = dict([(keys, np.argsort([[int(re.findall("\d+", subseg)[0]) for subseg in seg if "#" in subseg][0] 
                 for seg in [item.split("_") for item in values]])) 
                for keys, values in charge_keys.items()])

        discharge_data = dict([(label, 
                                dict([(discharge_keys[label][idx], 
                                       GamryData(os.path.join(path, discharge_keys[label][idx]), mass=None, CRate=None))
                                      for idx in values]))
                              for label, values in discharge_order.items()])

        charge_data = dict([(label, 
                                dict([(charge_keys[label][idx], 
                                       GamryData(os.path.join(path, charge_keys[label][idx]), mass=None, CRate=None))
                                      for idx in values]))
                              for label, values in charge_order.items()])
        
        self.discharge = discharge_data
        self.charge = charge_data
        
        self.current_labels = {}
        for keys, values in self.discharge.items():
            try:
                self.current_labels.update([(keys, 
                                         abs(np.nanmedian([*values.values()][0].df["Im"])))])
            except:
                self.current_labels.update([(keys, np.nan)])
        
#         self.current_labels = dict([(keys, abs(np.nanmedian([*values.values()][0].df["Im"]))) for keys, values in self.discharge.items()])
        
        self.labels = [*discharge_data.keys()]
        
    def plot_capacity_voltage(self, init_label="init_rate", main_label="main_rate",
                              annotation_loc=(0.05, 0.95), annotation_ha="left",
                              narrow_label=False, show_annotation=True):

        import matplotlib.pyplot as plt
        import numpy as np
        from seaborn import color_palette

        electrode_diameter = .3
        electrode_area = electrode_diameter**2/4*np.pi
        
        fig, ax = plt.subplots()
        
        if len(self.discharge) == 1:
            init_label = None
            main_label = [*self.discharge.keys()][0]
        
        if init_label != None:
            ax.set_prop_cycle("color", plt.cm.Greys_r(np.linspace(0, 0.5, len(self.discharge[init_label]))))
            for n, (keys, values) in enumerate(self.discharge[init_label].items()):
                d, = ax.plot(values.capacity,
                        values.voltage, label="Init. "+str(n+1))

                if "DISCHARGE" in keys and keys not in self.charge[init_label].keys():
                    charge_label = keys.replace("DISCHARGE", "CHARGE")
                else:
                    charge_label = keys

                try:
                    ax.plot(self.charge[init_label][charge_label].capacity, 
                        self.charge[init_label][charge_label].voltage, color=d.get_color())
                except:
                    print(charge_label)

        ax.set_prop_cycle("color", 
                          color_palette(palette="crest", n_colors=len(self.discharge[main_label])))
        for n, (keys, values) in enumerate(self.discharge[main_label].items()):
            d, = ax.plot(values.capacity,
                    values.voltage, label=str(n+1))
            if keys in self.charge[main_label].keys():
                ax.plot(self.charge[main_label][keys].capacity,
                        self.charge[main_label][keys].voltage, color=d.get_color())
                
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) > 20:
            ncol=2
        else:
            ncol=1
        
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=ncol)
        
        if init_label != None:
            if narrow_label == False:
                annotation = "Current 1: {:.2f} $\mu$A ({:.2f} mA/ cm$^{}$) \nCurrent 2: {:.2f} $\mu$A ({:.2f} mA/ cm$^{}$)".format(self.current_labels[init_label]*1e6, self.current_labels[init_label]/electrode_area*1e3, "2", self.current_labels[main_label]*1e6, self.current_labels[main_label]/electrode_area*1e3, "2")
                
            else:
                annotation = "i1: {:.2f} $\mu$A ({:.2f} mA/ cm$^{}$) \ni2: {:.2f} $\mu$A ({:.2f} mA/ cm$^{}$)".format(self.current_labels[init_label]*1e6,
                                                                                                                   self.current_labels[init_label]/electrode_area*1e3,
                                                                                                                   "2",
                                                                                                                     self.current_labels[main_label]*1e6, 
                                                                                                                  self.current_labels[main_label]/electrode_area*1e3,
                                                                                                                   "2")
            if show_annotation == True:
                ax.annotate(annotation, annotation_loc, ha=annotation_ha, va="top", xycoords=ax.transAxes)
        try:
            self.annotation_text = annotation
        except:
            pass
            
        ax.set_xlabel("Capacity (mAh)")
        ax.set_ylabel("Voltage vs. Li/ Li$^{+}$ (V)")
        ax.set_xlim([0, None])
        plt.tight_layout()
        return fig, ax