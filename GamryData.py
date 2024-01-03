class GamryData(object):
    def __init__(self, filename, CRate=0.05, mass=None):
        import pandas as pd
        import numpy as np
        
        data_read = []
        with open(filename, encoding="latin") as f:
            try:
                for line in f.readlines():
                    data_read.append(line.strip("\n").split("\t"))
            except:
                print(filename)
        end_of_header = [nline for nline, line in enumerate(data_read) if len(line)>1 and line[1]=="#"][0]
        headers = data_read[end_of_header-1][1:-1]
        data = np.array(np.array([line[1:-1] for line in data_read[end_of_header+1:-1] if len(line[1:-1])==len(headers)]), dtype=float)
        self.df = pd.DataFrame(data, columns=headers)
        
        current = abs(np.nanmedian(self.df["Im"])) ##A
        theoretical_capacity = 1.675 ## Ah/g
        
        if CRate != None:
            mass = current/theoretical_capacity/CRate        
            self.capacity = abs((self.df["T"]/60/60)*self.df["Im"]*1000)/mass
            
        elif CRate == None and mass== None:
            self.capacity = abs((self.df["T"]/60/60)*self.df["Im"]*1000)
            
        elif mass != None:
            self.capacity = abs((self.df["T"]/60/60)*self.df["Im"]*1000)/mass

        self.voltage = self.df["Vf"]