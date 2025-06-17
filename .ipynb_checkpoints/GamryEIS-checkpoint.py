class GamryEIS(object):
    def __init__(self, filename):
        import pandas as pd
        
        import warnings
        warnings.filterwarnings(action="ignore", category=FutureWarning)
        
        data = []
        data_read = []
        header_read = True

        with open(filename) as f:
            for nline, line in enumerate(f.readlines()):
                lineread = line.strip("\n").split("\t")
                data_read.append(lineread)
                if header_read == False:
                    data.append(lineread)

                if len(lineread)> 1:
                    if lineread[1] == "#":
                        header_read = False

                        columns = data_read[nline-1]

        df = pd.DataFrame(data, dtype=float, columns=columns)
        
        self.df = df
        self.freq = df["Freq"].to_numpy()
        self.Re_Z = df["Zreal"].to_numpy()
        self.Im_Z = -df["Zimag"].to_numpy()