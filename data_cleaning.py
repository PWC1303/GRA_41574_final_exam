import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 



def main():
    df = pd.read_csv("data/winequality_red.csv", sep =";")
    df["lq"] = np.where((df["quality"] == 3) | (df["quality"] == 4),1,0)
    df.to_csv("data/wine_red_encoded.csv")
    df = pd.read_csv("data/winequality_white.csv", sep =";")
    df["lq"] = np.where((df["quality"] == 3) | (df["quality"] == 4),1,0)
    df.to_csv("data/wine_white_encoded.csv")
 
    
main()



