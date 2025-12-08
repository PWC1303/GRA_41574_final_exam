import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 



def main():
 
    
    #____________RED WINE_____________________________________________________________________
    df = pd.read_csv("data/winequality_red.csv", sep =";")
    print(f"Number of observations in red wine dataset {df.shape[0]}")
    plt.hist(df["quality"],color ="darkred")
    plt.title("Red Wine Quality Scores")
    plt.savefig("figs/red_qual_dist")
    plt.close()
    low = (df["quality"] < 7).sum()
    high = (df["quality"] >= 7).sum()
    badpercent =  low / (low + high)
    print(f"percentage of low quality red wine according to that one study: {badpercent}")
    low = (df["quality"] < 5).sum()
    high = (df["quality"] >= 5).sum()
    badpercent =  low / (low + high)
    print(f"percentage of low quality red wine according to our methodology:{badpercent}")
    df["lq"] = np.where((df["quality"] == 3) | (df["quality"] == 4),1,0)

    print(df["quality"].value_counts())
    df.to_csv("data/wine_red_encoded.csv",index= False)
    
    
    
    
    #____________WHITE WINE_____________________________________________________________________
    df = pd.read_csv("data/winequality_white.csv", sep =";")
    print(f"Number of observations in white wine dataset {df.shape[0]}")

    
    plt.hist(df["quality"], color ="beige")
    plt.title("White Wine Quality Scores")
    plt.savefig("figs/white_qual_dist")
    plt.close()
    low = (df["quality"] < 5).sum()
    high = (df["quality"] >= 5).sum()
    badpercent =  low / (low + high)
    print(f"percentage of low quality white wine according to our methodologys:{badpercent}")
    mid = ((df["quality"] == 6) | (df["quality"] == 5)).sum()
    notmid = ((df["quality"] != 6) | (df["quality"] != 5)).sum()
    print(mid/(notmid+mid))

    df["lq"] = np.where((df["quality"] == 3) | (df["quality"] == 4),1,0)
    print(df["quality"].value_counts())
    print((df["lq"] == 1).sum())
    df.to_csv("data/wine_white_encoded.csv",index=False)



 
    
main()



