import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import ks_2samp
import argparse



def main():
   parser = argparse.ArgumentParser(prog='ProgramName',description='What the program does',epilog='Text at the bottom of help')
   parser.add_argument("--dset",type =str,default="red",help = "Select dset for KS, visualizing distribution of signficant features, and doing KDE")
   args = parser.parse_args()
   df = pd.read_csv(f"data\wine_{args.dset}_encoded.csv")
   features = [col for col in df.columns if col  != "lq" and col !="quality" ]


   #__________________Kolmogorov–Smirnov test____________________________________________________________-

   ks_results = {}
   for col in features:
      x1 = df.loc[df["lq"] == 1, col]
      x0 = df.loc[df["lq"] == 0, col]
      stat, pval = ks_2samp(x1, x0)
      ks_results[col] = {"KS_stat": stat, "p_value": pval}


   ks_df = pd.DataFrame(ks_results).T.sort_values("p_value")
   alpha = 0.05
   m = len(features)  # number of tests
   ks_df["p_bonf"] = ks_df["p_value"] * m
   ks_df["signif_bonf"] = ks_df["p_bonf"] < alpha
   print(ks_df)
   
   
   #______________________Visualzing distributions of Significant features_________________________________________________________
   signif_features = ks_df.index[ks_df["signif_bonf"]].tolist()
   df.loc[df["lq"] == 1, signif_features].hist(bins=50, figsize=(16, 12))
   plt.tight_layout()
   plt.savefig(f"figs\{args.dset}_signif_low_qual_dist")
   plt.close()
   df.loc[df["lq"] == 0, signif_features].hist(bins=50, figsize=(16, 12))
   plt.tight_layout()
   plt.savefig(f"figs\{args.dset}_signif_ok_qual_dist")
   plt.close()


   #______________________KDE on Significant features_________________________________________________________
   n = len(signif_features)
   ncols = 2
   nrows = (n + 1) // ncols
   fig, axes = plt.subplots(nrows, ncols, figsize=(16, 12))
   axes = axes.flatten()
   for i, col in enumerate(signif_features):
      sns.kdeplot(
         df.loc[df["lq"] == 1, col],
         ax=axes[i],
         label="Low quality",
         fill=True
      )
      sns.kdeplot(
         df.loc[df["lq"] == 0, col],
         ax=axes[i],
         label="OK quality",
         fill=True
      )
      
      axes[i].set_title(col)
      axes[i].legend()


   for j in range(i + 1, len(axes)):
      fig.delaxes(axes[j])
   plt.tight_layout()
   plt.savefig(f"figs/{args.dset}_kde_grid_signif.png")
   plt.close()


   


   #nbadqual = df["lq"].sum()/df.shape[0]

   #print(f"Number of low qual in dset {nbadqual}")


   #df.loc[df["lq"] == 1, features].hist(bins=50, figsize=(16, 12))
   #plt.tight_layout()
   #plt.savefig("figs\low_qual_dist")


   #df.loc[df["lq"] == 0, features].hist(bins=50, figsize=(16, 12))
   #plt.tight_layout()
   #plt.savefig("figs\ok_qual_dist")

if __name__ == "__main__":
    main()















