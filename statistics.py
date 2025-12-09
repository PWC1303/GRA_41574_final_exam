import pandas as pd 
import matplotlib.pyplot as plt 
import argparse



def main():
   parser = argparse.ArgumentParser(prog="visualizations",description='Visualises the distributions of quality and computes ratio of low quality wine for selected dataset')
   parser.add_argument("--dset",type =str,default="white",help = "Select dset white or red (white is used in report)")
   args = parser.parse_args()
   df = pd.read_csv(f"data\wine_{args.dset}_encoded.csv")

   

   print(f"Number of observations in {args.dset} wine dataset {df.shape[0]}")
   fig, ax = plt.subplots()
   ax.spines["top"].set_visible(False)
   ax.spines["right"].set_visible(False)
   plt.hist(df["quality"],color ="cornflowerblue")
   plt.savefig(f"figs/{args.dset}_qual_dist.png")
   plt.close()
   low = (df["quality"] < 5).sum()
   high = (df["quality"] >= 5).sum()
   badpercent =  low / (low + high)
   print(f"percentage of low quality {args.dset} wine according to our methodology:{badpercent}")
   plt.close()


   #_________________Summary Statistics_____________________________________
   pd.set_option("display.max_columns", None)
   pd.set_option("display.max_rows", None)
   pd.set_option("display.width", None)
   df = df.drop(columns = ["lq","quality"])
   print(df.describe())
if __name__ == "__main__":
    main()















