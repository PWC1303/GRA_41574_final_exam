
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import joblib
from utils.model_tester import model_tester



def main():
 
    parser = argparse.ArgumentParser(prog='test.py',description='Program tests the trained models using a helper tester function',epilog='Text at the bottom of help')
    parser.add_argument("--model", type=str, default="logistic")
    parser.add_argument("--dset", type=str, default="white")
    parser.add_argument("--alpha", type = float, default= 0.5, help= "Alpha sets the the prediction threhsolds")
    args = parser.parse_args()
    df = pd.read_csv(f"data/wine_{args.dset}_encoded.csv")
    y = np.array(df["lq"])
    X = np.array(df.drop(columns= ["quality","lq"]))
    
    
    model = joblib.load(f"tuning_results/models/{args.dset}_{args.model}.pkl")
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=33,stratify=y)
    yhat_prob= model.predict_proba(X_te)[:,1]
    model_tester(y_te,yhat_prob,args.model,args.alpha,args.dset)

if __name__ == "__main__":
    main()
       