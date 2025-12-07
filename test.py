
import numpy as np 
import pandas as pd
import json
import joblib
from utils.model_tester import model_tester

import argparse
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV

from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,precision_score,confusion_matrix,roc_curve
import json
import json

def save_json(fname,obj):
         with open(fname, "w") as f:
                json.dump(obj,f)


def main():
    def save_json(fname,obj):
         with open(fname, "w") as f:
                json.dump(obj,f)
    parser = argparse.ArgumentParser(prog='ProgramName',description='What the program does',epilog='Text at the bottom of help')
    parser.add_argument("--model", type=str, default="logistic")
    parser.add_argument("--alpha", type = float, default= 0.5, help= "Alpha sets the the prediction threhsolds")
    args = parser.parse_args()
    df = pd.read_csv("data/wine_red_encoded.csv")
    y = np.array(df["lq"])
    X = np.array(df.drop(columns= ["quality","lq"]))
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.3,random_state=33)


    


    if args.model == "logm_l1":
        model = joblib.load(f"tuning_results/models/{args.model}.pkl")
        yhat_prob= model.predict_proba(X_te)[:,1]
        model_tester(y_te,yhat_prob,args.model,args.alpha)

       
        

    if args.model == "logm_l2":
        model = joblib.load(f"tuning_results/models/{args.model}.pkl")
        yhat_prob= model.predict_proba(X_te)[:,1]
        model_tester(y_te,yhat_prob,args.model,args.alpha)
  


    if args.model == "logm_ela":
        model = joblib.load(f"tuning_results/models/{args.model}.pkl")
        yhat_prob= model.predict_proba(X_te)[:,1]
        model_tester(y_te,yhat_prob,args.model,args.alpha)


    if args.model.startswith("svc"):
        model = joblib.load(f"tuning_results/models/{args.model}.pkl")
        yhat_prob= model.predict_proba(X_te)[:,1]
        model_tester(y_te,yhat_prob,args.model,args.alpha)
    if args.model =="rfc":
        model = joblib.load(f"tuning_results/models/{args.model}.pkl")
        yhat_prob= model.predict_proba(X_te)[:,1]
        model_tester(y_te,yhat_prob,args.model,args.alpha)
    

if __name__ == "__main__":
    main()
       