import numpy as np 
import pandas as pd
import json
import joblib

import argparse
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV

from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV
from utils.alpha_plot import plot_accuracy_vs_threshold,plot_recall_vs_threshold,plot_recall_accuracy
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import json

def save_json(fname,obj):
         with open(fname, "w") as f:
                json.dump(obj,f)




def main():
    parser = argparse.ArgumentParser(prog='ProgramName',description='What the program does',epilog='Text at the bottom of help')
    parser.add_argument("--model", type=str, default="logm_l1")
    parser.add_argument("--dset", type=str, default="white")
    parser.add_argument("--drop_ks", action="store_true")
    parser.add_argument("--cv", type = int, default=10, help ="Choose cv")
    parser.add_argument("--svc_type",type=str,default="rbf",   help= "choose between poly and rbf kernel")
    args = parser.parse_args()
    if args.dset =="white":
         data = "white"
    else:
         data = "red"
    df = pd.read_csv(f"data/wine_{data}_encoded.csv")
    y = np.array(df["lq"])



    if args.drop_ks is True:
      X = np.array(df.drop(columns= ["quality","lq","chlorides","density","sulphates","pH"]))
      
    else:
      X = np.array(df.drop(columns= ["quality","lq"]))
    
    X_temp,X_te,y_temp,y_te = train_test_split(
        X, y, test_size=0.2, random_state=33, stratify=y
    )

    X_tr,X_val,y_tr,y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=33, stratify=y_temp
    )



    #___________Logistic_L1____________________________________________________________________________________
    if args.model == "logm_l1":
        logm_l1= LogisticRegressionCV(penalty="l1",solver="liblinear", cv=args.cv,scoring="average_precision" ,
                  class_weight="balanced", max_iter=1000,random_state=33,verbose=1)
        model = make_pipeline(StandardScaler(),logm_l1)
        model.fit(X_tr,y_tr)
        results = model.named_steps["logisticregressioncv"]
        cv_pr_auc = results.scores_[1].mean()
        params = results.get_params()
        best_c = float(results.C_[0])
        prefix = f"ks_dropped_{args.dset}" if args.drop_ks else args.dset

        file_name_params  = f"tuning_results/params/{prefix}_{args.model}_params.json"
        file_name_results = f"tuning_results/cv_pr_auc/{prefix}_{args.model}_cv_pr_auc.json"
        file_name_model   = f"tuning_results/models/{prefix}_{args.model}.pkl"

        y_val_prob = model.predict_proba(X_val)[:, 1]
        
        
        plot_accuracy_vs_threshold(args.model,y_val,y_val_prob)
        plot_recall_vs_threshold(args.model,y_val,y_val_prob)
        plot_recall_accuracy(args.model,y_val,y_val_prob)
      
        save_json(file_name_params,{"params": params, "best_C": best_c})
        save_json(file_name_results, {"cv_pr_auc": cv_pr_auc})
        joblib.dump(model, file_name_model)
        print(f"Fitted on dataset wine_{args.dset}_encoded, params, cv_pr_auc and model.pkl saved for {args.model}")


    #___________Logistic_L2____________________________________________________________________________________
    if args.model == "logm_l2":
        logm_l2= LogisticRegressionCV(penalty="l2",solver="liblinear", cv=args.cv,scoring="average_precision" ,
                  class_weight="balanced", max_iter=1000,random_state=33,verbose=1)
        model = make_pipeline(StandardScaler(),logm_l2)
        model.fit(X_tr,y_tr)
        results = model.named_steps["logisticregressioncv"]
        cv_pr_auc = results.scores_[1].mean()
        params = results.get_params()
        best_c = float(results.C_[0])

        prefix = f"ks_dropped_{args.dset}" if args.drop_ks else args.dset

        file_name_params  = f"tuning_results/params/{prefix}_{args.model}_params.json"
        file_name_results = f"tuning_results/cv_pr_auc/{prefix}_{args.model}_cv_pr_auc.json"
        file_name_model   = f"tuning_results/models/{prefix}_{args.model}.pkl"
      
        y_val_prob = model.predict_proba(X_val)[:, 1]
        
        plot_accuracy_vs_threshold(args.model,y_val,y_val_prob)
        plot_recall_vs_threshold(args.model,y_val,y_val_prob)
        plot_recall_accuracy(args.model,y_val,y_val_prob)
      
        save_json(file_name_params,{"params": params, "best_C": best_c})
        save_json(file_name_results, {"cv_pr_auc": cv_pr_auc})
        joblib.dump(model, file_name_model)
        print(f"Fitted on dataset wine_{args.dset}_encoded, params, cv_pr_auc and model.pkl saved for {args.model}")


    #___________Logistic_Elastic_Net____________________________________________________________________________________    
    if args.model == "logm_ela":
        logm_ela = LogisticRegressionCV(penalty="elasticnet",solver="saga",
                                l1_ratios=[x for x in np.linspace(0,1,10)],cv =args.cv,scoring="average_precision", class_weight= "balanced",
                                max_iter=1000,random_state=33,verbose =1)
        model =make_pipeline(StandardScaler(),logm_ela)
        model.fit(X_tr,y_tr)
        results = model.named_steps["logisticregressioncv"]
        cv_pr_auc = results.scores_[1].mean()
        params = results.get_params()
        best_c = float(results.C_[0])
        l1_ratio = float(results.l1_ratio_[0])


        prefix = f"ks_dropped_{args.dset}" if args.drop_ks else args.dset
        file_name_params  = f"tuning_results/params/{prefix}_{args.model}_params.json"
        file_name_results = f"tuning_results/cv_pr_auc/{prefix}_{args.model}_cv_pr_auc.json"
        file_name_model   = f"tuning_results/models/{prefix}_{args.model}.pkl"
        y_val_prob = model.predict_proba(X_val)[:, 1]
        
        
        plot_accuracy_vs_threshold(args.model,y_val,y_val_prob)
        plot_recall_vs_threshold(args.model,y_val,y_val_prob)
        plot_recall_accuracy(args.model,y_val,y_val_prob)
        
        save_json(file_name_params,{"params": params, "best_C": best_c,"l1_ratio":l1_ratio})
        save_json(file_name_results, {"cv_pr_auc": cv_pr_auc})
        joblib.dump(model, file_name_model)
        print(f"Fitted on dataset wine_{args.dset}_encoded, params, cv_pr_auc and model.pkl saved for {args.model}")

if __name__ == "__main__":
    main()



