import numpy as np 
import pandas as pd
import json
import joblib

import argparse
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV

from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV

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
    parser.add_argument("--dset", type=str, default="red")
    parser.add_argument("--cv", type = int, default=10, help ="Choose cv")
    parser.add_argument("--svc_type",type=str,default="rbf",   help= "choose between poly and rbf kernel")
    args = parser.parse_args()
    if args.dset =="white":
         data = "white"
    else:
         data = "red"
    df = pd.read_csv(f"data/{data}quality_encoded.csv")
    y = np.array(df["lq"])
    X = np.array(df.drop(columns= ["quality","lq"]))
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.3,random_state=33)


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
        save_json(f"tuning_results/params/{args.model}_params.json",
          {"params": params, "best_C": best_c})
        save_json(f"tuning_results/cv_pr_auc/{args.model}_cv_pr_auc.json", {"cv_pr_auc": cv_pr_auc})
        joblib.dump(model, f"tuning_results/models/{args.model}.pkl")
        print(f"params, cv_pr_auc and model.pkl saved for {args.model}")


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
        
        save_json(f"tuning_results/params/{args.model}_params.json",
          {"params": params, "best_C": best_c})
        save_json(f"tuning_results/cv_pr_auc/{args.model}_cv_pr_auc.json", {"cv_pr_auc": cv_pr_auc})
        joblib.dump(model, f"tuning_results/models/{args.model}.pkl")
        print(f"params, cv_pr_auc and model.pkl saved for {args.model}")

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
        save_json(f"tuning_results/params/{args.model}_params.json",
          {"params": params, "best_C": best_c,"l1_ratio":l1_ratio})
        save_json(f"tuning_results/cv_pr_auc/{args.model}_cv_pr_auc.json", {"cv_pr_auc": cv_pr_auc})
        joblib.dump(model, f"tuning_results/models/{args.model}.pkl")
        print(f"params, cv_pr_auc and model.pkl saved for {args.model}")

    if args.model =="rfc":
          param_dist = {
            "n_estimators":      [300, 400],
            "max_depth":         [None, 6, 12],
            "min_samples_leaf":  [1, 2, 4],
            "min_samples_split": [2, 5, 10],
            "max_samples": [0.6, 0.8, None],
            "bootstrap": [True],
            "class_weight": [ "balanced",None],
            "max_features":      ["sqrt",None] 
        }

          rfc = RandomForestClassifier(n_jobs=-1,random_state=33)
          grid_search = GridSearchCV(
          estimator=rfc,
          param_grid=param_dist,
          cv=3,
          scoring="average_precision",   
          verbose=2,
          n_jobs=-1,)
          grid_search.fit(X_tr,y_tr)
          params = grid_search.best_params_
          cv_pr_auc = grid_search.best_score_
          save_json(f"tuning_results/params/{args.model}_params.json",
            {"params": params})
          save_json(f"tuning_results/cv_pr_auc/{args.model}_cv_pr_auc.json", {"cv_pr_auc": cv_pr_auc})
          joblib.dump(grid_search.best_estimator_, f"tuning_results/models/{args.model}.pkl")
          print(f"params, cv_pr_auc and model.pkl saved for {args.model}")




















































































    #___________SVC______________________________________________________________________________________-
    if args.model =="svc":
        svc = SVC(probability=True,class_weight="balanced")
        model = make_pipeline(StandardScaler(),svc)
        if args.svc_type == "poly":
            with open("tuning_results/param_dist/svc_param_dist_poly.json", "r") as f:
                param_dist = json.load(f)

        if args.svc_type == "rbf":
            with open("tuning_results/param_dist/svc_param_dist_rbf.json", "r") as f:
                param_dist = json.load(f)

      
        grid_search = GridSearchCV(
              estimator=model,
              param_distributions=param_dist,  
              cv=3,
              scoring="average_precision",
              n_jobs=-1,
              verbose=2)

        grid_search.fit(X_tr,y_tr)

        params = grid_search.best_params_
        cv_pr_auc = grid_search.best_score_
        

        save_json(f"tuning_results/params/{args.model}_{args.svc_type}_params.json",
          {"params": params})
        save_json(f"tuning_results/cv_pr_auc/{args.model}_{args.svc_type}_cv_pr_auc.json", {"cv_pr_auc": cv_pr_auc})
        joblib.dump(grid_search.best_estimator_, f"tuning_results/models/{args.model}.pkl")
        print(f"params, cv_pr_auc and model.pkl saved for {args.model}_type_{args.svc_type} ")
    #____________Random_Forest__________________________________________________________________________________________
    if args.model =="rfc":
      param_dist = {
        "n_estimators":      [300, 400],
        "max_depth":         [None, 6, 12],
        "min_samples_leaf":  [1, 2, 4],
        "min_samples_split": [2, 5, 10],
        "max_samples": [0.6, 0.8, None],
        "bootstrap": [True],
        "class_weight": [ "balanced",None],
        "max_features":      ["sqrt",None] 
    }

      rfc = RandomForestClassifier(n_jobs=-1,random_state=33)
      grid_search = GridSearchCV(
      estimator=rfc,
      param_grid=param_dist,
      cv=3,
      scoring="average_precision",   
      verbose=2,
      n_jobs=-1,)
      grid_search.fit(X_tr,y_tr)
      params = grid_search.best_params_
      cv_pr_auc = grid_search.best_score_
      save_json(f"tuning_results/params/{args.model}_params.json",
        {"params": params})
      save_json(f"tuning_results/cv_pr_auc/{args.model}_cv_pr_auc.json", {"cv_pr_auc": cv_pr_auc})
      joblib.dump(grid_search.best_estimator_, f"tuning_results/models/{args.model}.pkl")
      print(f"params, cv_pr_auc and model.pkl saved for {args.model}")
if __name__ == "__main__":
    main()
       
        




