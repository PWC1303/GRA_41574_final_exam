import pandas as pd






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
import matplotlib.pyplot as plt




def main():
 
    parser = argparse.ArgumentParser(prog='ProgramName',description='What the program does',epilog='Text at the bottom of help')
    parser.add_argument("--model", type=str, default="logistic")
    parser.add_argument("--dset", type=str, default="white")
    args = parser.parse_args()

    df = pd.read_csv(f"data/wine_{args.dset}_encoded.csv")

    X =df.drop(columns= ["quality","lq"])

    features = [col for col in X.columns]
    model = joblib.load(f"tuning_results/models/{args.dset}_{args.model}.pkl")
    if args.model == "rfc":
        importances = model.feature_importances_
        forest_importances = pd.Series(importances, index=features)
        forest_importances = forest_importances.sort_values(ascending=False)

        fig, ax = plt.subplots()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        forest_importances.plot.bar(ax=ax, color="cornflowerblue")

        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")

        fig.tight_layout()
        fig.savefig("figs/rfc_feature_imp")

        print(forest_importances)

    if args.model.startswith("logm"):
            
        coefs = model[-1].coef_.ravel()

        logm_importance = (
            pd.DataFrame({
                "feature": X.columns,
                "abs_importance": np.abs(coefs)
            })
            .set_index("feature")
            .sort_values(by="abs_importance", ascending=False)
        )

        fig, ax = plt.subplots()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        logm_importance.plot.bar(ax=ax, legend=False, color="cornflowerblue")

        ax.set_title("Feature importances using absolute coefficient values")
        ax.set_ylabel(r"$|\hat{\beta}|$")
        fig.tight_layout()
        fig.savefig(f"figs/{args.model}_feature_imp")
        logm_importance = (
            pd.DataFrame({
                "feature": X.columns,
                "abs_importance": coefs
            })
            .set_index("feature")
            .sort_values(by="abs_importance", ascending=False)
        )
        print("Now printing coefs with the sign for intepretation")
        print(logm_importance)

    

    

if __name__ == "__main__":
    main()