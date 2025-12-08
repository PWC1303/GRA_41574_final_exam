from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,precision_score,confusion_matrix,roc_curve,recall_score
import numpy as np
import json
def save_json(fname,obj):
         with open(fname, "w") as f:
                json.dump(obj,f)
def model_tester(y_te,yhat_prob,model_name,alpha,dset,ks):

        """
        args: 
                y_te: testing data for y 
                yhat_prob: array of probabilites 


                

        """

        yhat = np.where(yhat_prob>=alpha,1,0)
        test_recall = recall_score(y_te,yhat)
        test_acc = accuracy_score(y_te,yhat) 
        test_auc = roc_auc_score(y_te,yhat_prob)
        test_prescision = precision_score(y_te,yhat)
        test_f1 = f1_score(y_te,yhat)
        cm = confusion_matrix(y_te,yhat).tolist()
        if ks is True:
                file_name = f"testing_results/metrics/ks_dropped_{dset}_{model_name}_alpha_{alpha}_metrics_.json"
        else:
                file_name = f"testing_results/metrics/{dset}_{model_name}_alpha_{alpha}_metrics_.json"

        save_json(f"{file_name}", 
                  {"alpha": alpha,
                   "test_recall":test_recall,
                   "testing_accuracy": test_acc,
                   "test_roc_auc": test_auc,
                   "test_precision":test_prescision,
                   "test_f1": test_f1,
                   "test_confusion_matrix": cm } )
        print(f"Testing perfomed for dataeset with model {model_name} at {file_name}")