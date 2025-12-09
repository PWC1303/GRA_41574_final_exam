


tune.py
Command-Line Arguments
---------------------------------------------------------------------------------------------------------
Argument	                    Default	            Description
--dset	                        white	            Choose between white and red dataset to fit model on (white was used for report, red is for convience if I wanted to compare)
--model                         logm_l1  	        Choose what model to fit. (logm_l1 = Logistic Regression with L1 penalty, logm_l2 for l2 penalty, rfc for Random Forest Classifer)
--cv	                        10    	            Chose how many folds to do CV on, not a choice for RFC is hardcoded as 3 w.r.t computational concerncs



test.py
Command-Line Arguments
---------------------------------------------------------------------------------------------------------
Argument	                    Default	            Description
--dset	                        white	            Choose between white and red dataset to fit model on (white was used for report, red is for convience if I wanted to compare)
--models                        logm_l1  	        Choose what model to to test on. (logm_l1 = Logistic Regression with L1 penalty, logm_l2 for l2 penalty, rfc for Random Forest Classifer)
--alpha                         0.5                 Chose what threshold to use to compute testing metrics. I used the plots to select this manually


statistics.py
Command-Line Arguments
---------------------------------------------------------------------------------------------------------
Argument	                    Default	            Description
--dset	                        white	            Choose between white and red dataset to compute summary statistics and visualize quality distrubution (white was used for report, red is for convience if I wanted to compare)




ist Data for training/testing to folder \data   Subclass of DataLoader

Helper programs
-----------------------------------------------------------------------------------------------------------
Name                Purpose                              
alpha_plot.py       Visualize accuracy/recallas a function of the threshold      
model_tester.py     perfom test of models on all project relevant metrics and some more              
data_encoding.py    Encode the low quality target variable 

Folders
Name                Purpose                                 
\figs               Store figures
\data               Store data 
\testing reuslts    Store all results from testing
\tuning results     Store results from tuning (cv_scores) and model.pkl files(useful for not having to retrain the RFC just to test it)
\utils              Store helper programs except data_ecnoding.py