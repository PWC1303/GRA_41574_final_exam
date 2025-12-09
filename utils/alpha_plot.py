
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score

def plot_recall_vs_threshold(model_name,y_true, y_prob, n_points=200):
    thresholds = np.linspace(0, 1, n_points)
    recalls = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        recalls.append(recall_score(y_true, y_pred))

    plt.figure()
    fig, ax = plt.subplots()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.plot(thresholds, recalls, color = "magenta")
    plt.xlabel("Threshold")
    plt.ylabel("Recall")
    plt.title("Recall vs Threshold (Validation Set)")
    plt.savefig(f"figs/{model_name}_recall_vs_alpha")

from sklearn.metrics import accuracy_score

def plot_accuracy_vs_threshold(model_name,y_true, y_prob, n_points=200):
    thresholds = np.linspace(0, 1, n_points)
    accuracies = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        accuracies.append(accuracy_score(y_true, y_pred))

    plt.figure()
    fig, ax = plt.subplots()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.plot(thresholds, accuracies, color = "magenta")
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs Threshold (Validation Set)")
    plt.savefig(f"figs/{model_name}_acc_vs_alpha")



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, accuracy_score

def plot_recall_accuracy(model_name, y_true, y_prob, n_points=200):
    thresholds = np.linspace(0, 1, n_points)
    recalls = []
    accuracies = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        recalls.append(recall_score(y_true, y_pred))
        accuracies.append(accuracy_score(y_true, y_pred))
    plt.figure()
    fig, ax = plt.subplots()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
   
    plt.plot(thresholds, recalls, label="Recall",color= "magenta")
    plt.plot(thresholds, accuracies, label="Accuracy", color ="cornflowerblue")
    plt.xlabel("Threshold")
    plt.ylabel("Value")
    plt.title("Recall & Accuracy vs Threshold (Validation Set)")
    plt.legend()
    plt.savefig(f"figs/{model_name}_recall_accuracy_overlay")
