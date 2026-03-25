import numpy as np
def confusion_matrix(labels: np.ndarray, predictions: np.ndarray):
    t_positive= int(np.sum((predictions == 1) & (labels == 1)))
    f_positive= int(np.sum((predictions == 1) & (labels == 0)))
    t_negative= int(np.sum((predictions == 0) & (labels == 0)))
    f_negative= int(np.sum((predictions == 0) & (labels == 1)))
    return {"TP": t_positive, "FP": f_positive, "TN": t_negative, "FN": f_negative}


def compute_metrics(labels: np.ndarray, predictions: np.ndarray):
    cm= confusion_matrix(labels, predictions)
    t_positive, f_positive, t_negative, f_negative    = cm["TP"], cm["FP"], cm["TN"], cm["FN"]
    accuracy=(t_positive+t_negative)/max(t_positive +f_positive+ t_negative +f_negative,1)
    tpr=t_positive/max(t_positive + f_negative, 1)
    tnr= t_negative/max(t_negative+f_positive, 1)
    fpr=f_positive/max(f_positive + t_negative, 1)
    precision =t_positive /max(t_positive +f_positive,1)
    f1=(2 *precision* tpr)/max(precision+tpr,1e-10)
    balanced_accuracy=(tpr+tnr)/2
    return {"accuracy":round(accuracy,4),"balanced_accuracy":round(balanced_accuracy,4),"f1":round(f1,4),"tpr":round(tpr, 4),"fpr": round(fpr, 4),"precision":round(precision, 4),"confusion_matrix":  cm}

def threshold_sweep(scores: np.ndarray, labels: np.ndarray,thresholds: np.ndarray):


    results = []
    for i in thresholds:
        preds = (scores >= i).astype(np.int32)
        m = compute_metrics(labels, preds)
        m["threshold"]=round(float(i), 6)
        results.append(m)


    return results


def select_threshold(sweep_results: list,rule:str= "max_balanced_accuracy"):

    if rule == "max_balanced_accuracy":
        return max(sweep_results, key=lambda x: x["balanced_accuracy"])["threshold"]
    elif rule == "max_f1":
        return max(sweep_results, key=lambda x: x["f1"])["threshold"]
    raise ValueError(f"threshold rule : '{rule}' not supported , Please only use max_balanced_accuracy or max_f1")




def roc_data(scores: np.ndarray, labels:np.ndarray,thresholds: np.ndarray):
    fprs=[]
    tprs= []
    for i in thresholds:


        preds = (scores >= i).astype(np.int32)
        cm=confusion_matrix(labels, preds)

        tprs.append(cm["TP"] /max(cm["TP"]+ cm["FN"],1))
        fprs.append(cm["FP"] /max(cm["FP"]+ cm["TN"],1))

    return np.array(fprs),np.array(tprs)