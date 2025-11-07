import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

directory = '/home/umflint.edu/brayclou/Health-AI-CLPsych/results/llama-2step-10epochs.csv'

print("Loading Dataset...")
pre_label_df = pd.read_csv(directory, header=0)
iloc_label_df = pre_label_df.iloc[0: len(pre_label_df)]#Ensure that this properly takes into account the train test split
label_df = iloc_label_df[iloc_label_df["predictions"].astype(str).str.strip() !="?"]

#label_df = iloc_label_df.filter(lambda x: x['raw_label'] != "?")
pre_y_true = label_df['raw_label'].tolist() 


y_true = []
for true in pre_y_true:
    # print(f"{type(true)}: {true}")
    if type(true) == float:
        y_true.append("no")
    elif true == "b" or true == "c" or true == "d":
        y_true.append("yes")
    else: raise ValueError
y_pred = label_df['predictions'].tolist()
print("Dataset Loaded...")

#Get prediction results
print("Overall Metrics:")
print(directory)
print(classification_report(y_true, y_pred, digits=4))

print(f"Datapoints with predictions: {len(y_pred)}/{len(y_true)}")
print(f"Unique labels in true data: {set(y_true)}")
print(f"Instances of true label 'yes': {y_true.count('yes')}")
print(f"Instances of predicted label 'yes': {y_pred.count('yes')}\n")
print(f"Instances of true label 'no': {y_true.count('no')}")
print(f"Instances of predicted label 'no': {y_pred.count('no')}\n")

print("Granular Metrics:")
#pre_y_true is the list of true labels "b", "c", "d", ""
#y_pred is the list of predicted labels "yes", "no"
#y_true is pre_y_true converted to "yes" and "no"
b_correct = 0
c_correct = 0
d_correct = 0
n_correct = 0
for count in range(len(y_true)):
    if pre_y_true[count] == "b":
        if y_pred[count] == "yes":
            b_correct += 1
    elif pre_y_true[count] == "c":
        if y_pred[count] == "yes":
            c_correct += 1
    elif pre_y_true[count] == "d":
        if y_pred[count] == "yes":
            d_correct += 1
    else:
        if y_pred[count] == "no":
            n_correct += 1
print(f"Correct 'b' predictions: {b_correct}/{pre_y_true.count('b')}")
print(f"Correct 'c' predictions: {c_correct}/{pre_y_true.count('c')}")
print(f"Correct 'd' predictions: {d_correct}/{pre_y_true.count('d')}")
print(f"Correct 'no' predictions: {n_correct}/{y_true.count('no')}")
        







# y_pred_bin = label_binarize(y_pred, classes=['a', 'b', 'c', 'd'])

# print("AUC and ROC")
# fpr, tpr, thresholds = roc_curve(y_true, y_pred)
# print(f"Thresholds: {thresholds}")
# roc_auc = roc_auc_score(y_true, y_pred)
# # Plot ROC curve
# plt.figure()
# plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # diagonal
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC: Multiclass(a, b, c, d)')
# plt.legend(loc='lower right')
# plt.savefig('/home/umflint.edu/brayclou/Health-AI-CLPsych/results/llama_engineered.png')
# plt.show()
# plt.close()
# print("ROC curve saved as llama_engineered.png")
