import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

directory = '/home/umflint.edu/brayclou/Health-AI-CLPsych(old)/crowd-test-llama-3.2-base.csv'

print("Loading Dataset...")
pre_label_df = pd.read_csv(directory, header=0)
iloc_label_df = pre_label_df.iloc[0: len(pre_label_df)]#Ensure that this properly takes into account the train test split
label_df = iloc_label_df[iloc_label_df["predictions"].astype(str).str.strip() !="?"]

#label_df = iloc_label_df.filter(lambda x: x['raw_label'] != "?")
pre_y_true = label_df['raw_label'].tolist() 
print(label_df)


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
print(f"Size of pre datasset: {len(pre_label_df)}")
print(f"Size of dataset: {len(y_true)}")

#Get prediction results
print("Overall Metrics:")
print(classification_report(y_true, y_pred, digits=4))
print(directory)
print(directory)

print(f"Datapoints: {len(y_true)}")
print(f"Datapoints with predictions: {len(y_pred)}")
print(f"Instances ? predictions: {len([x for x in iloc_label_df['predictions'].tolist() if x == '?'])}\n")
print(f"Unique labels in true data: {set(y_true)}")
print(f"Instances of true label 'yes': {y_true.count('yes')}")
print(f"Instances of predicted label 'yes': {y_pred.count('yes')}\n")
print(f"Instances of true label 'no': {y_true.count('no')}")
print(f"Instances of predicted label 'no': {y_pred.count('no')}\n")

print(f"Instances of true label 'yes': {y_true.count('yes')}")
print(f"Instances of predicted label 'yes': {y_pred.count('yes')}\n")
print(f"Instances of true label 'no': {y_true.count('no')}")
print(f"Instances of predicted label 'no': {y_pred.count('no')}\n")


exit()

# y_pred_bin = label_binarize(y_pred, classes=['a', 'b', 'c', 'd'])

print("AUC and ROC")
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
print(f"Thresholds: {thresholds}")
roc_auc = roc_auc_score(y_true, y_pred)
# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # diagonal
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC: Multiclass(a, b, c, d)')
plt.legend(loc='lower right')
plt.savefig('/home/umflint.edu/brayclou/Health-AI-CLPsych/results/llama_engineered.png')
plt.show()
plt.close()
print("ROC curve saved as llama_engineered.png")
