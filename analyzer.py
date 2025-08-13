import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

print("Loading Dataset...")
pre_label_df = pd.read_csv('/home/brayden/GitHub/Health-AI-CLPsych/results/expert-gemma-2-2b.csv', header=0)
label_df = pre_label_df.iloc[0: len(pre_label_df)]#Ensure that this properly takes into account the train test split

y_true = label_df['raw_label'].tolist() 
y_pred = label_df['predictions'].tolist()
print("Dataset Loaded...")
print(f"Size of pre datasset: {len(pre_label_df)}")
print(f"Size of dataset: {len(y_true)}")

#Get prediction results
print("Overall Metrics:")
print(classification_report(y_true, y_pred, digits=4))

print(f"Datapoints: {len(y_true)}")
print(f"Datapoints with predictions: {len(y_pred)}")
print(f"Instances with no predictions: {len([x for x in y_pred if x == '?'])}")
print(f"Unique labels in true data: {set(y_true)}")
print(f"Instances of true label 'a': {y_true.count('a')}")
print(f"Instances of predicted label 'a': {y_pred.count('a')}\n")
print(f"Instances of true label 'b': {y_true.count('b')}")
print(f"Instances of predicted label 'b': {y_pred.count('b')}\n")
print(f"Instances of true label 'c': {y_true.count('c')}")
print(f"Instances of predicted label 'c': {y_pred.count('c')}\n")
print(f"Instances of true label 'd': {y_true.count('d')}")
print(f"Instances of predicted label 'd': {y_pred.count('d')}")

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
plt.savefig('/home/umflint.edu/brayclou/Health-AI-CLPsych/results/llama_baseline.png')
plt.show()
plt.close()
print("ROC curve saved as llama_baseline.png")