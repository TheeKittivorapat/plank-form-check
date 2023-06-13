import pandas as pd
import numpy as np
from fastai.tabular.all import *
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data
data = pd.read_csv('data.csv')

# Split data into features and labels
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Convert numpy arrays to pandas DataFrame
X_train_df = pd.DataFrame(X, columns=data.columns[1:])
y_train_df = pd.DataFrame(y, columns=['label'])

# Split data into sets
splits = RandomSplitter(valid_pct=0.2, seed=42)(range_of(X))
X_train, X_valid, y_train, y_valid = X[splits[0]], X[splits[1]], y[splits[0]], y[splits[1]]
splits = RandomSplitter(valid_pct=0.5, seed=42)(range_of(X_valid))
X_valid, X_test, y_valid, y_test = X_valid[splits[0]], X_valid[splits[1]], y_valid[splits[0]], y_valid[splits[1]]

# Define data transformations
procs = [Normalize()]

# Create data loaders
label_dict = {0: 'correct', 1: 'high', 2: 'low'} 
dls = TabularDataLoaders.from_df(data, procs=procs, valid_idx=list(range(len(X_train), len(X))), bs=64, y_names='label')

# Define model architecture
learn = tabular_learner(dls, layers=[64], metrics=accuracy)

# Train model
learn.fit_one_cycle(10)

# Evaluate model on validation set
valid_preds = np.argmax(learn.get_preds(dl=dls.test_dl(pd.DataFrame(X_valid, columns=X_train_df.columns)))[0].numpy(), axis=1).astype(int)
# Convert integer labels back to string labels
valid_preds_str = np.vectorize(label_dict.get)(valid_preds)
valid_acc = accuracy_score(y_valid, valid_preds_str)
print(f'Validation Accuracy: {valid_acc:.4f}')
# Display confusion matrix for validation set
valid_cm = confusion_matrix(y_valid, valid_preds_str, labels=['correct', 'high', 'low'])
print('Validation Confusion Matrix:')
print(valid_cm)

# Evaluate model on test set
test_preds = np.argmax(learn.get_preds(dl=dls.test_dl(pd.DataFrame(X_test, columns=X_train_df.columns)))[0].numpy(), axis=1).astype(int)
# Convert integer labels back to string labels
test_preds_str = np.vectorize(label_dict.get)(test_preds)
test_acc = accuracy_score(y_test, test_preds_str)
print(f'Test Accuracy: {test_acc:.4f}')
# Display confusion matrix for test set
test_cm = confusion_matrix(y_test, test_preds_str, labels=['correct', 'high', 'low'])
print('Test Confusion Matrix:')
print(test_cm)

learn.export('model.pkl')