import os

#%matplotlib inline
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    make_scorer,
    #plot_confusion_matrix,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC

df = pd.read_csv("tropicalfishdiseasedata.csv", encoding="utf-8")
df.head()

df.shape


import sklearn
print(sklearn.__version__)

df.info()

df[["disease"]].value_counts()

df[["disease"]].value_counts().shape

train_df, test_df = train_test_split(df, test_size=0.3, random_state=123)

X_train = train_df.drop(columns=["disease", "source"])
y_train = train_df["disease"]

X_test = test_df.drop(columns=["disease", "source"])
y_test = test_df["disease"]

dc = DummyClassifier(strategy="most_frequent")

dc.fit(X_train, y_train)

print(dc.score(X_train, y_train))

lr = LogisticRegression(max_iter = 300)

lr.fit(X_train, y_train)
lr.score(X_test, y_test)


C_vals = 10.0 ** np.arange(-2, 2, 0.5)

score_best = 0
C_best = 0

scores = []

for c in C_vals:
    lr = LogisticRegression(max_iter=300, C = c)
    lr.fit(X_train, y_train)
    s = lr.score(X_test, y_test)
    scores.append(s)
    if (s > score_best):
        score_best = s
        C_best = c

plt.plot(C_vals, scores)
plt.title("Accuracy vs. C")
plt.xlabel("C")
plt.ylabel("Accuracy")
        
print("Best accuracy was ", score_best, " with C value = ", C_best) 



ex = X_train.iloc[[0]]
ex.columns[      
    (ex == 1)        # mask 
    .any(axis=0)     # mask
]


lr = LogisticRegression(max_iter = 300, C = 3.16)

lr.fit(X_train, y_train)

p = lr.predict_proba(ex)[0]

diseases = lr.classes_

df2 = pd.DataFrame([diseases,p]).T.sort_values(1,ascending=True)
    
plt.barh(df2[0], df2[1])
plt.title("Disease Probabilities")
plt.xlabel("Probability")

plt.xlim([0, 1])

rf = RandomForestClassifier(random_state=123)

rf.fit(X_train, y_train)
rf.score(X_test, y_test)

n_estimators_vals = [1, 10, 50, 100, 150, 200, 300]

score_best = 0
n_best = 0

scores = []

for n in n_estimators_vals:
    rf = RandomForestClassifier(random_state=123, n_estimators=n)
    rf.fit(X_train, y_train)
    s = rf.score(X_test, y_test)
    scores.append(s)
    if (s > score_best):
        score_best = s
        n_best = n
        
plt.plot(n_estimators_vals, scores)
plt.title("Accuracy vs. n_estimators")
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
        
print("Best accuracy was ", score_best, " with n_estimators value = ", n_best) 


knn = KNeighborsClassifier()

knn.fit(X_train, y_train)
knn.score(X_test, y_test)


n_neighbors_vals = [1, 2, 3, 4, 5, 6, 7]

score_best = 0
n_best = 0

scores = []

for n in n_neighbors_vals:
    rf = KNeighborsClassifier(n_neighbors=n)
    rf.fit(X_train, y_train)
    s = rf.score(X_test, y_test)
    scores.append(s)
    if (s > score_best):
        score_best = s
        n_best = n
        
plt.plot(n_neighbors_vals, scores)
plt.title("Accuracy vs. n_neighbors")
plt.xlabel("n_neighbors")
plt.ylabel("Accuracy")
        
print("Best accuracy was ", score_best, " with n_neighbors value = ", n_best) 


lr_f = LogisticRegression(max_iter = 300, C = 3.16)

X = df.drop(columns=["disease", "source"])
y = df["disease"]

lr_f.fit(X, y)

p_f = lr_f.predict_proba(ex)[0]

diseases_f = lr_f.classes_

df3 = pd.DataFrame([diseases_f,p_f]).T.sort_values(1,ascending=True)
    
plt.barh(df3[0], df3[1])
plt.title("Disease Probabilities with True Class = {0}".format(y_train.iloc[0]))
plt.xlabel("Probability")

plt.xlim([0, 1])

#exs = [X_train.iloc[[0]], X_train.iloc[[23]], X_train.iloc[[42]]]
idx = [2, 23, 42]

for i in idx:
    p = lr_f.predict_proba(X.iloc[[i]])[0]
    
    df = pd.DataFrame([lr_f.classes_,p]).T.sort_values(1,ascending=True)
    plt.barh(df[0], df[1])
    plt.title("Disease Probabilities with True Class = {0}".format(y.iloc[i]))
    plt.xlabel("Probability")

    plt.xlim([0, 1])
    plt.show()



def predict_disease(model):
    # Create an empty DataFrame with the same columns as your training data
    user_input = pd.DataFrame(columns=X_train.columns)

    # Ask the user to input values for each symptom
    for symptom in X_train.columns:
        value = input(f"Enter value for {symptom}: ")
        user_input.loc[0, symptom] = value

    # Use the model to predict the disease probabilities
    probabilities = model.predict_proba(user_input)[0]

    # Create a DataFrame to display the probabilities
    df = pd.DataFrame([model.classes_, probabilities]).T.sort_values(1, ascending=True)

    # Plot the probabilities
    plt.barh(df[0], df[1])
    plt.title("Disease Probabilities")
    plt.xlabel("Probability")
    plt.xlim([0, 1])
    plt.show()

# Use the function with your trained model
predict_disease(lr_f)


