import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report


df = pd.read_csv('C:/Users/Jenish Tamrakar/Desktop/DR/training_sample1.csv')
# print(df.tail())
# print(df.shape)
# print(df.size)
# print(df.count())
# print(df['Retinopathy grade'].value_counts())


# plot
nodr_df = df[df['Retinopathy grade'] == 0]
mild_df = df[df['Retinopathy grade'] == 1]
moderate_df = df[df['Retinopathy grade'] == 2]
severe_df = df[df['Retinopathy grade'] == 3]
vsevere_df = df[df['Retinopathy grade'] == 4]

axes = nodr_df.plot(kind='scatter', x='no_of_haemorrhages', y='no_of_microaneurysms', color='yellow', label='No Diabetic Retinopathy')
mild_df.plot(kind='scatter', x='no_of_haemorrhages', y='no_of_microaneurysms', color='black', label='Mild Diabetic Retinopathy', ax=axes)
moderate_df.plot(kind='scatter', x='no_of_haemorrhages', y='no_of_microaneurysms', color='green', label='Moderate Diabetic Retinopathy', ax=axes)
severe_df.plot(kind='scatter', x='no_of_haemorrhages', y='no_of_microaneurysms', color='blue', label='Severe Diabetic Retinopathy', ax=axes)
vsevere_df.plot(kind='scatter', x='no_of_haemorrhages', y='no_of_microaneurysms', color='red', label='Very Severe Diabetic Retinopathy', ax=axes)

# plt.scatter(axes)
# plt.show()
# print(df.dtypes)

# print(df.columns)
relevant_df = df[['density_of_blood_vessels', 'no_of_haemorrhages', 'no_of_microaneurysms']]
# print(relevant_df.columns)

X = np.asarray(relevant_df)                # multidimensional data
y = np.asarray(df['Retinopathy grade'])    # 1D data


# dividing into training set and testing set- 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=4)
# train, test = train_test_split(df, test_size=0.2)

# X_train (329 x 3)
# y_train (329 x 1)
# X_test (83 x 3)
# y_test (83 x 1)
# print(X_test.shape)

# modelling the SVM Classifier
classifier = svm.SVC(kernel='linear', gamma='auto', C=2)
# classifier = svm.SVC()

# training
classifier.fit(X_train, y_train)
# print(classifier.fit(X_train, y_train))

# testing
y_predicted = classifier.predict(X_test)

# get accuracy of testing
print("Accuracy:", accuracy_score(y_test, y_predicted))
# print("Precision:", precision_score(y_test, y_predicted))
# print("Recall:", recall_score(y_test, y_predicted))
print(classification_report(y_test, y_predicted))