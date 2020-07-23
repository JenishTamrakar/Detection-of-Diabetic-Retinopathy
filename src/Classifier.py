import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report

class Classifier:

    def training(self):
        df = pd.read_csv('C:/Users/Jenish Tamrakar/Desktop/DR/training_sample1.csv')

        # for plot
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
        # displaying plot
        # plt.scatter(axes)
        # plt.show()

        # selecting the relavant columns of the dataframe for classification
        relevant_df = df[['density_of_blood_vessels', 'no_of_haemorrhages', 'no_of_microaneurysms']]

        # converting the dataframe into numpy array for classification
        X = np.asarray(relevant_df)                # multidimensional data
        y = np.asarray(df['Retinopathy grade'])    # 1D data

        # dividing into training set and testing set- 80% for training and 20% for testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=4)
        # train, test = train_test_split(df, test_size=0.2)

        # the contents in the training set and testing set
        # X_train (329 x 3)
        # y_train (329 x 1)
        # X_test (83 x 3)
        # y_test (83 x 1)
        # print(X_test.shape)

        # modelling the SVM Classifier
        classifier = svm.SVC(kernel='linear', gamma='auto', C=2)

        # training
        classifier.fit(X_train, y_train)

        # getting data stored in csv for detection and prediction of DR from the image input from the user
        df1 = pd.read_csv('records.csv')
        relevant_df1 = df1[['density_of_blood_vessels', 'no_of_haemorrhages', 'no_of_microaneurysms']]
        X_predict = np.asarray(relevant_df1)
        y_pred = classifier.predict(X_predict)
        print("Retinopathy Grade=", y_pred)
        if y_pred == 0:
            status = "No"
            grade = 0
            severity = "Normal"
            print("No Diabetic Retinopathy")
            print("Normal")
        else:
            status = "Yes"
            print("Diabetic Retinopathy")
            if y_pred == 1:
                grade = 1
                severity = "Mild"
                print("Mild Diabetic Retinopathy")
            elif y_pred == 2:
                grade = 2
                severity = "Moderate"
                print("Moderate Diabetic Retinopathy")
            elif y_pred == 3:
                grade = 3
                severity = "Severe"
                print("Severe Diabetic Retinopathy")
            elif y_pred == 4:
                grade = 4
                severity = "Very Severe"
                print("Very Severe Diabetic Retinopathy")

        df3 = pd.read_csv('records.csv')
        df3["Diabetic_retinopathy_status"] = status
        df3["Diabetic_retinopathy_grade"] = grade
        df3["Severity"] = severity
        df3.to_csv("records.csv", index=False)
        # testing
        # y_predicted = classifier.predict(X_test)

        # appending results to the csv file
        df = pd.read_csv('records.csv')
        with open('data.csv', 'a') as f:
            df.to_csv(f, header=False)

        # get accuracy of testing
        # print("Accuracy:", accuracy_score(y_test, y_predicted))
        # print("Precision:", precision_score(y_test, y_predicted))
        # print("Recall:", recall_score(y_test, y_predicted))
        # print(classification_report(y_test, y_predicted))


    def main(self):
        cl.training()

if __name__ == "__main__":
    cl = Classifier()
    cl.main()