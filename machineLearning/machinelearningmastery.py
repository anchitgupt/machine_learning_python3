import sklearn
from sklearn.datasets import load_breast_cancer

# to divide the given data into train and test data randomlly
from sklearn.model_selection import train_test_split

# importing the naive bayes library
from sklearn.naive_bayes import GaussianNB

# test the accuracy of the model
from sklearn.metrics import accuracy_score

# load dataset
data = load_breast_cancer()
''' 
    @data is type of dictinary data type here
    The data variable represents a Python object that works like a dictionary. 
    The important dictionary keys to consider are the classification label names (target_names), the actual labels (target), 
    the attribute/feature names (feature_names), and the attributes (data).
    Attributes are a critical part of any classifier. Attributes capture important characteristics about the nature of the data. Given the label we are trying to predict (malignant versus benign tumor),
    possible useful attributes include the size, radius, and texture of the tumor.
'''

# organize our data

label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

# splitting our data

train, test, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size=0.33,
                                                          random_state=42)
# Initialize our classifier
gnb = GaussianNB()

# train our classifier
model = gnb.fit(train, train_labels)

# make predictions
preds = gnb.predict(test)
print("Set is Represented by: ",end=" ")
print(preds)

# evaluate accuracy
print("Accuracy of the project is: ", end=" ")
print(accuracy_score(test_labels, preds))
