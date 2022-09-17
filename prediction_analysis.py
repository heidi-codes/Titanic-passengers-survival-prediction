import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier

# Data loading
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')
# Data exploration
print(train_data.info())
print('-'*30)
print(train_data.describe())
print('-'*30)
print(train_data.describe(include=['O']))
print('-'*30)
print(train_data.head())
print('-'*30)
print(train_data.tail())
# Data cleaning
# Fill the nan value with the average age
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)
# Fill the nan value with the average of the ticket prices
train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)
print(train_data['Embarked'].value_counts())

# Use the port with the most logins to fill in the nan value
train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S',inplace=True)

# Feature selection
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]

dvec=DictVectorizer(sparse=False)
train_features=dvec.fit_transform(train_features.to_dict(orient='record'))
print(dvec.feature_names_)

# Construct ID3 Decision Tree
clf = DecisionTreeClassifier(criterion='entropy')
# Decision tree training
clf.fit(train_features, train_labels)

test_features=dvec.transform(test_features.to_dict(orient='record'))
# Decision tree prediction
pred_labels = clf.predict(test_features)

# Get the accuracy of the decision tree
acc_decision_tree = round(clf.score(train_features, train_labels), 6)
print(u'score accuracy rate is %.4lf' % acc_decision_tree)
