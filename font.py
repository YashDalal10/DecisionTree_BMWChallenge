import streamlit as st
import pandas as pd
from six import StringIO
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

# Load the iris dataset
df = pd.read_csv(r"C:\Users\heram\PycharmProjects\DecisionTree_BMWChallenge\result_cleaned.csv",
                 names=["col1", "processor", "batch_size", "latency", "throughput", "models", "precision"],
                 header=None, skiprows=1)
df1 = pd.read_csv(r"C:\Users\heram\PycharmProjects\DecisionTree_BMWChallenge\result_cleaned.csv",
                  names=["col1", "processor", "batch_size", "latency", "throughput", "models", "precision"],
                  header=None, skiprows=1)
df.drop(df.columns[0], axis=1, inplace=True)
le = LabelEncoder()
df[['processor', 'models', 'precision']] = \
    df[['processor', 'models', 'precision']].apply(le.fit_transform)
y = df[df.columns[0]]
X = df.drop([df.columns[0]], axis=1)

# Train a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Create a Streamlit app
st.title("Iris Dataset Decision Tree")

# Display the decision tree
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, rounded=True,
                feature_names=X.columns,
                class_names=df1.processor.unique())
graph = graphviz.Source(dot_data)
st.graphviz_chart(graph)
