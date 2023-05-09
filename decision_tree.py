import pickle

import pandas as pd
import pydotplus
from IPython.core.display import Image
from matplotlib import pyplot as plt
from six import StringIO
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz

if __name__ == '__main__':
    df = pd.read_csv(r"C:\Users\heram\PycharmProjects\DecisionTree_BMWChallenge\merged_data.csv",
                     names=["processor", "batch_size", "latency", "throughput", "models", "precision", "machine"],
                     header=None, skiprows=1)
    df1 = pd.read_csv(r"C:\Users\heram\PycharmProjects\DecisionTree_BMWChallenge\merged_data.csv",
                      names=["processor", "batch_size", "latency", "throughput", "models", "precision", "machine"],
                      header=None, skiprows=1)
    # df.drop(df.columns[0], axis=1, inplace=True)
    le = LabelEncoder()
    df[['processor', 'models', 'precision', 'machine']] = \
        df[['processor', 'models', 'precision', 'machine']].apply(le.fit_transform)

    dt = DecisionTreeClassifier()
    y = df[['processor', 'machine']]
    X = df.drop(['processor', 'machine'], axis=1)

    clf = dt.fit(X, y)

    fig = plt.figure(figsize=(50, 30))
    _ = tree.plot_tree(clf,
                       feature_names=X.columns,
                       class_names=y.columns,
                       filled=True)
    fig.tight_layout()
    plt.show()

    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data, filled=True, rounded=True,
                    feature_names=X.columns,
                    class_names=y.columns)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    Image(graph.create_png())
    graph.write_pdf("dt_telecom_churn.pdf")

    pickle.dump(clf, open(r'C:\Users\heram\PycharmProjects\DecisionTree_BMWChallenge\dt_model.pkl', 'wb'))
    print("Saved model to disk")

    export_graphviz(clf, out_file='tree.tree', filled=True, rounded=True, special_characters=True)
