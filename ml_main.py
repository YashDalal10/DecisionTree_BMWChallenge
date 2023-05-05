import os
import pickle
from contextlib import redirect_stdout

import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input, Model
from keras.api import keras
from keras.layers import Dense
from keras.utils import plot_model
from matplotlib import pyplot as plt
from sklearn import set_config
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, BaggingRegressor, VotingRegressor, \
    GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, r_regression, f_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import PassiveAggressiveRegressor, TweedieRegressor, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import RadiusNeighborsRegressor, KNeighborsRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression


########################################################################################################################
# Random Data Generator
########################################################################################################################

X,y = make_regression(n_samples=100, n_features=1, random_state=1)
df_data = pd.DataFrame(data = {'col1':X, 'col2':y}, columns = ['X','y'])
df_data.to_csv(r'./dataset/sample_data.csv', index=False)

########################################################################################################################
# Data Splitting
########################################################################################################################


def split_data(df_data):
    X = df_data.drop(['RUL'], axis=1)
    y = df_data['RUL']

    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.20, random_state=1)

    return X_train1, X_test1, y_train1, y_test1


########################################################################################################################
# Machine Learning Model and Results
########################################################################################################################

def save_model(hypermodel_in, model_name):
    try:
        os.makedirs(os.getcwd() + '/Models/' + str(model_name))

        path, _, _ = next(os.walk(os.getcwd() + '/Models/' + str(model_name)))

        pickle.dump(hypermodel_in, open(path + '/' + str(model_name) + '_model.pkl', 'wb'))
        print("Saved model to disk")

        pickled_model = pickle.load(open(path + '/' + str(model_name) + '_model.pkl', 'rb'))
    except OSError:
        path, _, _ = next(os.walk(os.getcwd() + '/Models/' + str(model_name)))

        pickle.dump(hypermodel_in, open(path + '/' + str(model_name) + '_model.pkl', 'wb'))
        print("Saved model to disk")

        pickled_model = pickle.load(open(path + '/' + str(model_name) + '_model.pkl', 'rb'))
    return pickled_model


def plot_data(y_test_model, predictions_model, model_name):
    path, _, _ = next(os.walk(os.getcwd() + '/Results/' + str(model_name)))

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(y_test_model.tolist(), color='green', label='Test Data')
    ax.plot(predictions_model.tolist(), color='red', label='Predicted Data')
    ax.legend(loc='upper left', framealpha=1, shadow=True)
    ax.set_xlabel('Sample Number')
    ax.set_ylabel('Remaining Useful Life (RUL)')
    ax.set_title('Model: ' + str(model_name))
    ax.grid(alpha=0.25)
    filename = str(model_name) + '_predictions_vs_true.png'
    plt.savefig(path + '/' + filename, dpi=150)
    plt.show()


def save_variables_model(y_test_model, predictions_model, acc, mse, mae, medae, rmse, mape, best_estimator,
                         model_name, pipeline):
    try:
        os.makedirs(os.getcwd() + '/Results/' + str(model_name))
        path, _, _ = next(os.walk(os.getcwd() + '/Results/' + str(model_name)))

        with open(path + '/' + 'pipeline.txt', 'w') as f:
            with redirect_stdout(f):
                print(pipeline)

        y_test_model = y_test_model.tolist()
        predictions_model = predictions_model.tolist()
        # input text
        y_test_str = repr(y_test_model[:5])
        predictions_str = repr(predictions_model[:5])
        acc_str = repr(acc)
        mse_str = repr(mse)
        mae_str = repr(mae)
        medae_str = repr(medae)
        rmse_str = repr(rmse)
        mape_str = repr(mape)
        est_str = repr(best_estimator)

        # open file
        filename = str(model_name) + '_performance.txt'
        file = open(path + '/' + filename, "w")

        # convert variable to string
        file.write('************************************************************************************************\n')
        file.write('Test Data (5 Values): ' + str(model_name) + ' = ' + y_test_str + "\n")
        file.write('************************************************************************************************\n')
        file.write('Predicted Data (5 Values): ' + str(model_name) + ' = ' + predictions_str + "\n")
        file.write('************************************************************************************************\n')
        file.write('Accuracy of Model: ' + str(model_name) + ' = ' + acc_str + "\n")
        file.write('************************************************************************************************\n')
        file.write('MSE of Model: ' + str(model_name) + ' = ' + mse_str + "\n")
        file.write('************************************************************************************************\n')
        file.write('MAE of Model: ' + str(model_name) + ' = ' + mae_str + "\n")
        file.write('************************************************************************************************\n')
        file.write('MEDAE of Model: ' + str(model_name) + ' = ' + medae_str + "\n")
        file.write('************************************************************************************************\n')
        file.write('RMSE of Model: ' + str(model_name) + ' = ' + rmse_str + "\n")
        file.write('************************************************************************************************\n')
        file.write('MAPE of Model: ' + str(model_name) + ' = ' + mape_str + "\n")
        file.write('************************************************************************************************\n')
        file.write('Best estimator of Model: ' + str(model_name) + ' = ' + est_str + "\n")
        file.write('************************************************************************************************\n')

        # close file
        file.close()
    except OSError:
        path, _, _ = next(os.walk(os.getcwd() + '/Results/' + str(model_name)))

        with open(path + '/' + 'pipeline.txt', 'w') as f:
            with redirect_stdout(f):
                print(pipeline)

        y_test_model = y_test_model.tolist()
        predictions_model = predictions_model.tolist()
        # input text
        y_test_str = repr(y_test_model[:5])
        predictions_str = repr(predictions_model[:5])
        acc_str = repr(acc)
        mse_str = repr(mse)
        mae_str = repr(mae)
        medae_str = repr(medae)
        rmse_str = repr(rmse)
        mape_str = repr(mape)
        est_str = repr(best_estimator)

        # open file
        filename = str(model_name) + '_performance.txt'
        file = open(path + '/' + filename, "w")

        # convert variable to string
        file.write('************************************************************************************************\n')
        file.write('Test Data (5 Values): ' + str(model_name) + ' = ' + y_test_str + "\n")
        file.write('************************************************************************************************\n')
        file.write('Predicted Data (5 Values): ' + str(model_name) + ' = ' + predictions_str + "\n")
        file.write('************************************************************************************************\n')
        file.write('Accuracy of Model: ' + str(model_name) + ' = ' + acc_str + "\n")
        file.write('************************************************************************************************\n')
        file.write('MSE of Model: ' + str(model_name) + ' = ' + mse_str + "\n")
        file.write('************************************************************************************************\n')
        file.write('MAE of Model: ' + str(model_name) + ' = ' + mae_str + "\n")
        file.write('************************************************************************************************\n')
        file.write('MEDAE of Model: ' + str(model_name) + ' = ' + medae_str + "\n")
        file.write('************************************************************************************************\n')
        file.write('RMSE of Model: ' + str(model_name) + ' = ' + rmse_str + "\n")
        file.write('************************************************************************************************\n')
        file.write('MAPE of Model: ' + str(model_name) + ' = ' + mape_str + "\n")
        file.write('************************************************************************************************\n')
        file.write('Best estimator of Model: ' + str(model_name) + ' = ' + est_str + "\n")
        file.write('************************************************************************************************\n')

        # close file
        file.close()


########################################################################################################################
# Artificial Neural Network Model and Results
########################################################################################################################

def save_model_ann(hypermodel_in, model_name):
    try:
        os.makedirs(os.getcwd() + '/Models/' + str(model_name))
        path, _, _ = next(os.walk(os.getcwd() + '/Models/' + str(model_name)))

        with open(path + '/model_summary_' + str(model_name) + '.txt', 'w') as f:
            with redirect_stdout(f):
                print(hypermodel_in.summary())

        plot_model(hypermodel_in, to_file=path + '/' + str(model_name) + '.png')

        # serialize model to JSON
        model_json = hypermodel_in.to_json()
        with open(path + '/' + str(model_name) + '.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        hypermodel_in.save_weights(path + '/' + str(model_name) + '.h5')
        print("Saved model to disk")

    except OSError:
        path, _, _ = next(os.walk(os.getcwd() + '/Models/' + str(model_name)))

        with open(path + '/model_summary_' + str(model_name) + '.txt', 'w') as f:
            with redirect_stdout(f):
                print(hypermodel_in.summary())

        plot_model(hypermodel_in, to_file=path + '/' + str(model_name) + '.png')

        # serialize model to JSON
        model_json = hypermodel_in.to_json()
        with open(path + '/' + str(model_name) + '.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        hypermodel_in.save_weights(path + '/' + str(model_name) + '.h5')
        print("Saved model to disk")


def plot_learning_curve_ann(history_model, epochs_model, model_train):
    try:
        os.makedirs(os.getcwd() + '/Results/' + str(model_train))
        path, _, _ = next(os.walk(os.getcwd() + '/Results/' + str(model_train)))

        # defining the range of the x-axis
        epoch_range = range(1, epochs_model + 1)

        # Plot training & validation accuracy values
        plt.plot(epoch_range, history_model.history['accuracy'])
        plt.plot(epoch_range, history_model.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        filename = 'accuracy_' + str(model_train) + '_' + str(epochs_model) + '_epoch.png'
        plt.savefig(path + '/' + filename, dpi=150)
        plt.close()

        # Plot training & validation loss values
        plt.plot(epoch_range, history_model.history['loss'])
        plt.plot(epoch_range, history_model.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        filename = 'loss_' + str(model_train) + '_' + str(epochs_model) + '_epoch.png'
        plt.savefig(path + '/' + filename, dpi=150)
        plt.close()

        # Plot training & validation mae values
        plt.plot(epoch_range, history_model.history['mae'])
        plt.plot(epoch_range, history_model.history['val_mae'])
        plt.title('Model mae')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        filename = 'mae_' + str(model_train) + '_' + str(epochs_model) + '_epoch.png'
        plt.savefig(path + '/' + filename, dpi=150)
        plt.close()
    except OSError:
        path, _, _ = next(os.walk(os.getcwd() + '/Results/' + str(model_train)))

        # defining the range of the x-axis
        epoch_range = range(1, epochs_model + 1)

        # Plot training & validation accuracy values
        plt.plot(epoch_range, history_model.history['accuracy'])
        plt.plot(epoch_range, history_model.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        filename = 'accuracy_' + str(model_train) + '_' + str(epochs_model) + '_epoch.png'
        plt.savefig(path + '/' + filename, dpi=150)
        plt.close()

        # Plot training & validation loss values
        plt.plot(epoch_range, history_model.history['loss'])
        plt.plot(epoch_range, history_model.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        filename = 'loss_' + str(model_train) + '_' + str(epochs_model) + '_epoch.png'
        plt.savefig(path + '/' + filename, dpi=150)
        plt.close()

        # Plot training & validation mae values
        plt.plot(epoch_range, history_model.history['mae'])
        plt.plot(epoch_range, history_model.history['val_mae'])
        plt.title('Model mae')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        filename = 'mae_' + str(model_train) + '_' + str(epochs_model) + '_epoch.png'
        plt.savefig(path + '/' + filename, dpi=150)
        plt.close()


def plot_data_ann(y_test_model, predictions_model, model_train):
    path, _, _ = next(os.walk(os.getcwd() + '/Results/' + str(model_train)))

    fig, ax = plt.subplots()
    ax.plot(y_test_model.tolist(), color='green', label='Test Data')
    ax.plot(predictions_model.tolist(), color='red', label='Predicted Data')
    ax.legend(loc='upper left')
    ax.set_title('Model: ' + str(model_train))
    filename = str(model_train) + '_predictions_vs_true.png'
    plt.savefig(path + '/' + filename, dpi=150)
    plt.close()


def save_variables_model_ann(y_test_model, predictions_model, acc, mse, mae, model_train):
    path, _, _ = next(os.walk(os.getcwd() + '/Results/' + str(model_train)))

    y_test_model = y_test_model.tolist()
    predictions_model = predictions_model.tolist()
    # input text
    y_test_str = repr(y_test_model[0:5])
    predictions_str = repr(predictions_model[:5])
    acc_str = repr(acc)
    mse_str = repr(mse)
    mae_str = repr(mae)

    # open file
    filename = str(model_train) + '_performance.txt'
    file = open(path + '/' + filename, "w")

    # convert variable to string
    file.write('****************************************************************************************************\n')
    file.write('Test Data (5 Values): ' + str(model_train) + ' = ' + y_test_str + "\n")
    file.write('****************************************************************************************************\n')
    file.write('Predicted Data (5 Values): ' + str(model_train) + ' = ' + predictions_str + "\n")
    file.write('****************************************************************************************************\n')
    file.write('Accuracy of Model: ' + str(model_train) + ' = ' + acc_str + "\n")
    file.write('****************************************************************************************************\n')
    file.write('MSE of Model: ' + str(model_train) + ' = ' + mse_str + "\n")
    file.write('****************************************************************************************************\n')
    file.write('MAE of Model: ' + str(model_train) + ' = ' + mae_str + "\n")
    file.write('****************************************************************************************************\n')

    # close file
    file.close()


########################################################################################################################
# Performance Metrics
########################################################################################################################

def mean_absolute_error(y_test_model, predictions_model):
    mae = np.mean(abs(predictions_model - y_test_model))
    return mae


def median_absolute_error(y_test_model, predictions_model):
    medae = np.median(abs(predictions_model - y_test_model))
    return medae


def mean_squared_error(y_test_model, predictions_model):
    mse = np.square(np.subtract(predictions_model, y_test_model)).mean()
    return mse


def root_mean_squared_error(y_test_model, predictions_model):
    rmse = np.sqrt(np.mean(np.square(predictions_model - y_test_model)))
    return rmse


def mean_absolute_percentage_error(y_test_model, predictions_model):
    mape = np.mean(np.abs((y_test_model - predictions_model) / y_test_model)) * 100
    return mape


########################################################################################################################
# Decision Tree Regressor
########################################################################################################################


def pipe_sc_decision_tree_regressor(train_data, test_data):
    parameters = {'dt__criterion': ['squared_error', 'friedman_mse'],
                  'dt__splitter': ['best', 'random'],
                  'dt__min_samples_split': [2, 3, 4, 5]}

    dt = DecisionTreeRegressor()
    scaler = StandardScaler()

    pipe = Pipeline(steps=[("scaler", scaler), ("dt", dt)])
    set_config(display="text")
    reg = GridSearchCV(pipe, parameters, cv=10, scoring="neg_mean_absolute_error", verbose=6, n_jobs=-5)
    reg.fit(train_data[0], train_data[1])

    predictions = reg.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    best_estimator = reg.best_estimator_

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=best_estimator, model_name=pipe_sc_decision_tree_regressor.__name__)

    hypermodel = save_model(reg, model_name=pipe_sc_decision_tree_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_decision_tree_regressor.__name__)


def pipe_sc_fs_decision_tree_regressor(train_data, test_data):
    parameters = {'selection__k': [7, 10, 15, 20, 25, 28, 29, 30, 31, 32, 33, 34]}

    dt = DecisionTreeRegressor(criterion='friedman_mse', min_samples_split=4)
    selection = SelectKBest(score_func=f_regression)
    scaler = StandardScaler()

    pipe = Pipeline(steps=[("scaler", scaler), ("selection", selection), ("dt", dt)])
    set_config(display="text")
    reg = GridSearchCV(pipe, parameters, cv=10, scoring="neg_mean_absolute_error", verbose=4, n_jobs=-2)

    reg.fit(train_data[0], train_data[1])

    predictions = reg.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    best_estimator = reg.best_estimator_

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=best_estimator, model_name=pipe_sc_fs_decision_tree_regressor.__name__)

    hypermodel = save_model(reg, model_name=pipe_sc_fs_decision_tree_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_fs_decision_tree_regressor.__name__)


def pipe_sc_pca_decision_tree_regressor(train_data, test_data):
    parameters = {'dt__criterion': ['squared_error', 'friedman_mse'],
                  'dt__splitter': ['best', 'random'],
                  'dt__max_features': ['sqrt', None, 0.2, 0.3, 0.4, 0.5],
                  'dt__min_samples_split': [2, 3, 4, 5],
                  'pca__n_components': [4, 5, 6, 7, 8, 9, 10],
                  'pca__svd_solver': ['auto', 'full', 'arpack', 'randomized']}

    dt = DecisionTreeRegressor()
    pca = PCA()
    scaler = StandardScaler()

    pipe = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("dt", dt)])
    set_config(display="text")
    reg = GridSearchCV(pipe, parameters, cv=10, scoring="neg_mean_absolute_error", verbose=4, n_jobs=-1)
    reg.fit(train_data[0], train_data[1])

    predictions = reg.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    best_estimator = reg.best_estimator_

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=best_estimator, model_name=pipe_sc_pca_decision_tree_regressor.__name__)

    hypermodel = save_model(reg, model_name=pipe_sc_pca_decision_tree_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_pca_decision_tree_regressor.__name__)


def pipe_sc_pca_fs_decision_tree_regressor(train_data, test_data):
    parameters = {'dt__min_samples_split': [2, 3],
                  'features__pca__n_components': [35, 40],
                  'features__pca__svd_solver': ['auto', 'full', 'arpack', 'randomized'],
                  'features__selection__k': [42, 43]}

    dt = DecisionTreeRegressor(criterion='friedman_mse')
    pca = PCA()
    selection = SelectKBest(score_func=f_regression)
    scaler = StandardScaler()

    combined_features = FeatureUnion([("pca", pca), ("selection", selection)])

    pipe = Pipeline(steps=[("scaler", scaler), ("features", combined_features), ("dt", dt)])
    set_config(display="text")
    reg = GridSearchCV(pipe, parameters, cv=10, scoring="neg_mean_absolute_error", verbose=4, n_jobs=-1)
    reg.fit(train_data[0], train_data[1])

    predictions = reg.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    best_estimator = reg.best_estimator_

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=best_estimator, model_name=pipe_sc_pca_fs_decision_tree_regressor.__name__)

    hypermodel = save_model(reg, model_name=pipe_sc_pca_fs_decision_tree_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_pca_fs_decision_tree_regressor.__name__)


def pipe_sc_tsvd_decision_tree_regressor(train_data, test_data):
    parameters = {'dt__criterion': ['squared_error', 'friedman_mse'],
                  'dt__splitter': ['best', 'random'],
                  'dt__max_features': ['sqrt', None, 0.2, 0.3, 0.4, 0.5],
                  'dt__min_samples_split': [2, 3, 4, 5],
                  'tsvd__n_components': [4, 5, 6, 7, 8, 9, 10],
                  'tsvd__algorithm': ['arpack', 'randomized']}

    dt = DecisionTreeRegressor()
    tsvd = TruncatedSVD()
    scaler = StandardScaler()

    pipe = Pipeline(steps=[("scaler", scaler), ("tsvd", tsvd), ("dt", dt)])
    set_config(display="text")
    reg = GridSearchCV(pipe, parameters, cv=10, scoring="neg_mean_absolute_error", verbose=4, n_jobs=-1)
    reg.fit(train_data[0], train_data[1])

    predictions = reg.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    best_estimator = reg.best_estimator_

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=best_estimator, model_name=pipe_sc_pca_decision_tree_regressor.__name__)

    hypermodel = save_model(reg, model_name=pipe_sc_pca_decision_tree_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_pca_decision_tree_regressor.__name__)


def pipe_sc_tsvd_fs_decision_tree_regressor(train_data, test_data):
    parameters = {'dt__min_samples_split': [2, 3],
                  'features__tsvd__n_components': [35, 40],
                  'features__tsvd__algorithm': ['arpack', 'randomized'],
                  'features__selection__k': [41, 42, 43]}

    dt = DecisionTreeRegressor(criterion='friedman_mse')
    tsvd = TruncatedSVD()
    selection = SelectKBest(score_func=f_regression)
    scaler = StandardScaler()

    combined_features = FeatureUnion([("tsvd", tsvd), ("selection", selection)])

    pipe = Pipeline(steps=[("scaler", scaler), ("features", combined_features), ("dt", dt)])
    set_config(display="text")
    reg = GridSearchCV(pipe, parameters, cv=10, scoring="neg_mean_absolute_error", verbose=4, n_jobs=-1)
    reg.fit(train_data[0], train_data[1])

    predictions = reg.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    best_estimator = reg.best_estimator_

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=best_estimator, model_name=pipe_sc_tsvd_fs_decision_tree_regressor.__name__)

    hypermodel = save_model(reg, model_name=pipe_sc_tsvd_fs_decision_tree_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_tsvd_fs_decision_tree_regressor.__name__)


########################################################################################################################
# Random Forest Regressor
########################################################################################################################

def pipe_sc_rf_regressor(train_data, test_data):
    parameters = {'rf__criterion': ['squared_error', 'absolute_error', 'friedman_mse']}

    rf = RandomForestRegressor(max_features=None, n_estimators=350)
    scaler = StandardScaler()

    pipe = Pipeline(steps=[("scaler", scaler), ("rf", rf)])
    set_config(display="text")
    reg = GridSearchCV(pipe, parameters, cv=10, scoring="neg_mean_absolute_error", verbose=4, n_jobs=-1)
    reg.fit(train_data[0], train_data[1])

    predictions = reg.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    best_estimator = reg.best_estimator_

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=best_estimator, model_name=pipe_sc_rf_regressor.__name__)

    hypermodel = save_model(reg, model_name=pipe_sc_rf_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_rf_regressor.__name__)


def pipe_sc_fs_rf_regressor(train_data, test_data):
    parameters = {'rf__criterion': ['squared_error', 'absolute_error', 'friedman_mse'],
                  'rf__max_features': ['sqrt', None, 0.4, 0.5],
                  'rf__min_samples_split': [2, 3, 4, 5],
                  'rf__min_samples_leaf': [1, 0.4, 0.5, 0.6],
                  'selection__k': [2, 3, 4, 5, 6],
                  'selection__score_func': [r_regression, f_regression]}

    rf = RandomForestRegressor()
    selection = SelectKBest()
    scaler = StandardScaler()

    pipe = Pipeline(steps=[("scaler", scaler), ("selection", selection), ("rf", rf)])
    set_config(display="text")
    reg = GridSearchCV(pipe, parameters, cv=10, scoring="neg_mean_absolute_error", verbose=4, n_jobs=-1)
    reg.fit(train_data[0], train_data[1])

    predictions = reg.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    best_estimator = reg.best_estimator_

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=best_estimator, model_name=pipe_sc_fs_rf_regressor.__name__)

    hypermodel = save_model(reg, model_name=pipe_sc_fs_rf_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_fs_rf_regressor.__name__)


def pipe_sc_pca_rf_regressor(train_data, test_data):
    parameters = {'rf__criterion': ['squared_error', 'absolute_error', 'friedman_mse'],
                  'rf__max_features': ['sqrt', None, 0.4, 0.5],
                  'rf__min_samples_split': [2, 3, 4, 5],
                  'rf__min_samples_leaf': [1, 0.4, 0.5, 0.6],
                  'pca__n_components': [4, 5, 6, 7, 8, 9, 10],
                  'pca__svd_solver': ['auto', 'full', 'arpack', 'randomized']}

    rf = RandomForestRegressor()
    pca = PCA()
    scaler = StandardScaler()

    pipe = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("rf", rf)])
    set_config(display="text")
    reg = GridSearchCV(pipe, parameters, cv=10, scoring="neg_mean_absolute_error", verbose=4, n_jobs=-1)
    reg.fit(train_data[0], train_data[1])

    predictions = reg.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    best_estimator = reg.best_estimator_

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=best_estimator, model_name=pipe_sc_pca_rf_regressor.__name__)

    hypermodel = save_model(reg, model_name=pipe_sc_pca_rf_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_pca_rf_regressor.__name__)


def pipe_sc_pca_fs_rf_regressor(train_data, test_data):
    parameters = {'rf__criterion': ['squared_error', 'absolute_error', 'friedman_mse'],
                  'rf__max_features': ['sqrt', None, 0.4, 0.5],
                  'rf__min_samples_split': [2, 3, 4, 5],
                  'rf__min_samples_leaf': [1, 0.4, 0.5, 0.6],
                  'features__pca__n_components': [4, 5, 6, 7, 8, 9, 10],
                  'features__pca__svd_solver': ['auto', 'full', 'arpack', 'randomized'],
                  'features__selection__k': [2, 3, 4, 5, 6],
                  'features__selection__score_func': [r_regression, f_regression]}

    rf = RandomForestRegressor()
    pca = PCA()
    selection = SelectKBest()
    scaler = StandardScaler()

    combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

    pipe = Pipeline(steps=[("scaler", scaler), ("features", combined_features), ("rf", rf)])
    set_config(display="text")
    reg = GridSearchCV(pipe, parameters, cv=10, scoring="neg_mean_absolute_error", verbose=4, n_jobs=-1)
    reg.fit(train_data[0], train_data[1])

    predictions = reg.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    best_estimator = reg.best_estimator_

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=best_estimator, model_name=pipe_sc_pca_fs_rf_regressor.__name__)

    hypermodel = save_model(reg, model_name=pipe_sc_pca_fs_rf_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_pca_fs_rf_regressor.__name__)


def pipe_sc_tsvd_rf_regressor(train_data, test_data):
    parameters = {'rf__criterion': ['squared_error', 'absolute_error', 'friedman_mse'],
                  'rf__max_features': ['sqrt', None, 0.4, 0.5],
                  'rf__min_samples_split': [2, 3, 4, 5],
                  'rf__min_samples_leaf': [1, 0.4, 0.5, 0.6],
                  'tsvd__n_components': [4, 5, 6, 7, 8, 9, 10],
                  'tsvd__algorithm': ['arpack', 'randomized']}

    rf = RandomForestRegressor()
    tsvd = TruncatedSVD()
    scaler = StandardScaler()

    pipe = Pipeline(steps=[("scaler", scaler), ("tsvd", tsvd), ("rf", rf)])
    set_config(display="text")
    reg = GridSearchCV(pipe, parameters, cv=10, scoring="neg_mean_absolute_error", verbose=4, n_jobs=-1)
    reg.fit(train_data[0], train_data[1])

    predictions = reg.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    best_estimator = reg.best_estimator_

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=best_estimator, model_name=pipe_sc_tsvd_rf_regressor.__name__)

    hypermodel = save_model(reg, model_name=pipe_sc_tsvd_rf_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_tsvd_rf_regressor.__name__)


def pipe_sc_tsvd_fs_rf_regressor(train_data, test_data):
    parameters = {'rf__criterion': ['squared_error', 'absolute_error', 'friedman_mse'],
                  'rf__max_features': ['sqrt', None, 0.4, 0.5],
                  'rf__min_samples_split': [2, 3, 4, 5],
                  'rf__min_samples_leaf': [1, 0.4, 0.5, 0.6],
                  'features__tsvd__n_components': [4, 5, 6, 7, 8, 9, 10],
                  'features__tsvd__algorithm': ['arpack', 'randomized'],
                  'features__selection__k': [2, 3, 4, 5, 6],
                  'features__selection__score_func': [r_regression, f_regression]}

    rf = RandomForestRegressor()
    tsvd = TruncatedSVD()
    selection = SelectKBest()
    scaler = StandardScaler()

    combined_features = FeatureUnion([("tsvd", tsvd), ("univ_select", selection)])

    pipe = Pipeline(steps=[("scaler", scaler), ("features", combined_features), ("rf", rf)])
    set_config(display="text")
    reg = GridSearchCV(pipe, parameters, cv=10, scoring="neg_mean_absolute_error", verbose=4, n_jobs=-1)
    reg.fit(train_data[0], train_data[1])

    predictions = reg.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    best_estimator = reg.best_estimator_

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=best_estimator, model_name=pipe_sc_tsvd_fs_rf_regressor.__name__)

    hypermodel = save_model(reg, model_name=pipe_sc_tsvd_fs_rf_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_tsvd_fs_rf_regressor.__name__)


########################################################################################################################
# Histogram Gradient Boosting Regressor
########################################################################################################################

def pipe_sc_hgbr_regressor(train_data, test_data):
    parameters = {'hgbr__learning_rate': [1e-2, 1e-3],
                  'hgbr__max_iter': [140, 150],
                  'hgbr__min_samples_leaf': [30, 40]}

    hgbr = HistGradientBoostingRegressor(max_leaf_nodes=None)
    scaler = StandardScaler()

    pipe = Pipeline(steps=[("scaler", scaler), ("hgbr", hgbr)])
    set_config(display="text")
    reg = GridSearchCV(pipe, parameters, cv=10, scoring="neg_mean_absolute_error", verbose=4, n_jobs=-1)
    reg.fit(train_data[0], train_data[1])

    predictions = reg.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    best_estimator = reg.best_estimator_

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=best_estimator, model_name=pipe_sc_hgbr_regressor.__name__)

    hypermodel = save_model(reg, model_name=pipe_sc_hgbr_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_hgbr_regressor.__name__)


def pipe_sc_fs_hgbr_regressor(train_data, test_data):
    parameters = {'hgbr__loss': ['squared_error', 'absolute_error'],
                  'hgbr__learning_rate': [1e-1, 1e-2, 1e-3],
                  'hgbr__max_iter': [100, 200, 300],
                  'hgbr__max_leaf_nodes': [31, None],
                  'hgbr__min_samples_leaf': [20, 30, 40],
                  'selection__k': [2, 3, 4, 5, 6],
                  'selection__score_func': [r_regression, f_regression]}

    rf = RandomForestRegressor()
    selection = SelectKBest()
    scaler = StandardScaler()

    pipe = Pipeline(steps=[("scaler", scaler), ("selection", selection), ("rf", rf)])
    set_config(display="text")
    reg = GridSearchCV(pipe, parameters, cv=10, scoring="neg_mean_absolute_error", verbose=4, n_jobs=-1)
    reg.fit(train_data[0], train_data[1])

    predictions = reg.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    best_estimator = reg.best_estimator_

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=best_estimator, model_name=pipe_sc_fs_hgbr_regressor.__name__)

    hypermodel = save_model(reg, model_name=pipe_sc_fs_hgbr_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_fs_hgbr_regressor.__name__)


def pipe_sc_pca_hgbr_regressor(train_data, test_data):
    parameters = {'hgbr__loss': ['squared_error', 'absolute_error'],
                  'hgbr__learning_rate': [1e-1, 1e-2, 1e-3],
                  'hgbr__max_iter': [100, 200, 300],
                  'hgbr__max_leaf_nodes': [31, None],
                  'hgbr__min_samples_leaf': [20, 30, 40],
                  'pca__n_components': [4, 5, 6, 7, 8, 9, 10],
                  'pca__svd_solver': ['auto', 'full', 'arpack', 'randomized']}

    rf = RandomForestRegressor()
    pca = PCA()
    scaler = StandardScaler()

    pipe = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("rf", rf)])
    set_config(display="text")
    reg = GridSearchCV(pipe, parameters, cv=10, scoring="neg_mean_absolute_error", verbose=4, n_jobs=-1)
    reg.fit(train_data[0], train_data[1])

    predictions = reg.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    best_estimator = reg.best_estimator_

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=best_estimator, model_name=pipe_sc_pca_hgbr_regressor.__name__)

    hypermodel = save_model(reg, model_name=pipe_sc_pca_hgbr_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_pca_hgbr_regressor.__name__)


def pipe_sc_pca_fs_hgbr_regressor(train_data, test_data):
    parameters = {'hgbr__loss': ['squared_error', 'absolute_error'],
                  'hgbr__learning_rate': [1e-1, 1e-2, 1e-3],
                  'hgbr__max_iter': [100, 200, 300],
                  'hgbr__max_leaf_nodes': [31, None],
                  'hgbr__min_samples_leaf': [20, 30, 40],
                  'features__pca__n_components': [4, 5, 6, 7, 8, 9, 10],
                  'features__pca__svd_solver': ['auto', 'full', 'arpack', 'randomized'],
                  'features__selection__k': [2, 3, 4, 5, 6],
                  'features__selection__score_func': [r_regression, f_regression]}

    rf = RandomForestRegressor()
    pca = PCA()
    selection = SelectKBest()
    scaler = StandardScaler()

    combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

    pipe = Pipeline(steps=[("scaler", scaler), ("features", combined_features), ("rf", rf)])
    set_config(display="text")
    reg = GridSearchCV(pipe, parameters, cv=10, scoring="neg_mean_absolute_error", verbose=4, n_jobs=-1)
    reg.fit(train_data[0], train_data[1])

    predictions = reg.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    best_estimator = reg.best_estimator_

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=best_estimator, model_name=pipe_sc_pca_fs_hgbr_regressor.__name__)

    hypermodel = save_model(reg, model_name=pipe_sc_pca_fs_hgbr_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_pca_fs_hgbr_regressor.__name__)


def pipe_sc_tsvd_hgbr_regressor(train_data, test_data):
    parameters = {'hgbr__loss': ['squared_error', 'absolute_error'],
                  'hgbr__learning_rate': [1e-1, 1e-2, 1e-3],
                  'hgbr__max_iter': [100, 200, 300],
                  'hgbr__max_leaf_nodes': [31, None],
                  'hgbr__min_samples_leaf': [20, 30, 40],
                  'tsvd__n_components': [4, 5, 6, 7, 8, 9, 10],
                  'tsvd__algorithm': ['arpack', 'randomized']}

    rf = RandomForestRegressor()
    tsvd = TruncatedSVD()
    scaler = StandardScaler()

    pipe = Pipeline(steps=[("scaler", scaler), ("tsvd", tsvd), ("rf", rf)])
    set_config(display="text")
    reg = GridSearchCV(pipe, parameters, cv=10, scoring="neg_mean_absolute_error", verbose=4, n_jobs=-1)
    reg.fit(train_data[0], train_data[1])

    predictions = reg.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    best_estimator = reg.best_estimator_

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=best_estimator, model_name=pipe_sc_tsvd_hgbr_regressor.__name__)

    hypermodel = save_model(reg, model_name=pipe_sc_tsvd_hgbr_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_tsvd_hgbr_regressor.__name__)


def pipe_sc_tsvd_fs_hgbr_regressor(train_data, test_data):
    parameters = {'hgbr__loss': ['squared_error', 'absolute_error'],
                  'hgbr__learning_rate': [1e-1, 1e-2, 1e-3],
                  'hgbr__max_iter': [100, 200, 300],
                  'hgbr__max_leaf_nodes': [31, None],
                  'hgbr__min_samples_leaf': [20, 30, 40],
                  'features__tsvd__n_components': [4, 5, 6, 7, 8, 9, 10],
                  'features__tsvd__algorithm': ['arpack', 'randomized'],
                  'features__selection__k': [2, 3, 4, 5, 6],
                  'features__selection__score_func': [r_regression, f_regression]}

    rf = RandomForestRegressor()
    tsvd = TruncatedSVD()
    selection = SelectKBest()
    scaler = StandardScaler()

    combined_features = FeatureUnion([("tsvd", tsvd), ("univ_select", selection)])

    pipe = Pipeline(steps=[("scaler", scaler), ("features", combined_features), ("rf", rf)])
    set_config(display="text")
    reg = GridSearchCV(pipe, parameters, cv=10, scoring="neg_mean_absolute_error", verbose=4, n_jobs=-1)
    reg.fit(train_data[0], train_data[1])

    predictions = reg.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    best_estimator = reg.best_estimator_

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=best_estimator, model_name=pipe_sc_tsvd_fs_hgbr_regressor.__name__)

    hypermodel = save_model(reg, model_name=pipe_sc_tsvd_fs_hgbr_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_tsvd_fs_hgbr_regressor.__name__)


########################################################################################################################
# Bagging Regressor
########################################################################################################################


def pipe_sc_bagging_regressor(train_data, test_data):
    parameters = {'br__n_estimators': [10, 20, 30, 40],
                  'br__max_features': [40, 41, 42, 43]}

    br = BaggingRegressor()
    scaler = StandardScaler()

    pipe = Pipeline(steps=[("scaler", scaler), ("br", br)])
    set_config(display="text")
    reg = GridSearchCV(pipe, parameters, cv=10, scoring="neg_mean_absolute_error", verbose=4, n_jobs=-1)
    reg.fit(train_data[0], train_data[1])

    predictions = reg.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    best_estimator = reg.best_estimator_

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=best_estimator, model_name=pipe_sc_bagging_regressor.__name__)

    hypermodel = save_model(reg, model_name=pipe_sc_bagging_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_bagging_regressor.__name__)


def pipe_sc_fs_bagging_regressor(train_data, test_data):
    parameters = {'selection__k': [39, 40, 41, 42]}

    br = BaggingRegressor(n_estimators=40)
    selection = SelectKBest(score_func=f_regression)
    scaler = StandardScaler()

    pipe = Pipeline(steps=[("scaler", scaler), ("selection", selection), ("br", br)])
    set_config(display="text")
    reg = GridSearchCV(pipe, parameters, cv=10, scoring="neg_mean_absolute_error", verbose=4, n_jobs=-1)

    reg.fit(train_data[0], train_data[1])

    predictions = reg.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    # best_estimator = reg.best_estimator_

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=None, model_name=pipe_sc_fs_bagging_regressor.__name__)

    hypermodel = save_model(reg, model_name=pipe_sc_fs_bagging_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_fs_bagging_regressor.__name__)


def pipe_sc_pca_bagging_regressor(train_data, test_data):
    parameters = {'br__n_estimators': [10, 20, 30, 40],
                  'br__max_features': [40, 41, 42, 43],
                  'pca__n_components': [4, 5, 6, 7, 8, 9, 10],
                  'pca__svd_solver': ['auto', 'full', 'arpack', 'randomized']}

    br = BaggingRegressor()
    pca = PCA()
    scaler = StandardScaler()

    pipe = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("br", br)])
    set_config(display="text")
    reg = GridSearchCV(pipe, parameters, cv=10, scoring="neg_mean_absolute_error", verbose=4, n_jobs=-1)
    reg.fit(train_data[0], train_data[1])

    predictions = reg.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    best_estimator = reg.best_estimator_

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=best_estimator, model_name=pipe_sc_pca_bagging_regressor.__name__)

    hypermodel = save_model(reg, model_name=pipe_sc_pca_bagging_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_pca_bagging_regressor.__name__)


def pipe_sc_pca_fs_bagging_regressor(train_data, test_data):
    parameters = {'br__n_estimators': [10, 20, 30, 40],
                  'br__max_features': [40, 41, 42, 43],
                  'features__pca__n_components': [4, 5, 6, 7, 8, 9, 10],
                  'features__pca__svd_solver': ['auto', 'full', 'arpack', 'randomized'],
                  'features__selection__k': [2, 3, 4, 5, 6],
                  'features__selection__score_func': [r_regression, f_regression]}

    br = BaggingRegressor()
    pca = PCA()
    selection = SelectKBest()
    scaler = StandardScaler()

    combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

    pipe = Pipeline(steps=[("scaler", scaler), ("features", combined_features), ("br", br)])
    set_config(display="text")
    reg = GridSearchCV(pipe, parameters, cv=10, scoring="neg_mean_absolute_error", verbose=4, n_jobs=-1)
    reg.fit(train_data[0], train_data[1])

    predictions = reg.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    best_estimator = reg.best_estimator_

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=best_estimator, model_name=pipe_sc_pca_fs_bagging_regressor.__name__)

    hypermodel = save_model(reg, model_name=pipe_sc_pca_fs_bagging_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_pca_fs_bagging_regressor.__name__)


def pipe_sc_tsvd_bagging_regressor(train_data, test_data):
    parameters = {'br__n_estimators': [10, 20, 30, 40],
                  'br__max_features': [40, 41, 42, 43],
                  'tsvd__n_components': [4, 5, 6, 7, 8, 9, 10],
                  'tsvd__algorithm': ['arpack', 'randomized']}

    br = BaggingRegressor()
    tsvd = TruncatedSVD()
    scaler = StandardScaler()

    pipe = Pipeline(steps=[("scaler", scaler), ("tsvd", tsvd), ("br", br)])
    set_config(display="text")
    reg = GridSearchCV(pipe, parameters, cv=10, scoring="neg_mean_absolute_error", verbose=4, n_jobs=-1)
    reg.fit(train_data[0], train_data[1])

    predictions = reg.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    best_estimator = reg.best_estimator_

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=best_estimator, model_name=pipe_sc_tsvd_bagging_regressor.__name__)

    hypermodel = save_model(reg, model_name=pipe_sc_tsvd_bagging_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_tsvd_bagging_regressor.__name__)


def pipe_sc_tsvd_fs_bagging_regressor(train_data, test_data):
    parameters = {'br__n_estimators': [10, 20, 30, 40],
                  'br__max_features': [40, 41, 42, 43],
                  'features__tsvd__n_components': [4, 5, 6, 7, 8, 9, 10],
                  'features__tsvd__algorithm': ['arpack', 'randomized'],
                  'features__selection__k': [2, 3, 4, 5, 6],
                  'features__selection__score_func': [r_regression, f_regression]}

    br = BaggingRegressor()
    tsvd = TruncatedSVD()
    selection = SelectKBest()
    scaler = StandardScaler()

    combined_features = FeatureUnion([("tsvd", tsvd), ("univ_select", selection)])

    pipe = Pipeline(steps=[("scaler", scaler), ("features", combined_features), ("br", br)])
    set_config(display="text")
    reg = GridSearchCV(pipe, parameters, cv=10, scoring="neg_mean_absolute_error", verbose=4, n_jobs=-1)
    reg.fit(train_data[0], train_data[1])

    predictions = reg.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    best_estimator = reg.best_estimator_

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=best_estimator, model_name=pipe_sc_tsvd_fs_bagging_regressor.__name__)

    hypermodel = save_model(reg, model_name=pipe_sc_tsvd_fs_bagging_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_tsvd_fs_bagging_regressor.__name__)


########################################################################################################################
# Neighbor Regressors
########################################################################################################################

def pipe_sc_radius_neighbor_regressor(train_data, test_data):
    parameters = {'rnr__radius': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                  'rnr__weights': ['uniform', 'distance', None],
                  'rnr__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                  'rnr__leaf_size': [30, 40, 50],
                  'rnr__p': [1, 2, 3]}

    rnr = RadiusNeighborsRegressor()
    scaler = StandardScaler()

    pipe = Pipeline(steps=[("scaler", scaler), ("rnr", rnr)])
    set_config(display="text")
    reg = GridSearchCV(pipe, parameters, cv=10, scoring="neg_mean_absolute_error", verbose=4, n_jobs=-1)
    reg.fit(train_data[0], train_data[1])

    predictions = reg.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    best_estimator = reg.best_estimator_

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=best_estimator, model_name=pipe_sc_radius_neighbor_regressor.__name__)

    hypermodel = save_model(reg, model_name=pipe_sc_radius_neighbor_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_radius_neighbor_regressor.__name__)


def pipe_sc_k_neighbor_regressor(train_data, test_data):
    parameters = {'knr__n_neighbors': [5, 6, 7, 8],
                  'knr__weights': ['uniform', 'distance', None],
                  'knr__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                  'knr__leaf_size': [30, 40, 50],
                  'knr__p': [1, 2, 3]}

    knr = KNeighborsRegressor(n_jobs=-1)
    scaler = StandardScaler()

    pipe = Pipeline(steps=[("scaler", scaler), ("knr", knr)])
    set_config(display="text")
    reg = GridSearchCV(pipe, parameters, cv=10, scoring="neg_mean_absolute_error", verbose=4, n_jobs=-1)
    reg.fit(train_data[0], train_data[1])

    predictions = reg.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    best_estimator = reg.best_estimator_

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=best_estimator, model_name=pipe_sc_k_neighbor_regressor.__name__)

    hypermodel = save_model(reg, model_name=pipe_sc_k_neighbor_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_k_neighbor_regressor.__name__)


########################################################################################################################
# Miscellaneous Regressors
########################################################################################################################

def pipe_sc_passive_aggressive_regressor(train_data, test_data):
    parameters = {'par__C': [1.0, 0.1, 10.0, 100.0, 1000.0],
                  'par__max_iter': [1000, 2000, 3000],
                  'par__loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']}

    par = PassiveAggressiveRegressor()
    scaler = StandardScaler()

    pipe = Pipeline(steps=[("scaler", scaler), ("par", par)])
    set_config(display="text")
    reg = GridSearchCV(pipe, parameters, cv=10, scoring="neg_mean_absolute_error", verbose=4, n_jobs=-1)
    reg.fit(train_data[0], train_data[1])

    predictions = reg.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    best_estimator = reg.best_estimator_

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=best_estimator, model_name=pipe_sc_passive_aggressive_regressor.__name__)

    hypermodel = save_model(reg, model_name=pipe_sc_passive_aggressive_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_passive_aggressive_regressor.__name__)


def pipe_sc_tweedie_regressor(train_data, test_data):
    parameters = {'tr__power': [0, 1, 1.5],
                  'tr__link': ['auto', 'identity', 'log'],
                  'tr__max_iter': [100, 200]}

    tr = TweedieRegressor()
    scaler = StandardScaler()

    pipe = Pipeline(steps=[("scaler", scaler), ("tr", tr)])
    set_config(display="text")
    reg = GridSearchCV(pipe, parameters, cv=10, scoring="neg_mean_absolute_error", verbose=4, n_jobs=-1)
    reg.fit(train_data[0], train_data[1])

    predictions = reg.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    best_estimator = reg.best_estimator_

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=best_estimator, model_name=pipe_sc_tweedie_regressor.__name__)

    hypermodel = save_model(reg, model_name=pipe_sc_tweedie_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_tweedie_regressor.__name__)


def pipe_sc_matern_gaussian_process_regressor(train_data, test_data):
    gpr = GaussianProcessRegressor(kernel=Matern(nu=1.5))
    scaler = StandardScaler()

    pipe = Pipeline(steps=[("scaler", scaler), ("gpr", gpr)])
    set_config(display="text")
    pipe.fit(train_data[0], train_data[1])

    predictions = pipe.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    # best_estimator = reg.best_estimator_

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=None, model_name=pipe_sc_matern_gaussian_process_regressor.__name__)

    hypermodel = save_model(pipe, model_name=pipe_sc_matern_gaussian_process_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_matern_gaussian_process_regressor.__name__)


def pipe_sc_voting_regressor(train_data, test_data):
    r2 = DecisionTreeRegressor(criterion='friedman_mse', min_samples_split=3)
    r1 = RidgeCV(cv=10)

    vr = VotingRegressor(estimators=[('dt', r2), ('l', r1)])
    scaler = StandardScaler()

    pipe = Pipeline(steps=[("scaler", scaler), ("vr", vr)])
    set_config(display="text")
    pipe.fit(train_data[0], train_data[1])

    predictions = pipe.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    best_estimator = pipe.get_params(deep=True)

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=best_estimator, model_name=pipe_sc_voting_regressor.__name__)

    hypermodel = save_model(pipe, model_name=pipe_sc_voting_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_voting_regressor.__name__)


def pipe_sc_gradient_boosting_regressor(train_data, test_data):
    parameters = {'gbr__loss': ['squared_error', 'absolute_error'],
                  'gbr__criterion': ['friedman_mse', 'squared_error'],
                  'gbr__learning_rate': [1e-1, 1e-2],
                  'gbr__n_estimators': [100, 150, 200],
                  'gbr__min_samples_split': [2, 3, 4],
                  'gbr__min_samples_leaf': [30, 40],
                  'gbr__max_features': [None, 42, 41, 40]}

    gbr = GradientBoostingRegressor()
    scaler = StandardScaler()

    pipe = Pipeline(steps=[("scaler", scaler), ("gbr", gbr)])
    set_config(display="text")
    reg = GridSearchCV(pipe, parameters, cv=10, scoring="neg_mean_absolute_error", verbose=4, n_jobs=-1)
    reg.fit(train_data[0], train_data[1])

    predictions = reg.predict(test_data[0])
    acc = r2_score(test_data[1], predictions)
    mse = mean_squared_error(test_data[1], predictions)
    mae = mean_absolute_error(test_data[1], predictions)
    rmse = root_mean_squared_error(test_data[1], predictions)
    mape = mean_absolute_percentage_error(test_data[1], predictions)
    medae = median_absolute_error(test_data[1], predictions)
    best_estimator = reg.best_estimator_

    save_variables_model(y_test_model=test_data[1], predictions_model=predictions, acc=acc, mse=mse, mae=mae,
                         pipeline=pipe, rmse=rmse, mape=mape, medae=medae,
                         best_estimator=best_estimator, model_name=pipe_sc_gradient_boosting_regressor.__name__)

    hypermodel = save_model(reg, model_name=pipe_sc_gradient_boosting_regressor.__name__)

    deploy_data_test = hypermodel.predict(test_data[0])
    plot_data(test_data[1], deploy_data_test, model_name=pipe_sc_gradient_boosting_regressor.__name__)


########################################################################################################################
# Artificial Neural Networks
########################################################################################################################

def tuner_ann_functional(train_data, test_data):
    def model_builder_ann_functional(hp):
        hp_units1 = hp.Int('units1', min_value=128, max_value=512, step=64)
        hp_units2 = hp.Int('units2', min_value=64, max_value=256, step=32)
        hp_units3 = hp.Int('units3', min_value=32, max_value=128, step=16)
        hp_units4 = hp.Int('units4', min_value=16, max_value=64, step=8)
        hp_units5 = hp.Int('units5', min_value=8, max_value=32, step=4)
        hp_units6 = hp.Int('units6', min_value=4, max_value=16, step=2)
        hp_activation = hp.Choice('activation', values=['relu', 'PReLU', 'LeakyReLU'])
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3])

        visible = Input(shape=(43,))

        # first dense feature extractor
        hidden1 = Dense(hp_units1, activation=hp_activation)(visible)
        hidden2 = Dense(hp_units2, activation=hp_activation)(hidden1)

        # second dense feature extractor
        hidden3 = Dense(hp_units2, activation=hp_activation)(hidden2)
        hidden4 = Dense(hp_units3, activation=hp_activation)(hidden3)

        # third dense feature extractor
        hidden5 = Dense(hp_units3, activation=hp_activation)(hidden4)
        hidden6 = Dense(hp_units4, activation=hp_activation)(hidden5)

        # fourth dense feature extractor
        hidden7 = Dense(hp_units4, activation=hp_activation)(hidden6)
        hidden8 = Dense(hp_units5, activation=hp_activation)(hidden7)

        # final dense layer
        # merge1 = concatenate([hidden2, hidden4, hidden6, hidden8])
        # hidden9 = Dense(hp_units4, activation=hp_activation)(merge1)
        hidden10 = Dense(hp_units5, activation=hp_activation)(hidden8)
        hidden11 = Dense(hp_units6, activation=hp_activation)(hidden10)

        # output layer
        output = Dense(1, activation=hp_activation)(hidden11)

        model_train = Model(inputs=visible, outputs=output)

        model_train.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                            loss='mse', metrics=['accuracy', 'mae'])

        return model_train

    tuner = kt.Hyperband(model_builder_ann_functional,
                         objective='val_loss',
                         max_epochs=50,
                         factor=3,
                         directory='ANN Hyper parameter Tuning',
                         project_name='Project 1 Adam')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    epochs = 50
    tuner.search(train_data[0], train_data[1], epochs=epochs,
                 validation_data=(test_data[0], test_data[1]), callbacks=[stop_early])

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
      The hyper parameter search is complete. The optimal number of units in the first densely-connected
      layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
      is {best_hps.get('learning_rate')}.
      """)

    # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(train_data[0], train_data[1], epochs=epochs,
                        validation_data=(test_data[0], test_data[1]))

    val_mae_per_epoch = history.history['val_mae']
    best_epoch = val_mae_per_epoch.index(min(val_mae_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    hypermodel = tuner.hypermodel.build(best_hps)

    # Retrain the model
    model_history = hypermodel.fit(train_data[0], train_data[1], epochs=best_epoch,
                                   validation_data=(test_data[0], test_data[1]))

    eval_result = hypermodel.evaluate(test_data[0], test_data[1])
    print("[test loss, test accuracy, test mae]:", eval_result)

    plot_learning_curve_ann(model_history, best_epoch, model_train=model_builder_ann_functional.__name__)

    save_model_ann(hypermodel_in=hypermodel, model_name=model_builder_ann_functional.__name__)

    predictions = hypermodel.predict(test_data[0])

    plot_data_ann(test_data[1], predictions, model_train=model_builder_ann_functional.__name__)
    save_variables_model_ann(test_data[1], predictions, acc=eval_result[1], mse=eval_result[0],
                             mae=eval_result[2], model_train=model_builder_ann_functional.__name__)


########################################################################################################################


if __name__ == '__main__':
    df = pd.read_csv(r'F:\New ML 8020\Dataframe\features_dataframe_new_roll.csv', delimiter=',')
    df.dropna(inplace=True)

    X_train, X_test, y_train, y_test = split_data(df_data=df)

    train_data_in = [X_train, y_train]
    test_data_in = [X_test, y_test]

    pipe_sc_decision_tree_regressor(train_data=train_data_in, test_data=test_data_in)
