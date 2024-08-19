import argparse

import os
import joblib
import json
from copy import copy
from collections import defaultdict
from time import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.stats

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_validate, RepeatedStratifiedKFold, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, RationalQuadratic, WhiteKernel

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score, classification_report, make_scorer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', nargs='+', default=None, help='Features used in the model training')
    parser.add_argument('--train_size', type=float, default=.7, help='Cohort split proportion')
    parser.add_argument('--preprocessing', type=str, default='standard', help='Preprocessing of the data, one of the standard, minmax, normalizer,and robust')
    parser.add_argument('--decomposition', type=str, default=None, help='Decomposition for the data, one of the pca, tsne, and umap')
    parser.add_argument('--dec_kwargs', type=json.loads, default=None, help='The dictionary of the parameter for the decomposition, see details in sklearn.decomposition.PCA, sklearn.manifold.TSNE, and umap')
    parser.add_argument('--ranked', type=int, default=0, help='Ranked preprocessing data further')
    parser.add_argument('--cv_folds', type=int, default=5, help='Cross validation folds')
    parser.add_argument('--n_jobs', type=int, default=4, help='Count of CPU cores used for the model training')
    parser.add_argument('--topn', type=int, default=10, help='Selet top N best models for the downstream hyperparameter tuning')
    parser.add_argument('--cutoffs', nargs='+', default=[.9], help='The cutoff of the metric for the best model selection')
    parser.add_argument('--tuned_metrics', type=str, default='', help='The metrics used for the hyperparameter tuning')
    parser.add_argument('--data_inpath', type=str, default='./train_data.csv', help='Train data of the given donor')
    parser.add_argument('--output_path', type=str, default='./', help='Output path')
    parser.add_argument('--prefix_name', type=str, default='', help='Output prefix name')
    return parser.parse_args()

def model_benchmarking(cv_folds, X_train, y_train):
    global MLA
    MLA = [
        # Ensemble Methods
        ensemble.AdaBoostClassifier(random_state=42),
        ensemble.BaggingClassifier(random_state=42),
        ensemble.ExtraTreesClassifier(random_state=42),
        ensemble.GradientBoostingClassifier(random_state=42),
        ensemble.RandomForestClassifier(random_state=42),

        # Gaussian Processes
        gaussian_process.GaussianProcessClassifier(random_state=42),

        # GLM
        # linear_model.LogisticRegressionCV(random_state=42),
        linear_model.PassiveAggressiveClassifier(random_state=42),
        linear_model.RidgeClassifierCV(),
        linear_model.Perceptron(random_state=42),

        # Navies Bayes
        naive_bayes.BernoulliNB(),
        naive_bayes.GaussianNB(),

        # Nearest Neighbor
        neighbors.KNeighborsClassifier(),

        # SVM
        svm.SVC(probability=True, random_state=42),
        # svm.NuSVC(probability=True),

        # Trees
        tree.DecisionTreeClassifier(random_state=42),
        tree.ExtraTreeClassifier(random_state=42),

        # Discriminant Analysis
        discriminant_analysis.LinearDiscriminantAnalysis(),
        discriminant_analysis.QuadraticDiscriminantAnalysis(),

        # xgboost:
        XGBClassifier(random_state=42)
    ]

    CV_FOLDS = cv_folds
    CV_REPEATS = 10
    cv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=42)
    if target == 'binary':
        scoring = ['precision', 'recall', 'f1', 'balanced_accuracy', 'roc_auc']
    else:
        scoring = {
            'accuracy': make_scorer(balanced_accuracy_score),
            'precision_weighted': make_scorer(precision_score, average='weighted'),
            'recall_weighted': make_scorer(recall_score, average='weighted'),
            'f1_weighted': make_scorer(f1_score, average='weighted'),
            'roc_auc': make_scorer(roc_auc_score, average='weighted', multi_class='ovo', needs_proba=True)
        }

    summary = []

    for model in MLA:
        print('benchmark {}'.format(model))
        result = cross_validate(
             model,
             X=X_train,
             y=y_train,
             groups=y_train,
             cv=cv,
             scoring=scoring
        )

        result['model'] = model.__class__.__name__
        summary.append(pd.DataFrame(result))
    print("Model Benchmarking complete!")  
    return summary

def hyperparameter_tuning(X_train, y_train, topn_used_model, cv_folds, scoring, n_jobs):
    tree_based_params = {                
        'n_estimators': range(10, 201, 10),
        'max_features': range(1, 5, 1),
        'min_samples_leaf': range(1, 6, 1),
        'min_samples_split': range(2, 6, 1),
        'class_weight': ['balanced', 'balanced_subsample'],
        'random_state': [42]
        }

    xgboost_params = {
        'max_depth':range(2,6,1),
        'min_child_weight':range(1,6,2),
        'max_depth':[3,4,5],
        'min_child_weight':[3,4,5],
        'min_child_weight':[3,4,5],
        'gamma':[i/10.0 for i in range(1, 9, 2)],
        'subsample':[i/10.0 for i in range(1, 9, 2)],
        'colsample_bytree':[i/10.0 for i in range(1, 9, 2)],
        'learning_rate': [i/10.0 for i in range(1, 9, 2)],
        'random_state': [42]
    }

    ridge_params = {
        'alphas': np.linspace(0.1, 10, 100),
        'class_weight': ['balanced']
    }

    logistic_params = {
        'Cs': range(1, 101),
        'class_weight': ['balanced'],
        'random_state': [42]
    }

    lda_params = {
        'solver': ['svd', 'lsqr', 'eigen'],
        'shrinkage': np.arange(0, 1, 0.01),
        
    }

    NB_params = {
        'alpha': np.linspace(0, 1, 51),
        'force_alpha': [True]
    }

    DB_params = {
        'var_smoothing': np.logspace(0,-9, num=10)
    }

    svc_params = {
        'C': np.logspace(-5, 5, num=11),
        'gamma': np.logspace(-5, 5, num=11),
        'kernel': ['rbf', 'sigmoid'],
        'class_weight': ['balanced'],
        'random_state': [42],
        'max_iter': [int(100000)]
    }

    gpc_params = {
        'kernel': [1*RBF(), 1*DotProduct(), 1*Matern(),  1*RationalQuadratic(), 1*WhiteKernel()]
    }

    gb_params = {
        'n_estimators':range(10,101,10),
        'max_depth':range(2,9,1),
        'min_samples_split':range(2,6,1),
        'min_samples_leaf': range(1, 6, 1),
        'subsample':[i/10.0 for i in range(1, 9, 2)],
        'learning_rate': [i/10.0 for i in range(1, 9, 2)],
        'random_state': [42],
        'max_features': range(2, 6, 1)
    }

    nusvc_params = {
        'nu': np.linspace(0, 1, 11),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': np.linspace(0, 1, 101),
        'probability': [True],
        'class_weight': ['balanced'],
        'random_state': [42],
        'max_iter': [int(100000)]
    }

    bagging_params = {
        'n_estimators': range(10, 201, 10),
        'max_samples': np.linspace(.5, 1, 6),
        'max_features': np.linspace(.5, 1, 6),
        'random_state': [42]
    }

    adab_params = {
        'n_estimators': range(10, 201, 10),
        'learning_rate': [i/10.0 for i in range(1, 9, 2)],
        'random_state': [42]
    }

    knn_params = {
        'n_neighbors': range(5, 21, 1),
        'weights': ['uniform', 'distance'],
        'p': range(1, 5, 1)
    }

    pac_params = {
        'C': np.linspace(0.01, 1, 100),
        'random_state': [42],
        'class_weight': ['balanced']
    }

    percep_params = {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'alpha': np.logspace(-10, 0, 11),
        'l1_ratio': np.linspace(0, 1, 101),
        'class_weight': ['balanced']
    }

    qda_params = {
        
    }

    tree_params = {
        'criterion':['gini','entropy'],
        'max_depth':np.arange(1, 5),
        'min_samples_split':np.arange(2, 5),
        'max_leaf_nodes':np.arange(3, 5)
    }

    MLA_tuned_params = {
        'RandomForestClassifier': tree_based_params,
        'XGBClassifier': xgboost_params,
        'ExtraTreesClassifier': tree_based_params,
        'RidgeClassifierCV': ridge_params,
        'LogisticRegressionCV': logistic_params,
        'LinearDiscriminantAnalysis': lda_params,
        'BernoulliNB': NB_params,
        'GaussianNB': DB_params,
        'SVC': svc_params,
        'GaussianProcessClassifier': gpc_params,
        'GradientBoostingClassifier': gb_params,
        'NuSVC': nusvc_params, 
        'BaggingClassifier': bagging_params, 
        'AdaBoostClassifier': adab_params,
        'KNeighborsClassifier': knn_params,
        'PassiveAggressiveClassifier': pac_params,
        'Perceptron': percep_params,
        'QuadraticDiscriminantAnalysis': qda_params,
        'DecisionTreeClassifier': tree_params,
        'ExtraTreeClassifier': tree_params
    }

    tuned_topn_model = {}
    models = {model.__class__.__name__: model for model in MLA}
    for model_name in topn_used_model:
        print(model_name)
        clf = models[model_name]
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        random_search = RandomizedSearchCV(
            clf, 
            param_distributions=MLA_tuned_params[model_name], 
            n_iter=300,
            scoring=scoring, 
            n_jobs=n_jobs, 
            cv=skf.split(X_train, y_train), 
            verbose=1, 
            random_state=42
        )
        random_search.fit(X_train, y_train)
        tuned_topn_model[model_name] = random_search
    print("Models hyperparameter tuning complete!")  
    return tuned_topn_model

def select_best(tuned_topn_model, X_train, y_train, cutoff):
    scores = defaultdict(dict)
    for k1, v1 in tuned_topn_model.items():
        model = v1.best_estimator_.fit(X_train, y_train)
        for t in [f1_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score]:
            if target == 'binary':
                if t.__name__ == 'roc_auc_score':
                    try:
                        y_predict = model.predict_proba(X_train)[:, 1]
                    except:
                        y_predict = model.decision_function(X_train)
                else:
                    y_predict = model.predict(X_train)
                
                scores[k1]['Train_'+t.__name__] = t(y_train, y_predict)
            else:
                if t.__name__ == 'roc_auc_score':
                    try:
                        y_predict = model.predict_proba(X_train)
                    except:
                        y_predict = model.decision_function(X_train)
                else:
                    y_predict = model.predict(X_train)
                
                if t.__name__ == 'roc_auc_score':
                    params = {'average': 'weighted', 'multi_class': 'ovr'}
                elif t.__name__ != 'balanced_accuracy_score':
                    params = {'average': 'weighted'}
                else:
                    params = {}
                scores[k1]['Train_'+t.__name__] = t(y_train, y_predict, **params)

    scores_df = pd.DataFrame().from_dict(scores).rename_axis('model', axis=1).rename_axis('metrics').mean().sort_values(ascending=False)
    if scores_df.max() < cutoff:
        print('strategy 1')
        best_model = tuned_topn_model[scores_df.index[0]].best_estimator_
    else: 
        print('strategy 2')
        best_model = tuned_topn_model[scores_df[scores_df>cutoff].index[-1]].best_estimator_
    return scores, best_model

def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def main():
    global target
    args = parse_args()
    feature = args.feature
    train_size = args.train_size
    preprocessing = args.preprocessing
    decomposition = args.decomposition
    dec_kwargs = args.dec_kwargs
    ranked = args.ranked
    cv_folds = args.cv_folds
    n_jobs = args.n_jobs
    topn = args.topn
    cutoffs = args.cutoffs
    tuned_metrics = args.tuned_metrics
    data_inpath = args.data_inpath
    output_path = args.output_path
    prefix_name = args.prefix_name
    
    pipeline = []

    if prefix_name:
        prefix_name = prefix_name + '_'

    mkdir(output_path)
    output_path = os.path.join(output_path, preprocessing)
    mkdir(output_path)

    if ranked != 0:
        output_path = os.path.join(output_path, 'ranked')
    else:
        output_path = os.path.join(output_path, 'without_ranked')
    mkdir(output_path)

    data = pd.read_csv(data_inpath, index_col=0).dropna()
    if feature:
        X = data[feature].astype(np.float64)
    else:
        X = data.iloc[:, 1:].astype(np.float64)

    y = data.iloc[:, 0]
    le = LabelEncoder()
    pipeline.append(('label', le))
    y = le.fit_transform(y)

    class_ = len(np.unique(y))
    if class_ == 1:
        raise ValueError('target should at least 2 groups!')
    elif class_ == 2:
        target = 'binary'
    else:
        target = 'multiple'

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, stratify=y, random_state=42)
    Scaler = {'standard': StandardScaler(), 'minmax': MinMaxScaler(), 'normalizer': Normalizer(), 'robust': RobustScaler()}.get(preprocessing, 'no')
    if Scaler != 'no':
        scaler = Scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        pipeline.append((preprocessing, Scaler))

    if decomposition:
        decomposition_algorithm = {'pca': PCA(**dec_kwargs), 'tsne': TSNE(**dec_kwargs), 'umap': umap.UMAP(**dec_kwargs)}.get(decomposition)
        dec = decomposition_algorithm.fit(X_train)
        X_train = dec.transform(X_train)
        X_test = dec.transform(X_test)
        pipeline.append((decomposition, decomposition_algorithm))

    if ranked != 0:
        X_train = scipy.stats.rankdata(X_train, axis=1)
        X_test = scipy.stats.rankdata(X_test, axis=1)
    
    benchmarking_summary = model_benchmarking(cv_folds, X_train, y_train)
    if isinstance(benchmarking_summary, list):
        summary_df = pd.concat(benchmarking_summary)

    summary_df.to_csv(os.path.join(output_path, prefix_name+'model_benchmarking_result.csv'))
    topn_used_model = summary_df.groupby(['model']).mean().iloc[:, 2:].mean(axis=1).sort_values(ascending=False).iloc[:topn].index.tolist()
    print('Top best models used for the hyperparameter tuning: '.format(', '.join(topn_used_model)))

    if not tuned_metrics:
        if target == 'binary':
            scoring = 'roc_auc'
        else:
            scoring = 'roc_auc_ovo_weighted'

    tuned_topn_model = hyperparameter_tuning(X_train, y_train, topn_used_model, cv_folds, scoring, n_jobs)
    for k, v in tuned_topn_model.items():
        pipeline_ = copy(pipeline)
        pipeline_.append((k, v.best_estimator_))
        pipe = Pipeline(pipeline_)
        joblib.dump(pipe, os.path.join(output_path, prefix_name+'{}_tuned_model.joblib'.format(k)))

    for cutoff in cutoffs:
        cutoff_output_path = os.path.join(output_path, 'model_cutoff_{}'.format(cutoff))
        if not os.path.isdir(cutoff_output_path):
            os.mkdir(cutoff_output_path)
        scores, best_model = select_best(tuned_topn_model, X_train, y_train, float(cutoff))

        test_report = classification_report(y_test, best_model.predict(X_test), output_dict=True)
        test_report_df = pd.DataFrame().from_dict(test_report)
        train_report = classification_report(y_train, best_model.predict(X_train), output_dict=True)
        train_report_df = pd.DataFrame().from_dict(train_report)

        test_report_df.to_csv(os.path.join(cutoff_output_path, prefix_name+'best_model_test_report.csv'))
        train_report_df.to_csv(os.path.join(cutoff_output_path, prefix_name+'best_model_train_report.csv'))

        joblib.dump(scores, os.path.join(cutoff_output_path, prefix_name+'tuned_model_scores.joblib'))
        pipeline_ = copy(pipeline)
        pipeline_.append((k, best_model))
        joblib.dump(pipeline_, os.path.join(cutoff_output_path, prefix_name+'best_model_{}.joblib'.format(best_model.__class__.__name__)))


if __name__ == '__main__':
    start = time()
    main()
    end = time()
    print("Total time: {:.2f} mins".format((end - start) / 60))