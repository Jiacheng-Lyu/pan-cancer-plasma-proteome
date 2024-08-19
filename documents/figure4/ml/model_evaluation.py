import argparse

import os
import joblib
from collections import defaultdict
from time import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score, classification_report, make_scorer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=str, help='File name')
    parser.add_argument('--model_inpath', type=str, default='./', help='trained model path')
    return parser.parse_args()

def evaluate_multiple_metrics(model, X, y, average=None, multi_class=None):
    name_space = {'roc_auc_score': 'AUROC', 'balanced_accuracy_score': 'Balanced accuracy', 'precision_score': 'Precision', 'recall_score': 'Recall', 'f1_score': 'F1'}
    n_class = np.unique(y).shape[0]
    if average and isinstance(average, str):
        average = [average]
    elif n_class > 2:
        average = ['weighted']
    else:
        average = [average]

    out = {}
    for i in [roc_auc_score, balanced_accuracy_score, precision_score, recall_score, f1_score]:
        metrics_kwargs = {}
        name = i.__name__
        name_ = name_space[name]
        if name == 'roc_auc_score':
            try:
                if n_class > 2:
                    predict_prob_X = model.predict_proba(X)[:, :n_class]
                else:
                    predict_prob_X = model.predict_proba(X)[:, 1]
            except:
                predict_prob_X = model.decision_function(X)
            if multi_class:
                if isinstance(multi_class, str):
                    multi_class = [multi_class]
            elif n_class > 2:
                multi_class = ['ovo']
            else:
                multi_class = [multi_class]

            for average_ in average:
                if not average_:
                    average_ = 'macro'
                    metrics_kwargs['average'] = average_
                if multi_class:
                    for multi_class_ in multi_class:
                        if multi_class_ != None:                            
                            metrics_kwargs['multi_class'] = multi_class_
                    try:
                        if n_class > 2:
                            metric_name = ' '.join((name_, average_, multi_class_))
                        else:
                            metric_name = name_
                        out[metric_name] = i(y, predict_prob_X, **metrics_kwargs)
                        
                    except:
                        pass
        else:
            predict_X = model.predict(X)
            if name == 'balanced_accuracy_score':
                out[name_] = i(y, predict_X, **metrics_kwargs)
            else:
                if n_class > 2:
                    metric_name = ' '.join((name_, average_, multi_class_))
                else:
                    metric_name = name_
                for average_ in average:
                    if not average_:
                        average_ = 'binary'
                    metrics_kwargs = {'average': average_}
                    out[metric_name] = i(y, predict_X, **metrics_kwargs)
    return out

def main():
    args = parse_args()
    name = args.name
    combine_path = args.model_inpath
    tumor_type = 'all'    
    model_dir = os.path.join(combine_path, 'no', 'without_ranked')
    outputpath = os.path.join(combine_path, 'evaluated')
    if not os.path.isdir(outputpath):
        os.mkdir(outputpath)
    discovery_data = pd.read_csv(os.path.join(combine_path, name+'_ml_data.csv'))
    # validation_data = pd.read_csv(os.path.join(combine_path, name+'_validation_ml_data.csv'))

    for model_file_name in os.listdir(model_dir):
        if not model_file_name.endswith('joblib'):
            continue
        print(model_file_name)
        train_test_scores = {}
        # validation_scores = {}

        # for tumor_type, model_file_name in final_models.items():
        model_path = os.path.join(model_dir, model_file_name)
        model = joblib.load(model_path)
        scores = defaultdict(dict)

        le = model.steps[0][1]
        clf = model.steps[1][1]

        X = discovery_data.iloc[:, 2:]
        y = discovery_data.iloc[:, 1]
        # X_valid = validation_data.iloc[:, 2:]
        # y_valid = validation_data.iloc[:, 1]    

        # bootstrap evaluation
        for random_state in tqdm(range(1, 101, 1)):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, stratify=y, random_state=random_state)
            y_train = le.transform(y_train)
            y_test = le.transform(y_test)
            clf.fit(X_train, y_train)

            for cohort_name, j in zip(['train', 'test'], [[X_train, y_train], [X_test, y_test]]):
                scores[random_state][cohort_name] = evaluate_multiple_metrics(clf, j[0], j[1])
            train_test_scores[tumor_type] = scores

        train_test_scores_df = pd.DataFrame()
        for tumor_type, scores in train_test_scores.items():
            df = pd.concat([pd.DataFrame().from_dict(v, orient='columns').rename(columns=lambda x: str(k)+'_'+x) for k, v in scores.items()], axis=1)
            df.columns = df.columns.map(lambda x: (x.split('_')[0], x.split('_')[1]))
            df.columns.names=('repeat', 'cohort')
            df = df.T.swaplevel(0, 1).rename_axis('score_type', axis=1).stack().rename('scores').reset_index()
            train_test_scores_df = pd.concat([train_test_scores_df, df])
        y = le.transform(y)
        # y_valid = le.transform(y_valid)
        clf.fit(X, y)
        # validation_scores[tumor_type] = evaluate_multiple_metrics(clf, X_valid, y_valid)
        joblib.dump(train_test_scores, os.path.join(outputpath, '{}_train-test.joblib').format(model_file_name.split('.')[0]))
        # joblib.dump(validation_scores, os.path.join(outputpath, '{}_validation.joblib').format(model_file_name.split('.')[0]))

if __name__ == '__main__':
    start = time()
    main()
    end = time()
    print("Total time: {:.2f} mins".format((end - start) / 60))