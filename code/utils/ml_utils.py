import os
import joblib
from tqdm.notebook import tqdm

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, precision_score, recall_score, f1_score, RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix, roc_curve, auc
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt

from utils.function import rec_dd
from utils.eplot.constants import MCMAP, R_CMAP
from utils.eplot.core import heatmap, lineplot

class ML_model():
    def __init__(self, path, model_type):
        self.ml_path = path
        self.model_type = model_type
        self._ml_model = {}
        self._clf_tumor_types = []
        self._load_models()

        self._clf_tumor_types = list(self._ml_model.keys())
        if self.model_type == 'multi':
            self._tumor_types = np.unique(np.hstack(list(map(lambda x: x.split('_'), self._clf_tumor_types))))
        self._ml_data = rec_dd()
        for tumor_type in self._clf_tumor_types:
            self._load_data(tumor_type)

    def _load_models(self):
        model_path = os.path.join(self.ml_path, self.model_type, 'final_model')
        for f in os.listdir(model_path):
            if not f.endswith('joblib'):
                continue
            _, name1, name2, *_ = f.split('_')
            if self.model_type == 'binary':
                tumor_type = name1
                self._clf_tumor_types.append(tumor_type)
            elif self.model_type == 'multi':
                tumor_type = '_'.join((name1, name2))
                self._clf_tumor_types.extend([name1, name2])
            self._ml_model[tumor_type] = joblib.load(os.path.join(model_path, f))
            self._clf_tumor_types = list(set(self._clf_tumor_types))
    
    def _load_data(self, tumor_type):
        for cohort_ in ['discovery', 'validation']:
            if self.model_type == 'binary':
                data_path = os.path.join(self.ml_path, self.model_type, 'data', tumor_type+f'_{cohort_}_ml_data.csv')
            else:
                data_path = os.path.join(self.ml_path, self.model_type, 'data', f'multi_{cohort_}_ml_data.csv')
            data = pd.read_csv(data_path)
            suffix_ = {'discovery': '', 'validation': '_val'}.get(cohort_)
            if self.model_type == 'binary':
                self._ml_data[tumor_type][cohort_]['X'+suffix_] = data.iloc[:, 2:]
                self._ml_data[tumor_type][cohort_]['y'+suffix_] = data.iloc[:, 1]
            else:
                self._ml_data[cohort_]['X'+suffix_] = data.iloc[:, 2:]
                self._ml_data[cohort_]['y'+suffix_] = data.iloc[:, 1]

    def _handle_model_data(self, cohort, tumor_type=None):
        if self.model_type == 'binary':
            X = self._ml_data[tumor_type]['discovery']['X']
            y = self._ml_data[tumor_type]['discovery']['y']
            clf = self._ml_model[tumor_type]
            clf.fit(X, y)

            if cohort == 'validation':
                X_ = self._ml_data[tumor_type]['validation']['X_val']
                y_ = self._ml_data[tumor_type]['validation']['y_val']
                return clf, X_, y_
            else:
                return clf, X, y
                
        else:
            if cohort == 'discovery':
                X_ = self._ml_data['discovery']['X']
                y_ = self._ml_data['discovery']['y']
            if cohort == 'validation':
                X_ = self._ml_data['validation']['X_val']
                y_ = self._ml_data['validation']['y_val']            
            return X_, y_

    def evaluate_multiple_metrics(self, model, X, y, average=None, multi_class=None, pos_label=None):
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
                        metrics_kwargs = {'average': average_, 'pos_label': pos_label}
                        out[metric_name] = i(y, predict_X, **metrics_kwargs)
        return out

    def _evaluation(self, tumor_type, n_repeats=100, cohort=None):
        X = self._ml_data[tumor_type]['discovery']['X']
        y = self._ml_data[tumor_type]['discovery']['y']
        X_val = self._ml_data[tumor_type]['validation']['X_val']
        y_val = self._ml_data[tumor_type]['validation']['y_val']

        clf = self._ml_model[tumor_type]

        discovery_scores = rec_dd()
        validation_scores = {}
        if 'discovery' in cohort:
            for random_state in tqdm(range(1, n_repeats+1, 1)):
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, stratify=y, random_state=random_state)
                clf.fit(X_train, y_train)
                for cohort_name, j in zip(['train', 'test'], [[X_train, y_train], [X_test, y_test]]):
                    discovery_scores[random_state][cohort_name] = self.evaluate_multiple_metrics(clf, j[0], j[1], pos_label=tumor_type)
        if 'validation' in cohort:
            clf.fit(X, y)
            validation_scores = self.evaluate_multiple_metrics(clf, X_val, y_val, pos_label=tumor_type)
        mapping_ = {'discovery': discovery_scores, 'validation': validation_scores}
        return {cohort: mapping_[cohort] for cohort in cohort}

    def evaluate_model_performance(self, tumor_types=None, cohort=None, n_repeats=100):
        if isinstance(cohort, str):
            cohort = [cohort]
        if self.model_type == 'binary':
            if tumor_types == None:
                tumor_types = self._ml_model.keys()

            if 'discovery' in cohort:
                train_test_scores = {}
                for tumor_type in tumor_types:
                    train_test_scores[tumor_type] = self._evaluation(tumor_type, n_repeats=n_repeats, cohort=['discovery'])
                self.train_test_scores_df = pd.concat([pd.DataFrame().from_dict(v1).stack().rename_axis(['Metric', 'Cohort']).rename('Score').to_frame().assign(**{'Tumor type': tumor_type, 'Repeat': repeat}) for tumor_type, v in train_test_scores.items() for repeat, v1 in v['discovery'].items()]).reset_index()
            if 'validation' in cohort:
                validation_scores = {}
                for tumor_type in self._ml_model.keys():
                    validation_scores[tumor_type] = self._evaluation(tumor_type, cohort=['validation'])
                self.validation_scores_df = pd.concat([pd.DataFrame().from_dict(v, orient='index').stack().rename_axis(['Cohort', 'Metric']).rename('Score').to_frame().assign(**{'Tumor type': k}) for k, v in validation_scores.items()]).reset_index()

    def feature_importance(self, cohorts='discovery', tumor_types=None, n_repeats=100, random_state=42):
        self.fi = pd.DataFrame()
        if isinstance(cohorts, str):
            cohorts = [cohorts]
        for cohort in cohorts:
            if not tumor_types:
                tumor_types = self._clf_tumor_types
            for tumor_type in tumor_types:
                clf, X_, y_ = self._handle_model_data(tumor_type=tumor_type, cohort=cohort)
                r = permutation_importance(clf, X_, y_, n_repeats=n_repeats, random_state=random_state)
                fi = pd.DataFrame(np.repeat(r['importances'], 2, axis=0), index=np.hstack(X_.columns.map(lambda x: x.split('_large_than_'))), columns=['Repeat'+str(i) for i in range(1, 101, 1)]).rename_axis('Feature').rename_axis('Repeat', axis=1).groupby('Feature').sum().unstack().rename('Score').to_frame()
                fi.loc[:, 'for_order'] = fi.groupby(['Feature'])['Score'].transform(np.mean).values
                fi = fi.sort_values('for_order', ascending=False).iloc[:, :-1].assign(**{'Tumor type': tumor_type, 'Cohort': cohort})
                self.fi = pd.concat([self.fi, fi])

    def _multi_fit(self, cohort):
        fitted_models = {}
        for tumor_type, model in self._ml_model.items():
            X_, y_ = self._handle_model_data(tumor_type=tumor_type, cohort=cohort)
            fitted_models[tumor_type] = model.fit(X_, y_)
        self._ml_model = fitted_models

    def multi_predict(self, cohort, predict_proba=False):
        def mode_(a):
            unique_, count_ = np.unique(a, return_counts=True)
            return unique_[count_.argmax()]
        
        predict_ = {}
        for tumor_type, clf in self._ml_model.items():
            X_, _ = self._handle_model_data(cohort=cohort)
            predict_[tumor_type] = clf.predict(X_.reindex(clf.feature_names_in_, axis=1).dropna(how='all', axis=1))
        predict_ = np.vstack(list(predict_.values()))
        self.predict = np.apply_along_axis(mode_, axis=0, arr=predict_)

        if predict_proba:
            tmp = pd.DataFrame(predict_).rename_axis('base_model').rename_axis('sample', axis=1).stack().rename('predict').reset_index()
            self.predict_proba = tmp.assign(predict_count=tmp.groupby(['sample', 'predict']).transform(np.size))[['sample', 'predict', 'predict_count']].drop_duplicates().set_index(['sample', 'predict']).unstack().droplevel(0, 1).fillna(0)[self._tumor_types]
            self.predict_proba.index = X_.index

    def plot_roc(self, cohort='discovery', figsize=(2, 2), palette=MCMAP):
        _, ax = plt.subplots(1, 1, figsize=figsize)
        for tumor_type in self._clf_tumor_types:
            clf, X_, y_ = self._handle_model_data(tumor_type=tumor_type, cohort=cohort)
            RocCurveDisplay.from_estimator(clf, X_, y_, name=tumor_type, alpha=1, lw=1, ax=ax, color=palette[tumor_type])

        ax.plot([0, 1], [0, 1], linestyle='--', lw=.5, color='gray', label='Chance (AUC = 0.50)', alpha=.8)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Binary classifier ROC on the\n{} cohort'.format({'discovery':' Discovery', 'validation': 'Indenpendent validation'}.get(cohort)))
        ax.legend(loc='lower center', bbox_to_anchor=(0.6, .02), frameon=True, fontsize=5)
        return ax
    
    def plot_recall_precision_curve(self, cohort='discovery', figsize=(2, 2), palette=MCMAP):
        _, ax = plt.subplots(1, 1, figsize=figsize)
        for tumor_type in self._clf_tumor_types:
            clf, X_, y_ = self._handle_model_data(tumor_type=tumor_type, cohort=cohort)
            PrecisionRecallDisplay.from_estimator(clf, X_, y_, name=tumor_type, alpha=1, lw=1, ax=ax, color=palette[tumor_type])

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Binary classifier recall-precision curve\non the {} cohort'.format({'discovery':' Discovery', 'validation': 'Indenpendent validation'}.get(cohort)))
        ax.legend(loc='lower center', bbox_to_anchor=(0.6, .02), frameon=True, fontsize=5)
        
        f_scores = np.linspace(0.2, 0.8, num=4)
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            ax.plot(x[(y >= 0)&(y <= 1)], y[(y >= 0)&(y <= 1)], color="gray", alpha=0.2)
            ax.annotate("F1 = {0:0.1f}".format(f_score), xy=(1.05, y[45] - 0.02))
        return ax

    def _plot_confusion_matrix(self, y_, predict_, cohort, tumor_type=None, figsize=None):
        label_ = pd.unique(y_)
        cm = pd.DataFrame(confusion_matrix(y_, predict_), index=label_, columns=label_)
        ax = heatmap(cm, xticklabels=True, square=True, linewidth=1, annot=True, fmt='.0f', cmap=R_CMAP, figsize=figsize, vmax=cm.max().max()/1.5, vmin=None, center=None, cbar=False)
        ax.set_title('{} {} cohort\nconfusion matrix'.format({'multi': 'Pan-cancer', 'binary': tumor_type}.get(self.model_type), {'discovery': 'Discovery', 'validation': 'Independent validation'}.get(cohort)))
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        return ax 
    
    def plot_cm(self, tumor_types=None, cohort='discovery', figsize=(2, 2)):
        if self.model_type == 'multi':
            X_, y_ = self._handle_model_data(cohort=cohort)
            self.multi_predict(cohort)
            ax = self._plot_confusion_matrix(y_, self.predict, cohort, figsize=figsize)
        elif self.model_type == 'binary':
            if not tumor_types:
                tumor_types = self._clf_tumor_types
            ax = []
            for tumor_type in tumor_types:
                clf, X_, y_ = self._handle_model_data(cohort=cohort, tumor_type=tumor_type)
                predict_ = clf.predict(X_)
                ax_ = self._plot_confusion_matrix(y_, predict_, tumor_type, cohort, figsize)
                ax.append(ax_)
        return ax

    def plot_multi_roc(self, cohort='discovery', figsize=(2.5, 2.5), palette=MCMAP):
        _, y_ = self._handle_model_data(cohort=cohort)
        self.multi_predict(cohort, predict_proba=True)
        y_score = self.predict_proba.values
        label_binarizer = LabelBinarizer().fit(self._tumor_types)
        y_onehot_val = label_binarizer.transform(y_)
        fpr, tpr, roc_auc = dict(), dict(), dict()
        fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_val.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        iter_classes = np.where(np.in1d(self._tumor_types, pd.unique(y_)))[0]
        n_classes = iter_classes.shape[0]
        for i in iter_classes:
            fpr[i], tpr[i], _ = roc_curve(y_onehot_val[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr_grid = np.linspace(0.0, 1.0, 1000)
        mean_tpr = np.zeros_like(fpr_grid)
        for i in iter_classes:
            mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation
        mean_tpr /= n_classes
        fpr["macro"] = fpr_grid
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        _, ax = plt.subplots(figsize=figsize)
        micro_data = pd.DataFrame(np.vstack((fpr["micro"], tpr["micro"]))).T
        lineplot(micro_data, label=f"macro-average ROC (AUC = {roc_auc['macro']:.2f})", color="deeppink", linestyle=":", linewidth=2, ax=ax, spines_hide=[], ticklabels_hide=[])
        macro_data = pd.DataFrame(np.vstack((fpr["macro"], tpr["macro"]))).T
        lineplot(macro_data, label=f"macro-average ROC (AUC = {roc_auc['macro']:.2f})", color="navy", linestyle=":", linewidth=2, ax=ax, spines_hide=[], ticklabels_hide=[])
        if isinstance(palette, dict):
            palette = {k: palette[k] for k in self._tumor_types}
        else:
            palette = dict(zip(self._tumor_types, MCMAP[:len(self._tumor_types)]))
        for class_id, tumor_type in zip(iter_classes, pd.unique(y_)):
            RocCurveDisplay.from_predictions(
                y_onehot_val[:, class_id],
                y_score[:, class_id],
                name=f"ROC for {tumor_type}",
                color=palette[tumor_type],
                ax=ax,
                linewidth=1
            )
        ax.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Pan-cancer ROC (OVR) on the\n{} cohort".format({'discovery': 'Discovery', 'validation': 'Independent validation'}.get(cohort)))
        return ax
