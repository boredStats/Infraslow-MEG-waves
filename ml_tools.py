# -*- coding: utf-8 -*-

"""Workhorse file to keep analysis scripts pretty."""
import os
import numpy as np
import pandas as pd
from utils import ctime, save_xls, _get_meg_metadata
from copy import deepcopy
from sklearn import ensemble, svm, metrics
from sklearn.utils import resample
from sklearn.preprocessing import RobustScaler
from sklearn.cross_decomposition import PLSSVD
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.model_selection import GridSearchCV, PredefinedSplit


def _make_ExtraTrees(n_estimators=1000, criterion='mae', random_state=None):
    """Create an ExtraTreesRegressor object."""
    model = ensemble.ExtraTreesRegressor(
        n_estimators=n_estimators,
        criterion=criterion,
        random_state=random_state
        )
    return model


def _make_SVM(kernel='linear'):
    """Create an SVR object."""
    model = svm.SVR(kernel=kernel)
    return model


def generate_predefined_split(n=87, n_sessions=3):
    """Create a test_fold array for the PredefinedSplit function."""
    test_fold = []
    for s in range(n_sessions):
        test_fold.extend([s] * n)
    return test_fold


def _stack_session_data(session_dict, return_df=False):
    """Stack a dictionary of dataframes."""
    session_list = [session_dict[s] for s in session_dict]
    stacked_data = pd.concat(session_list, axis=0, ignore_index=True)
    if return_df is False:
        return stacked_data.values
    else:
        return stacked_data


def _unstack_session_data(array, sessions, n=87):
    split_arrays = np.split(array, len(sessions))
    unstacked = {}
    for s, data in enumerate(split_arrays):
        unstacked[sessions[s]] = pd.DataFrame(data)
    return unstacked


def _performance_battery(y_test, prediction, progress=False, debug=False):
    """Calculate MSE, MAE and r2 on fold result."""
    mse = metrics.mean_squared_error(y_test, prediction)
    mae = metrics.mean_absolute_error(y_test, prediction)
    expvar = metrics.explained_variance_score(y_test, prediction)

    if progress:
        print('MSE: %.4f' % mse)
        print('MAE: %.4f' % mae)
        print('Explained variance: %.4f' % expvar)
    if debug:
        print('Fold runtime: %s' % ctime())
    return mse, mae, expvar


def add_conjunction(feature_df, conjunction_test='all'):
    """Run a conjunction analysis to determine which features to keep."""
    output = deepcopy(feature_df)
    # Adding a "conjunction" column to the feature dataframe
    if conjunction_test == 'all':
        conj = [int(all(feature_df.loc[f] > 0)) for f in feature_df.index]
    elif conjunction_test == 'any':
        conj = [int(any(feature_df.loc[f] > 0)) for f in feature_df.index]
    output['Conjunction'] = conj

    # Create dataframe with features that pass a conjunction test
    conj_df = output[output['Conjunction'] == 1].copy()
    conj_df.drop(columns='Conjunction', inplace=True)
    average_importances = np.mean(conj_df.values, axis=1)
    conj_df['Average Importance'] = average_importances

    # Create a sorted conjunction dataframe with the average feature importance
    sorted_conj_df = pd.DataFrame(
        average_importances,
        index=conj_df.index,
        columns=['Average Importance'])
    sorted_conj_df.sort_values(
        by='Average Importance',
        inplace=True,
        ascending=False)
    return output, sorted_conj_df


def plsc(x, y, sessions=None):
    if not sessions:
        _, sessions = _get_meg_metadata()

    if type(x) == dict:
        x_ = ml_tools._stack_session_data(x)
    else:
        x_ = x.values
    y.tolist()
    y_ = []
    for s in range(len(sessions)):
        y_.extend(y)

    est = PLSSVD()
    lv, _ = est.fit_transform(x_, np.asarray(y_))
    return lv


class ML_pipeline:
    """Machine learning pipeline for this project."""

    def __init__(
            self,
            predictors,
            targets,
            run_PLSC=False,
            feature_selection_gridsearch=None,
            model_gridsearch=None,
            feature_names=None,
            session_names=None,  # session names or n_sessions
            random_state=None,
            debug=False):
        """Initialize pipeline with data and targets."""
        if type(predictors) == dict:
            self.X = _stack_session_data(predictors)
        elif type(predictors) == pd.DataFrame:
            self.X = predictors.values
        else:
            self.X = predictors

        ylist = []
        try:
            for s in range(len(session_names)):
                ylist.append(targets)
        except Exception:
            for s in range(session_names):
                ylist.append(targets)

        self.y = pd.concat(ylist, axis=0, ignore_index=True).values

        self.pls_check = run_PLSC
        self.fs_grid = feature_selection_gridsearch
        self.reg_grid = model_gridsearch

        if feature_names is not None:
            self.feat_names = feature_names
        else:
            n_cols = predictors.shape[1]
            quick_feats = ['Feature%04d' % f for f in range(n_cols)]
            self.feat_names = quick_feats
        if type(session_names) == list:
            self.sessions = session_names
        else:
            self.sessions = ['Session%d' % (s+1) for s in range(session_names)]
        self.random_state = random_state
        self.debug = debug

    def feat_select(
            self,
            estimator,
            X_train, X_test, y_train,
            n_feats=None,
            method='SelectFromModel'):
        """Feature selection for fold data."""
        if self.pls_check is False:
            preproc = RobustScaler().fit(X_train)
            X_train_normalized = preproc.transform(X_train)
            X_test_normalized = preproc.transform(X_test)
        else:
            plsc = PLSSVD()
            plsc.fit(X_train, y_train)
            X_train_normalized = plsc.transform(X_train)
            X_test_normalized = plsc.transform(X_test)

        if method == 'RFE':
            fs_model = RFE(estimator, n_features_to_select=n_feats)
            fs_model.fit(X_train_normalized, y_train)
        elif method == 'SelectFromModel':
            estimator.fit(X_train_normalized, y_train)
            fs_model = SelectFromModel(
                estimator, prefit=True, max_features=n_feats)

        X_train_fs = fs_model.transform(X_train_normalized)
        X_test_fs = fs_model.transform(X_test_normalized)

        self.feature_indices = list(fs_model.get_support(indices=True))

        self.X_train_fs = X_train_fs
        self.X_test_fs = X_test_fs
        return self

    def regress(self, estimator, y_train,):
        """Regress x onto y."""
        estimator.fit(self.X_train_fs, y_train)
        predicted = estimator.predict(self.X_test_fs)

        return predicted, estimator

    def run_predictions(
            self,
            permute=False,
            feature_selection='SVM',
            feature_selection_kernel='linear',
            model='ExtraTrees',
            model_kernel='linear'):
        """Core machine learning pipeline."""
        if type(permute) == int:
            self.X = resample(self.X)

        # Outputs
        k = len(self.sessions)
        performance_array = np.ndarray(shape=(3, k))
        feature_array = np.zeros(shape=(len(self.feat_names), k))
        prediction_dict = {}

        test_fold = generate_predefined_split(n_sessions=k)
        ps = PredefinedSplit(test_fold)

        # Feature selection setup
        if feature_selection == 'SVM':
            fs_model = _make_SVM(kernel=feature_selection_kernel)
        elif feature_selection == 'ExtraTrees':
            if self.perm_check is True:
                seed = None
            else:
                seed = self.random_state
            fs_model = _make_ExtraTrees(random_state=seed)

        if self.fs_grid is not None:
            gridsearch = GridSearchCV(
                fs_model, self.fs_grid,
                cv=ps, scoring='explained_variance')
            gridsearch.fit(self.X, self.y)
            kwargs = gridsearch.best_params_
            fs_model.set_params(**kwargs)

            search_results = gridsearch.cv_results_
            grid_df = pd.DataFrame.from_dict(search_results)
            self.feature_selection_gridsearch_results = grid_df

        # Regression model setup
        if model == 'SVM':
            reg_model = _make_SVM(kernel=model_kernel)
        elif model == 'ExtraTrees':
            reg_model = _make_ExtraTrees(random_state=self.random_state)

        if self.reg_grid is not None:
            gridsearch = GridSearchCV(
                reg_model, self.reg_grid,
                cv=ps, scoring='explained_variance')
            gridsearch.fit(self.X, self.y)
            kwargs = gridsearch.best_params_
            reg_model.set_params(**kwargs)

            search_results = gridsearch.cv_results_
            grid_df = pd.DataFrame.from_dict(search_results)
            self.regression_gridsearch_results = grid_df

        for s, (tr, ts) in enumerate(ps.split()):
            session = self.sessions[s]
            train_index, test_index = tr, ts

            X_train, X_test = self.X[train_index, :], self.X[test_index, :]
            y_train, y_test = self.y[train_index], self.y[test_index]

            self.feat_select(fs_model, X_train, X_test, y_train)
            prediction, reg = self.regress(reg_model, y_train)

            mse, mae, expvar = _performance_battery(
                y_test, prediction, debug=self.debug)
            performance_array[:, s] = np.array([mse, mae, expvar])
            try:
                importances = np.ndarray.flatten(reg.feature_importances_)
                for i, index in enumerate(self.feature_indices):
                    feature_array[index, s] = importances[i]
            except AttributeError:
                pass

            prediction_df = pd.DataFrame(
                np.array([y_test, prediction]).T,
                columns=['True Score', 'Predicted Score'])
            prediction_dict[session] = prediction_df

        self.predictions = prediction_dict

        feature_df = pd.DataFrame(
            feature_array,
            index=self.feat_names,
            columns=self.sessions)
        self.feature_importances = feature_df

        perf_df = pd.DataFrame(
            performance_array,
            index=['MSE', 'MAE', 'ExplainedVariance'],
            columns=self.sessions)
        self.model_performance = perf_df


def save_outputs(ML_pipe, output_dir):
    """Quick convenience function for saving outputs."""
    stack, conjunction = add_conjunction(ML_pipe.feature_importances)
    conjunction_results = {'Features': conjunction, 'Raw': stack}
    save_xls(
        conjunction_results, os.path.join(output_dir, 'features.xlsx'))

    save_xls(
        ML_pipe.predictions, os.path.join(output_dir, 'predictions.xlsx'))

    ML_pipe.model_performance.to_excel(
        os.path.join(output_dir, 'performance.xlsx'))

    try:
        ML_pipe.feature_selection_gridsearch_results.to_excel(
            os.path.join(output_dir, 'feature_selection_gridsearch.xlsx'))
        ML_pipe.regression_gridsearch_results.to_excel(
            os.path.join(output_dir, 'regression_gridsearch.xlsx'))
    except AttributeError:
        pass


def perm_tests(ML_pipeline, n_iters):
    """Do permutation testing."""
    def _perm_df(n_iters, sessions):
        rownames = ['Iter %04d' % (n+1) for n in range(n_iters)]
        df = pd.DataFrame(index=rownames, columns=sessions)
        return df

    sessions = ML_pipeline.sessions
    expvar_perm = _perm_df(n_iters, sessions)
    mae_perm = _perm_df(n_iters, sessions)
    mse_perm = _perm_df(n_iters, sessions)

    if type(n_iters) != int:
        raise ValueError('Parameter "n_iters" must be an integer.')
    n = 0
    while n != n_iters:
        print('Permutation %03d: %s' % (n+1, ctime()))
        ML_pipeline.run_predictions(permute=n_iters)

        row = list(expvar_perm.index)[n]
        perm_perf = ML_pipeline.model_performance
        for s in sessions:
            expvar_perm.loc[row][s] = perm_perf.loc['ExplainedVariance'][s]
            mae_perm.loc[row][s] = perm_perf.loc['MAE'][s]
            mse_perm.loc[row][s] = perm_perf.loc['MSE'][s]

        n += 1
    perm_dict = {
        'ExplainedVariance': expvar_perm,
        'MSE': mse_perm,
        'MAE': mae_perm
        }
    return perm_dict


def compare_algorithms(band='infraslow', model='PSD'):
    results_dir = "./results"
    flist = os.listdir(results_dir)
    psd_dirs = [d for d in flist if model in d and '.xlsx' not in d]
    band_dirs = [d for d in psd_dirs if band in d]
    algorithms = [' '.join(str(d).split('_')[2:]) for d in band_dirs]

    temp_dir = os.path.join(results_dir, band_dirs[0])
    temp_df = pd.read_excel(
        os.path.join(temp_dir, 'performance.xlsx'), index_col=0)
    performance_measures = list(temp_df.index)
    sessions = list(temp_df)

    compare_dict = {}
    for m in performance_measures:
        compare_df = pd.DataFrame(index=algorithms, columns=sessions)
        avgs = []
        for i, d in enumerate(band_dirs):
            algorithm = algorithms[i]
            folder = os.path.join(results_dir, d)
            performance_file = os.path.join(folder, 'performance.xlsx')
            performance_df = pd.read_excel(performance_file, index_col=0)
            tally = []
            for s in sessions:
                val = performance_df.loc[m][s]
                compare_df.loc[algorithm][s] = val
                tally.append(val)
            avgs.append(np.mean(tally))
        compare_df['Average'] = avgs
        compare_dict[m] = compare_df

    return compare_dict


def pick_algorithm(comparison, model='PSD', criterion='ExplainedVariance'):
    df = comparison[criterion]
    pick = df['Average']
    chosen_alg = pick.idxmax()

    results_dir = "./results"
    flist = os.listdir(results_dir)
    model_dirs = [d for d in flist if model in d and '.xlsx' not in d]
    ad = [d for d in model_dirs if chosen_alg in d][-1]
    alg_dir = os.path.join(results_dir, ad)
    feat_file = [f for f in os.listdir(alg_dir) if 'features' in f][-1]

    feature_df = pd.read_excel(os.path.join(alg_dir, feat_file), index_col=0)
    rois_to_return = list(feature_df.index)
    return rois_to_return
