import pandas as pd
import json
import configparser
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, roc_curve, auc, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from xgboost.sklearn import XGBRegressor
from sklearn.compose import TransformedTargetRegressor
from imblearn.over_sampling import ADASYN, SMOTE
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import math
import copy
import itertools
import shap
import pathlib
import os
from scipy.stats import shapiro


def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor

def delete_unwanted_covariates(codebook: pd.core.frame.DataFrame, features : pd.DataFrame, selection: str, one_hot_columns: list):
    continuous_covariates = list(codebook['variable_name'].loc[
        ((codebook['var_type'] == 'numeric') | (codebook['var_type'] == 'unknown') | (codebook['var_type'] == 'quintiles')) &
        (codebook['selected'] == 1) &
        (codebook['domain'] == "Covariates")
    ])
    categorical_covariates = list(codebook['variable_name'].loc[
        ((codebook['var_type'] == 'categorical') | (codebook['var_type'] == 'labels (0-1)') | (codebook['var_type'] == 'labels')
         | (codebook['var_type'] == 'factor')) &
        (codebook['selected'] == 1) &
        (codebook['domain'] == "Covariates")
    ])
    
    selected_continuous_covariates = list(codebook['variable_name'].loc[
        ((codebook['var_type'] == 'numeric') | (codebook['var_type'] == 'unknown') | (codebook['var_type'] == 'quintiles')) &
        (codebook[selection] == 1) &
        (codebook['domain'] == "Covariates")
    ])
    selected_categorical_covariates = list(codebook['variable_name'].loc[
        ((codebook['var_type'] == 'categorical') | (codebook['var_type'] == 'labels (0-1)') | (codebook['var_type'] == 'labels')
         | (codebook['var_type'] == 'factor')) &
        (codebook[selection] == 1) &
        (codebook['domain'] == "Covariates")
    ])
    
    print("selected continuous covariates:", selected_continuous_covariates)
    print("selected_categorical_covariates", selected_categorical_covariates)
    
    # get the whole covariate list
    covariates_list = list(codebook['variable_name'].loc[
        (codebook['domain'] == 'Covariates')
    ])
    
    unwanted_covariates = set(continuous_covariates + categorical_covariates) - set(selected_continuous_covariates + selected_categorical_covariates)    
    unwanted_covariates_ohupdated = get_one_hot_columns_list(unwanted_covariates, one_hot_columns)

    # deleted_cols = [col for col in features.columns if col in unwanted_covariates_ohupdated]
    features = features.drop(unwanted_covariates_ohupdated, axis=1)
    print("Dropped", len(unwanted_covariates_ohupdated), "covariates:", unwanted_covariates_ohupdated)
    return features


def delete_unwanted_PRS(codebook: pd.core.frame.DataFrame, features : pd.DataFrame, selection: str, one_hot_columns: list):
    prs_list = list(codebook['variable_name'].loc[
        ((codebook['domain'] == 'Polygenic_risk_score')) &
        (codebook['selected'] == 1)
    ])
    # print(prs_list)
    selected_prs_list = list(codebook['variable_name'].loc[
        ((codebook['domain'] == 'Polygenic_risk_score')) &
        (codebook['selected'] == 1) &
        (codebook[selection] == 1)
    ])
    # print(selected_prs_list)
    unwanted_prs = set(prs_list) - set(selected_prs_list)
    unwanted_prs_ohupdated = get_one_hot_columns_list(unwanted_prs, one_hot_columns)
    
    features = features.drop(unwanted_prs_ohupdated, axis=1)
    print("Dropped", len(unwanted_prs_ohupdated), "prs:", unwanted_prs_ohupdated)
    return features


def delete_unwanted_clinical_factors(codebook: pd.core.frame.DataFrame, features : pd.DataFrame, selection: str, one_hot_columns: list):
    clinical_factors_list = list(codebook['variable_name'].loc[
        ((codebook['domain'] == 'Clinical markers') | (codebook['domain'] == 'Health_outcomes') | (codebook['domain'] == 'Health_outcomes (parents)')) &
        (codebook['selected'] == 1)
    ])
    # print(clinical_factors_list)
    selected_clinical_factors_list = list(codebook['variable_name'].loc[
        ((codebook['domain'] == 'Clinical markers') | (codebook['domain'] == 'Health_outcomes') | (codebook['domain'] == 'Health_outcomes (parents)')) &
        (codebook[selection] == 1)
    ])
    # print(selected_clinical_factors_list)
    
    unwanted_clinical_factors = set(clinical_factors_list) - set(selected_clinical_factors_list)
    unwanted_clinical_factors_ohupdated = get_one_hot_columns_list(unwanted_clinical_factors, one_hot_columns)
    
    features = features.drop(unwanted_clinical_factors_ohupdated, axis=1, errors='ignore')
    print("Dropped", len(unwanted_clinical_factors_ohupdated), "clinical factors:", unwanted_clinical_factors_ohupdated)
    return features
    

def extract_selected_features(codebook: pd.DataFrame, features: pd.DataFrame, selection: str, one_hot_columns: list):
    features = delete_unwanted_covariates(codebook, features, selection, one_hot_columns)
    features = delete_unwanted_PRS(codebook, features, selection, one_hot_columns)
    features = delete_unwanted_clinical_factors(codebook, features, selection, one_hot_columns)
    return features


def get_one_hot_columns_list(features_list:list, one_hot_columns : list):
    features_list_ohupdated = []
    for col in features_list:
        added_cat = False
        for oh in one_hot_columns:
            if oh.startswith(col):
                features_list_ohupdated.append(oh)
                added_cat = True
        if added_cat==False:
            features_list_ohupdated.append(col)
    return features_list_ohupdated


def adjust_sample_size_to_outcome(features, outcome): 
    # Select only features with non null outcomes.
    print("Total number of missing values:", outcome.isna().sum())
    print("Using a feature dataframe of", len(features), "rows")
    print("Target column got", outcome.notnull().sum(), "non missing values")
    print("Adjusting sample size...")

    data = pd.merge(features, outcome, left_index=True, right_index=True, how='left')
    print("DF shape after merge:", data.shape)
    data = data.dropna(subset=[outcome.name])
    print("DF shape after droping NaNs", data.shape)
    
    target = data[outcome.name]
    assert(target.isna().sum() == 0) # assert non missing values in outcome column
    features = data.drop(outcome.name, axis=1)
    return features, target


def continuous_benchmark_cv(model, features, target, is_feature_selection, standardize=False, n_splits=10,
                            feature_importance:str=None,
                            results_section=None, verbose=False):
    if feature_importance is not None:
        importances_sets = []
    
    kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
    scores_r2, scores_rmse = [], []
    for i, (train_index, test_index) in enumerate(kf.split(features)):
        X_train, X_test = features.iloc[train_index].to_numpy(), features.iloc[test_index].to_numpy()
        y_train, y_test = target.iloc[train_index].to_numpy(), target.iloc[test_index].to_numpy()
        
        if standardize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test).reshape(-1,)
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = mean_squared_error(y_test, y_pred, squared=False)
        
        train_pred = model.predict(X_train)
        train_r2 = r2_score(y_train, train_pred)
        train_rmse = mean_squared_error(y_train, train_pred, squared=False)
        if verbose:
            print("Split {0:2d}".format(i))
            print("\tR2:   train={0:.4f} test={1:.4f}".format(train_r2, test_r2))
            print("\tRMSE: train={0:.4f} test={1:.4f}".format(train_rmse, test_rmse))
            
        scores_r2.append((train_r2, test_r2))
        scores_rmse.append((train_rmse, test_rmse))
        
        if feature_importance is not None:
            # data = None if feature_importance == 'embedded' else X_test
            if feature_importance != 'shap':
                importances = compute_embedded_feature_importance(model, data=X_test)
            else:
                explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
                shap_values = explainer(X_test)
                importances = [np.mean(abs(shap_values.values[:,i])) for i in range(len(shap_values.values[0]))]
                # family_shap_values = compute_family_shap(shap_values)
            
            importances_sets.append(importances)
    
    res_train_r2, res_train_r2_std = np.mean([e[0] for e in scores_r2]), np.std([e[0] for e in scores_r2])
    res_test_r2, res_test_r2_std = np.mean([e[1] for e in scores_r2]), np.std([e[1] for e in scores_r2])
    res_train_rmse, res_train_rmse_std = np.mean([e[0] for e in scores_rmse]), np.std([e[0] for e in scores_rmse])
    res_test_rmse, res_test_rmse_std = np.mean([e[1] for e in scores_rmse]), np.std([e[1] for e in scores_rmse])

    if results_section is not None: # save results to file:
        config = configparser.ConfigParser()
        config.read('./results.ini')
        if results_section not in config.sections():
            config[results_section] = {}
        if isinstance(model, TransformedTargetRegressor): model = model.regressor_
        describer="fs" if is_feature_selection else "all"
        config[results_section][f"{type(model).__name__.lower()}_cv_{describer}_train_R2"] = str([res_train_r2, res_train_r2_std])
        config[results_section][f"{type(model).__name__.lower()}_cv_{describer}_test_R2"] = str([res_test_r2, res_test_r2_std])
        config[results_section][f"{type(model).__name__.lower()}_cv_{describer}_train_RMSE"] = str([res_train_rmse, res_train_rmse_std])
        config[results_section][f"{type(model).__name__.lower()}_cv_{describer}_test_RMSE"] = str([res_test_rmse, res_test_rmse_std])
        with open('./results.ini', 'w+') as configfile:
            config.write(configfile)
    
    print("CV mean scores:")
    print("R2:  Train={0:.4f} +/- {1:.4f}. Test={2:.4f} +/- {3:.4f}".format(res_train_r2, res_train_r2_std,
                                                                            res_test_r2, res_test_r2_std))
    print("RMSE: Train={0:.4f} +/- {1:.4f}. Test={2:.4f} +/- {3:.4f}".format(res_train_rmse, res_train_rmse_std,
                                                                            res_test_rmse, res_test_rmse_std))
    
    if feature_importance is not None: # plot feature importance
        importances_means = np.array(importances_sets).mean(0)
        importances_std = np.array(importances_sets).std(0)
        plot_cvfeature_importance(importances_means, importances_std, features.columns, kmost=10)
        
        # save feature importance results
        if feature_importance == 'shap':
            results_path = os.path.join(os.path.dirname(pathlib.Path(__file__).parent.parent.resolve()), 'data/results/standard_fi.xlsx')
            save_feature_importances(results_path, importances_means, importances_std, features.columns, target.name, make_new=False)
         
    
def plot_cvfeature_importance(importances, importances_std, columns, kmost=20):
    indices = np.argsort(importances)[::-1]
    mapping_featurename_x_displayname = get_display_feature_mapping(columns)

    fig, ax = plt.subplots(figsize=(19,10))
    ind= range(kmost)
    x = importances[indices[0:kmost]][::-1]
    xerr = importances_std[indices[0:kmost]][::-1]
    labels = [mapping_featurename_x_displayname[key] for key in columns[indices[0:kmost]]][::-1]
    ax.set_title(f"Feature importances ({kmost} most)")
    ax.barh(ind, x, color="r", xerr=xerr, align="center")
    ax.set_yticks(ticks=range(kmost))
    ax.set_yticklabels(labels, rotation = 0, fontdict={'fontsize': 10})
    ax.set_ylim([-1, kmost])
    # add error values
    for k, y in enumerate(ind):
        ax.annotate("{0:.2f} +/- {1:.2f}".format(x[k], xerr[k]), (x[k] + xerr[k], y), textcoords='offset points',
                    xytext=(0, 3), ha='center', va='bottom', fontsize=10)
    plt.show()
    
def covariates_2steps_target_adjustement_cv(cov_model, res_model, covariate_list, features, target,
                                            standardize=False, feature_selection_reg=None, n_splits=10, verbose=False, save_results=False,
                                            feature_importance:str=None, show_residual_plots:bool=None, show_residual_histograms:bool=None):
    step1_scores_train, step1_scores_test = [], []
    step2_scores_train, step2_scores_test = [], []
    final_scores_train, final_scores_test = [], []
    final_train_residuals_array, final_test_residuals_array = [], []
    importances_sets, family_importances_sets, oh_adj_imp_sets, period_sets, vartype_sets = [], [], [], [], []
    selected_columns = [ele for ele in features.columns]
    
    codebook = pd.read_excel('../../../data/Helix data codebook.xlsx', na_values='NA')
    families = list(set(codebook['family'].loc[
        (codebook['selected'] == 1)
    ]))
    families.remove('ID')
    periods = ['Postnatal', 'Pregnancy']
    kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(features)):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]
        
        if standardize:
            # cat_variables = []
            # for var in X_train.columns:
            #     if var.endswith(".0"):
            #         cat_variables.append(var)
            cat_variables = [f"cohort_{i}.0" for i in range(1, 6)]
            scaler = StandardScaler()
            std_X_train = scaler.fit_transform(X_train)
            std_X_test = scaler.transform(X_test)
            std_X_train = pd.DataFrame(data=std_X_train, columns=[ele for ele in features.columns], index=X_train.index)
            std_X_test = pd.DataFrame(data=std_X_test, columns=[ele for ele in features.columns], index=X_test.index)
            std_X_train[cat_variables] = X_train[cat_variables]
            std_X_test[cat_variables] = X_test[cat_variables]
            X_train, X_test = std_X_train, std_X_test
            # display(X_train[cohorts].head())
            
        x_train_covariates, x_test_covariates = X_train[covariate_list], X_test[covariate_list] # data step 1
        
        # feature selection (EDITED)
        if feature_selection_reg is not None:
            lr_cov = LinearRegression()
            lasso_risk = Lasso(selection='random', alpha=feature_selection_reg, random_state=42, max_iter = 1000)
            lr_cov.fit(x_train_covariates, y_train)
            s1_y_pred_train = lr_cov.predict(x_train_covariates)
            residuals_train = y_train - s1_y_pred_train
            lasso_risk.fit(X_train.drop(["cohort_1.0", "cohort_2.0", "cohort_3.0", 'cohort_4.0', 'cohort_5.0'], axis=1, errors="ignore").to_numpy(), residuals_train)
            feature_selection = get_lasso_feature_selection(lasso_risk,
                                                            features.drop(["cohort_1.0", "cohort_2.0", "cohort_3.0", 'cohort_4.0', 'cohort_5.0'], axis=1).columns,
                                                            verbose=False)
            selected_columns = [ele for ele in feature_selection]
        
        x_train_no_cov = X_train[selected_columns].drop(["cohort_1.0", "cohort_2.0", "cohort_3.0", 'cohort_4.0', 'cohort_5.0'], axis=1, errors="ignore") # data step 2
        x_test_no_cov = X_test[selected_columns].drop(["cohort_1.0", "cohort_2.0", "cohort_3.0", 'cohort_4.0', 'cohort_5.0'], axis=1, errors="ignore") # data step 2
        updated_selected_cols = x_train_no_cov.columns
        
        
        cov_model.fit(x_train_covariates, y_train)
        s1_y_pred_train = cov_model.predict(x_train_covariates)
        s1_y_pred_test = cov_model.predict(x_test_covariates)

        score1_train = metrics.r2_score(y_train, s1_y_pred_train)
        score1_test = metrics.r2_score(y_test, s1_y_pred_test)
        step1_scores_train.append(score1_train)
        step1_scores_test.append(score1_test)
        score1_test_rmse = metrics.mean_squared_error(y_test, s1_y_pred_test, squared=False)
        
        
        residuals_train = y_train - s1_y_pred_train
        residuals_test = y_test - s1_y_pred_test

        res_model.fit(x_train_no_cov.to_numpy(), residuals_train)
        
        s2_y_pred_train = res_model.predict(x_train_no_cov.to_numpy())
        s2_y_pred_test = res_model.predict(x_test_no_cov.to_numpy())
        
        score2_train = metrics.r2_score(residuals_train, s2_y_pred_train)
        score2_test = metrics.r2_score(residuals_test, s2_y_pred_test)
        step2_scores_train.append(score2_train)
        step2_scores_test.append(score2_test)
        score2_test_rmse = metrics.mean_squared_error(residuals_test, s2_y_pred_test, squared=False)
        
        final_pred_train = s1_y_pred_train + s2_y_pred_train
        final_pred_test = s1_y_pred_test + s2_y_pred_test
        final_score_train = metrics.r2_score(y_train, final_pred_train)
        final_score_test = metrics.r2_score(y_test, final_pred_test)
        final_scores_train.append(final_score_train)
        final_scores_test.append(final_score_test)
        final_scores_test_rmse = metrics.mean_squared_error(y_test, final_pred_test, squared=False)
        
        # residual plots:
        final_train_residuals = y_train - final_pred_train
        final_test_residuals = y_test - final_pred_test
        final_train_residuals_array.append((y_train - final_pred_train, y_train))
        final_test_residuals_array.append((y_test - final_pred_test, y_test))
#         fig, ax = plt.subplots(1, 2, figsize=(12, 3), sharey=True)
#         fig.suptitle("Residual plots")
#         ax[0].set_ylabel("Predicted values")
#         ax[0].set_xlabel("Training residuals")
#         # ax[0].set_xlim(-10, 10)
#         # ax[0].set_ylim(-10, 10)
#         ax[1].set_xlabel("Residuals on test set")
#         ax[0].scatter(final_train_residuals, y_train, alpha=0.5)
#         ax[1].scatter(final_test_residuals, y_test, alpha=0.5)
#         plt.show()
#         print(f"Mean of residuals on train set: {round(final_train_residuals.mean(), 2)}, On test set {round(final_test_residuals.mean(), 2)}")
#         print(f"Std of residuals on train set: {round(final_train_residuals.std(), 2)}, On test set {round(final_test_residuals.std(), 2)}")
#         print(shapiro(final_train_residuals).pvalue>0.05)
#         print(shapiro(final_test_residuals).pvalue>0.05)
        
#         # residual histograms
#         fig, ax = plt.subplots(1, 2, figsize=(12, 3), sharey=True)
#         fig.suptitle("Residual Histograms")
#         ax[0].hist(final_train_residuals, bins='auto')
#         ax[1].hist(final_test_residuals, bins='auto')
#         ax[0].title.set_text('Residuals distribution on training set')
#         ax[1].title.set_text('Residuals distribution on testing set')
#         plt.show()
        
        if feature_importance is not None:
            if feature_importance != 'shap':
                importances = compute_embedded_feature_importance(res_model, data = X_test[updated_selected_cols])
            else:
                if isinstance(res_model, Lasso):
                    explainer = shap.LinearExplainer(res_model, x_test_no_cov)
                else:
                    explainer = shap.TreeExplainer(res_model, feature_perturbation='tree_path_dependent')
                
                shap_values = explainer(x_test_no_cov)
                # oh_adjusted_shap_values = compute_oh_adjusted_shap(shap_values)
                family_shap_values = compute_family_shap(shap_values)
                pppreg_shap_values = compute_period_shap(shap_values)
                vartype_shap_values = compute_vartype_shap(shap_values)
                
                # importances = [np.mean(abs(shap_values.values[:,i])) for i in range(len(shap_values.values[0]))]
                selected_importances = [np.mean(abs(shap_values.values[:,i])) for i in range(len(shap_values.values[0]))]

                c = features.drop(["cohort_1.0", "cohort_2.0", "cohort_3.0", 'cohort_4.0', 'cohort_5.0'], axis=1, errors="ignore").columns
                importances = []
                for i, col in enumerate(c):
                    if col in updated_selected_cols:
                        importances.append(selected_importances[list(updated_selected_cols).index(col)])
                    else:
                        importances.append(0)
                
                # familly importances
                # oh_adj_importances = [np.mean(abs(oh_adjusted_shap_values.values[:,i])) for i in range(len(oh_adjusted_shap_values.values[0]))]
                selected_fam_importances = [np.mean(abs(family_shap_values.values[:,i])) for i in range(len(family_shap_values.values[0]))]
                fam_importances = []
                for col in families:
                    if col in family_shap_values.feature_names:
                        fam_importances.append(selected_fam_importances[family_shap_values.feature_names.index(col)])
                    else:
                        fam_importances.append(0)
                
                # period importances
                selected_period_importances = [np.mean(abs(pppreg_shap_values.values[:,i])) for i in range(len(pppreg_shap_values.values[0]))]
                period_importances = []
                for col in periods:
                    if col in pppreg_shap_values.feature_names:
                        period_importances.append(selected_period_importances[pppreg_shap_values.feature_names.index(col)])
                    else:
                        period_importances.append(0)
                        
                # vartypes importances        
                selected_vartype_importances = [np.mean(abs(vartype_shap_values.values[:,i])) for i in range(len(vartype_shap_values.values[0]))]
                variables_importances = []
                for col in ['Clinical factors', 'Metabolites/Proteins', 'Exposures', 'Covariates']:
                    if col in vartype_shap_values.feature_names:
                        variables_importances.append(selected_vartype_importances[vartype_shap_values.feature_names.index(col)])
                    else:
                        variables_importances.append(0)
            
            importances_sets.append(importances)
            family_importances_sets.append(fam_importances)
            # oh_adj_imp_sets.append(oh_adj_importances)
            period_sets.append(period_importances)
            vartype_sets.append(variables_importances)
            
        
        if verbose:
            print(f"{i+1} split:")
            print("\t step1 TRAIN r2: {0:.2f}".format(score1_train))
            print("\t step1 test r2: {0:.2f}".format(score1_test))
            print("\t step2 TRAIN r2: {0:.2f}".format(score2_train))
            print("\t step2 test r2: {0:.2f}".format(score2_test))
            print("\t step2 TRAIN r2: {0:.2f}".format(final_scores_train))
            print("\t step2 test r2: {0:.2f}".format(final_scores_test))
            
    mean_step1_train_r2, std_step1_train_r2 = np.mean(step1_scores_train), np.std(step1_scores_train)
    mean_step1_test_r2, std_step1_test_r2 = np.mean(step1_scores_test), np.std(step1_scores_test)
    mean_step2_train_r2, std_step2_train_r2 = np.mean(step2_scores_train), np.std(step2_scores_train)
    mean_step2_test_r2, std_step2_test_r2 = np.mean(step2_scores_test), np.std(step2_scores_test)
    mean_final_train_r2, std_final_train_r2 = np.mean(final_scores_train), np.std(final_scores_train)
    mean_final_test_r2, std_final_test_r2 = np.mean(final_scores_test), np.std(final_scores_test)
    if save_results:
        config = configparser.ConfigParser()
        config.read('./results.ini')
        if 'COVARIATES' not in config.sections():
            config['COVARIATES'] = {}
        config['COVARIATES'][f"step1_{type(cov_model).__name__.lower()}_cv_train_R2"] = str([mean_step1_train_r2, std_step1_train_r2])
        config['COVARIATES'][f"step1_{type(cov_model).__name__.lower()}_cv_test_R2"] = str([mean_step1_test_r2, std_step1_test_r2])
        # print(f"step1_{type(cov_model).__name__.lower()}_cv_test_R2")
        if '2STEPS' not in config.sections():
            config['2STEPS'] = {}
        describer="all" if feature_selection_reg is None else "fs"
        config['2STEPS'][f"step2_{type(res_model).__name__.lower()}_cv_{describer}_train_R2"] = str([mean_step2_train_r2, std_step2_train_r2])
        config['2STEPS'][f"step2_{type(res_model).__name__.lower()}_cv_{describer}_test_R2"] = str([mean_step2_test_r2, std_step2_test_r2])
        config['2STEPS'][f"final_{type(res_model).__name__.lower()}_cv_{describer}_train_R2"] = str([mean_final_train_r2, std_final_train_r2])
        config['2STEPS'][f"final_{type(res_model).__name__.lower()}_cv_{describer}_test_R2"] = str([mean_final_test_r2, std_final_test_r2])
        with open('./results.ini', 'w+') as configfile:
            config.write(configfile)
            
    if show_residual_plots:
        fig, ax = plt.subplots(n_splits, 2, figsize=(10, 1.5*n_splits), sharey=True)
        fig.suptitle("Residual plots")
        ax[0, 0].text(0.5, 1.05, 'Residuals on training set', ha='center', va='bottom', transform=ax[0, 0].transAxes)
        ax[0, 1].text(0.5, 1.05, 'Residuals on test set', ha='center', va='bottom', transform=ax[0, 1].transAxes)
        for i in range(n_splits):
            ax[i, 0].set_ylabel("Predicted values")
            ax[i, 0].set_xlabel(f"Shapiro-Wilk test pvalue {round(shapiro(final_train_residuals_array[i][0]).pvalue, 3)}")
            ax[i, 1].set_xlabel(f"Shapiro-Wilk test pvalue {round(shapiro(final_test_residuals_array[i][0]).pvalue, 3)}")
            ax[i, 0].scatter(final_train_residuals_array[i][0], final_train_residuals_array[i][1], alpha=0.5)
            ax[i, 1].scatter(final_test_residuals_array[i][0], final_test_residuals_array[i][1], alpha=0.5)
        plt.tight_layout()
        plt.show()
        
    count_training_shapiros = 0
    count_testing_shapiros = 0
    for i in range(n_splits):
        count_training_shapiros += 1 if shapiro(final_train_residuals_array[i][0]).pvalue > 0.05 else 0
        count_testing_shapiros += 1 if shapiro(final_test_residuals_array[i][0]).pvalue > 0.05 else 0
    # print(f"Number of normaly distributed rediduals (Shapiro-Wild) on training set: {count_training_shapiros}")
    print(f"Number of normaly distributed rediduals (Shapiro-Wild) on testing set: {count_testing_shapiros}, means: {np.mean([np.mean(res) for res in final_test_residuals_array])}")
        
        
    if show_residual_histograms:
        fig, ax = plt.subplots(n_splits, 2, figsize=(10, 1.5*n_splits), sharey=True)
        fig.suptitle("Residual Histograms")
        for i in range(n_splits):
            ax[i, 0].hist(final_train_residuals_array[i][0], bins='auto')
            ax[i, 1].hist(final_test_residuals_array[i][0], bins='auto')
            ax[i, 0].title.set_text('Residuals distribution on training set')
            ax[i, 1].title.set_text('Residuals distribution on testing set')
        plt.tight_layout()
        plt.show()
    
    if feature_importance is None:
        print(f"Results:")
        print("\tstep 1 mean train score: {0:.3f} std:{1:.2f}".format(mean_step1_train_r2, std_step1_train_r2))
        print("\tstep 1 mean test score: {0:.3f} std:{1:.2f}".format(mean_step1_test_r2, std_step1_test_r2))
        print("\tstep 2 mean train score: {0:.3f} std:{1:.2f}".format(mean_step2_train_r2, std_step2_train_r2))
        print("\tstep 2 mean test score: {0:.3f} std:{1:.2f}".format(mean_step2_test_r2, std_step2_test_r2))
        print("\tfinal mean train score: {0:.3f} std:{1:.2f}".format(mean_final_train_r2, std_final_train_r2))
        print("\tfinal mean test score: {0:.3f} std:{1:.2f}".format(mean_final_test_r2, std_final_test_r2))
    
    if feature_importance is not None: # plot feature importance  
        importances_means = np.array(importances_sets).mean(0)
        importances_std = np.array(importances_sets).std(0)
        
        # fig = plt.figure(figsize=(15,5))
        # ax1 = plt.subplot(111)
        
        indices = np.argsort(importances_means)[::-1]
        # plt.plot(np.cumsum(importances_means[indices]))
        # plt.plot()
        
        # plot_cvfeature_importance(importances_means, importances_std, pd.Index(updated_selected_cols), kmost=20)
        plot_cvfeature_importance(importances_means, importances_std, pd.Index(c), kmost=20)
        
        # save feature importances
        if feature_importance == 'shap':
            # ohadj_importances_means = np.array(oh_adj_imp_sets).mean(0)
            # ohadj_importances_std = np.array(oh_adj_imp_sets).std(0)
            # plot_cvfeature_importance(ohadj_importances_means, ohadj_importances_std, pd.Index(oh_adjusted_shap_values.feature_names), kmost=20)
            
            # plot family shap
            family_importances_means = np.array(family_importances_sets).mean(0)
            family_importances_std = np.array(family_importances_sets).std(0)
            # plot_cvfeature_importance(family_importances_means, family_importances_std, pd.Index(family_shap_values.feature_names), kmost=20)
            indices = np.argsort(family_importances_means)[::-1]
            fig, ax = plt.subplots(figsize=(19,10))
            kmost = 20
            x = family_importances_means[indices[0:kmost]][::-1]
            xerr = family_importances_std[indices[0:kmost]][::-1]
            kmost = min(20, len(x))
            ind= range(kmost)
            # labels = [key for key in pd.Index(family_shap_values.feature_names)[indices[0:kmost]]][::-1]
            labels = [key for key in pd.Index(families)[indices[0:kmost]]][::-1]
            ax.set_title(f"Family wise (SHAP) feature importances ({kmost} most)")
            ax.barh(ind, x, color="r", xerr=xerr, align="center")
            ax.set_yticks(ticks=range(kmost))
            ax.set_yticklabels(labels, rotation = 0, fontdict={'fontsize': 10})
            ax.set_ylim([-1, kmost])
            # add error values
            for k, y in enumerate(ind):
                ax.annotate("{0:.2f} +/- {1:.2f}".format(x[k], xerr[k]), (x[k] + xerr[k], y), textcoords='offset points',
                            xytext=(0, 3), ha='center', va='bottom', fontsize=10)
            plt.show()
            
            # plot period shap importance
            period_importances_means = np.array(period_sets).mean(0)
            period_importances_std = np.array(period_sets).std(0)
            indices = np.argsort(period_importances_means)[::-1]
            fig, ax = plt.subplots(figsize=(19,2))
            kmost = 20
            x = period_importances_means[indices[0:kmost]][::-1]
            xerr = period_importances_std[indices[0:kmost]][::-1]
            kmost = min(20, len(x))
            ind= range(kmost)
            labels = [key for key in pd.Index(periods)[indices[0:kmost]]][::-1]
            ax.set_title(f"Period wise (SHAP) feature importance")
            ax.barh(ind, x, color="r", xerr=xerr, align="center")
            ax.set_yticks(ticks=range(kmost))
            ax.set_yticklabels(labels, rotation = 0, fontdict={'fontsize': 10})
            ax.set_ylim([-1, kmost])
            # add error values
            for k, y in enumerate(ind):
                ax.annotate("{0:.2f} +/- {1:.2f}".format(x[k], xerr[k]), (x[k] + xerr[k], y), textcoords='offset points',
                            xytext=(0, 3), ha='center', va='bottom', fontsize=10)
            plt.show()
            
            # plot vartype shap importance
            vartype_importances_means = np.array(vartype_sets).mean(0)
            vartype_importances_std = np.array(vartype_sets).std(0)
            indices = np.argsort(vartype_importances_means)[::-1]
            fig, ax = plt.subplots(figsize=(19,2))
            kmost = 20
            x = vartype_importances_means[indices[0:kmost]][::-1]
            xerr = vartype_importances_std[indices[0:kmost]][::-1]
            kmost = min(20, len(x))
            ind= range(kmost)
            vartypes = ['Clinical factors (parents and child)', 'Metabolites/Proteins (child)', 'Exposures (mother and child)', 'Covariates (mother and child)']
            labels = [key for key in pd.Index(vartypes)[indices[0:kmost]]][::-1]
            ax.set_title(f"Variable types' wise (SHAP) feature importance")
            ax.barh(ind, x, color="r", xerr=xerr, align="center")
            ax.set_yticks(ticks=range(kmost))
            ax.set_yticklabels(labels, rotation = 0, fontdict={'fontsize': 10})
            ax.set_ylim([-1, kmost])
            # add error values
            for k, y in enumerate(ind):
                ax.annotate("{0:.2f} +/- {1:.2f}".format(x[k], xerr[k]), (x[k] + xerr[k], y), textcoords='offset points',
                            xytext=(0, 3), ha='center', va='bottom', fontsize=10)
            plt.show()
            
            if isinstance(res_model, Lasso):
                results_path = os.path.join(os.path.dirname(pathlib.Path(__file__).parent.parent.resolve()), 'data/results/2steps_fi_lasso.xlsx')
                fam_results_path = os.path.join(os.path.dirname(pathlib.Path(__file__).parent.parent.resolve()), 'data/results/family_2steps_fi_lasso.xlsx')
                period_results_path = os.path.join(os.path.dirname(pathlib.Path(__file__).parent.parent.resolve()), 'data/results/period_2steps_fi_lasso.xlsx')
                vartype_results_path = os.path.join(os.path.dirname(pathlib.Path(__file__).parent.parent.resolve()), 'data/results/vartype_2steps_fi_lasso.xlsx')
            else:
                results_path = os.path.join(os.path.dirname(pathlib.Path(__file__).parent.parent.resolve()), 'data/results/2steps_fi.xlsx')
                fam_results_path = os.path.join(os.path.dirname(pathlib.Path(__file__).parent.parent.resolve()), 'data/results/family_2steps_fi.xlsx')
                period_results_path = os.path.join(os.path.dirname(pathlib.Path(__file__).parent.parent.resolve()), 'data/results/period_2steps_fi.xlsx')
                vartype_results_path = os.path.join(os.path.dirname(pathlib.Path(__file__).parent.parent.resolve()), 'data/results/vartype_2steps_fi.xlsx')
            
            save_feature_importances(results_path,
                                     importances_means,
                                     importances_std,
                                     c,
                                     target.name,
                                     make_new=False)
            save_family_feature_importances(fam_results_path,
                                            family_importances_means,
                                            family_importances_std,
                                            families,
                                            target.name,
                                            make_new=False)
            
            save_family_feature_importances(period_results_path,
                                            period_importances_means,
                                            period_importances_std,
                                            periods,
                                            target.name,
                                            make_new=False)
            
            save_family_feature_importances(vartype_results_path,
                                            vartype_importances_means,
                                            vartype_importances_std,
                                            vartypes,
                                            target.name,
                                            make_new=False)
            
    return final_scores_test
            

    
def compute_covariates_variance_explained(covariates, target, cov_name, n_splits=10, save_results=False):
    lr = LinearRegression()
    scaler = StandardScaler()
    covariates = scaler.fit_transform(covariates)
    kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
    scores = cross_val_score(lr, covariates, target, cv=kf, scoring='r2', verbose=0, n_jobs=n_splits)
    print(cov_name,"explain %0.4f/100 variance with a standard deviation of %0.4f" % (scores.mean()*100, scores.std()*100))
    
    if save_results:
        config = configparser.ConfigParser()
        config.read('./results.ini')
        if 'COVARIATES' not in config.sections():
            config['COVARIATES'] = {}
        config['COVARIATES'][f"{cov_name}_cv_r2"] = str([scores.mean(), scores.std()])
        with open('./results.ini', 'w+') as configfile:
            config.write(configfile)
    return scores


def fit_skmodel(model, X_train, y_train, X_test, y_test):
    if isinstance(model, XGBRegressor):
        model.fit(X_train, y_train, 
                  # eval_metric='gamma-deviance', 
                  eval_metric='rmse', 
                  eval_set=[(X_test, y_test)],
                  early_stopping_rounds = 20, 
                  verbose = False)
    else:
        model.fit(X_train, y_train)

        
def compute_embedded_feature_importance(model, data):
    if isinstance(model, RandomForestRegressor):
        importances = model.feature_importances_
    if isinstance(model, XGBRegressor):
        # importances = list(model.get_booster().get_score(importance_type="gain").values())
        importances_dict = model.get_booster().get_score(importance_type='gain')
        importances = []
        # print(importances_dict)
        for i in range(data.shape[1]):
            key = 'f' + str(i)
            if key in importances_dict:
                importances.append(importances_dict[key])
            else:
                importances.append(0.0)
        # print(importances)
    else:
        print('WARNING: Unknown model instance type, must be RandomForestRegressor or XGBRegressor. Returning null.')
        importances = None
    return importances
        

def plot_training_summary(model, X_test, test_predictions, y_test, model_name: str):
    p1 = math.floor(min(min(test_predictions), min(y_test)))
    p2 = math.ceil(max(max(test_predictions), max(y_test)))
    if isinstance(model, RandomForestRegressor) or isinstance(model, XGBRegressor):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19,6), gridspec_kw={"width_ratios" : [1,2]})
        fig.suptitle(f"{model_name}: {y_test.name}")

        sns.regplot(x=y_test, y=test_predictions, ax=ax1, truncate=False, scatter=True, fit_reg=True, marker="")
        ax1.scatter(y_test, test_predictions, c='crimson', s=8)
        ax1.plot([p1, p2], [p1, p2], '--k')
        ax1.set_title("Actual vs predicted")
        ax1.set_xlim([p1, p2])
        ax1.set_ylim([p1, p2])
        ax1.set_xlabel("True values")
        ax1.set_ylabel("Predictions")
        ax1.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

        # Plot feature importance
        mapping_featurename_x_displayname = get_display_feature_mapping(X_test.columns)
        if isinstance(model, RandomForestRegressor):
            importances = model.feature_importances_
            std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
            indices = np.argsort(importances)[::-1]

            # Print the feature rankingµ
            # print("\nFeature ranking:")
            # for f in range(5):
            #     print("%d. feature %s (%f)" % (f + 1, X_train.columns[indices[f]], importances[indices[f]]))
            # print("...")

            # Plot the impurity-based feature importances of the forest
            ax2.set_title("Feature importances (10 most)")
            ax2.bar(range(10), importances[indices[0:10]],
                    color="r", yerr=std[indices[0:10]], align="center")
            ax2.set_xticks(ticks=range(10))
            labels = [mapping_featurename_x_displayname[key] for key in X_test.columns[indices[0:10]]]
            ax2.set_xticklabels(labels, rotation = 45, fontdict={'fontsize': 10})
            ax2.set_xlim([-1, 10])
            for container in ax2.containers:
                if isinstance(container, matplotlib.container.ErrorbarContainer): continue
                ax2.bar_label(container, labels=[f'{x:.2f}' for x in container.datavalues])
            plt.show()

        if isinstance(model, XGBRegressor):
            # feat_imp = pd.Series(model.get_booster().get_fscore()).sort_values(ascending=False)
            feat_imp = pd.Series(model.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
            feat_imp = feat_imp[0:10]
            labels = [mapping_featurename_x_displayname[key] for key in feat_imp.index]
            feat_imp.set_axis(labels, inplace=True)
            feat_imp.plot(kind='bar', title='Feature Importances', ax=ax2, rot=45, colormap='flag')
            for container in ax2.containers:
                ax2.bar_label(container, labels=[f'{x:.2f}' for x in container.datavalues])
            ax2.set_ylabel('Feature importance f-score')
    else:
        plt.figure(figsize=(6,6))
        plt.xlim(p1,p2)
        plt.ylim(p1,p2)
        sns.regplot(x=y_test, y=test_predictions, truncate=False)
        plt.scatter(y_test, test_predictions, c='crimson')
        plt.plot([p1, p2], [p1, p2], 'b-')
        plt.xlabel('True Values', fontsize=15)
        plt.ylabel('Predictions', fontsize=15)
        plt.axis('equal')
        plt.show()
    
    
def fit_regr_skmodel(model, X_train, X_test, y_train, y_test, model_name:str, standardize: bool, verbose:bool=False):
    if standardize:
        columns = X_train.columns
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_train = pd.DataFrame(data=X_train, columns=columns)
        X_test = pd.DataFrame(data=X_test, columns=columns)
    fit_skmodel(model, X_train, y_train, X_test, y_test)
    test_predictions = model.predict(X_test)
    train_predictions = model.predict(X_train)
    # print(np.mean(predictions))
    
    # print(type(model))
    print('Coefficient of determination r2: %.3f'% r2_score(y_test, test_predictions))
    print('Root Mean Squared Error: %.4f'% mean_squared_error(y_test, test_predictions, squared=False))
    if verbose:
        print('\nTraining R2: %.3f'% r2_score(y_train, train_predictions))
        print('Training RMSE: %.4f'% mean_squared_error(y_train, train_predictions, squared=False))
        plot_training_summary(model, X_test, test_predictions, y_test, model_name=model_name)
        
            
def fit_2step_regr_skmodel(cov_model, res_model, covariates: list, X_train, X_test, y_train, y_test, model_name : str, standardize:bool,
                           selected_features: list=None, verbose=False):
    selected_columns = [ele for ele in X_train.columns]
    if selected_features is not None:
        selected_columns = [ele for ele in selected_features]
    if standardize:
        cat_variables = []
        for var in X_train.columns:
            if var.endswith(".0"):
                cat_variables.append(var)
        # cat_variables = [f"cohort_{i}.0" for i in range(1, 6)]
        scaler = StandardScaler()
        std_X_train = scaler.fit_transform(X_train)
        std_X_test = scaler.transform(X_test)
        std_X_train = pd.DataFrame(data=std_X_train, columns=[ele for ele in features.columns], index=X_train.index)
        std_X_test = pd.DataFrame(data=std_X_test, columns=[ele for ele in features.columns], index=X_test.index)
        std_X_train[cat_variables] = X_train[cat_variables]
        std_X_test[cat_variables] = X_test[cat_variables]
        X_train, X_test = std_X_train, std_X_test
    
    # step 1
    train_covariates, test_covariates = X_train[covariates], X_test[covariates]
    x_train_no_cov = X_train[selected_columns].drop(["cohort_1.0", "cohort_2.0", "cohort_3.0", 'cohort_4.0', 'cohort_5.0'], axis=1, errors="ignore") # data step 2
    x_test_no_cov = X_test[selected_columns].drop(["cohort_1.0", "cohort_2.0", "cohort_3.0", 'cohort_4.0', 'cohort_5.0'], axis=1, errors="ignore") # data step 2
    updated_selected_cols = x_train_no_cov.columns
    
    fit_skmodel(cov_model, train_covariates, y_train, test_covariates, y_test)    
    test_predictions = cov_model.predict(test_covariates)
    train_predictions = cov_model.predict(train_covariates)

    test_r2_covmodel, test_rmse_covmodel = r2_score(y_test, test_predictions), mean_squared_error(y_test, test_predictions, squared=False)
    train_r2_covmodel, train_rmse_covmodel = r2_score(y_train, train_predictions), mean_squared_error(y_train, train_predictions, squared=False)
    residuals_test = y_test - test_predictions
    residuals_train = y_train - train_predictions
    
    # step 2
    fit_skmodel(res_model, X_train=x_train_no_cov, y_train=residuals_train, X_test=x_test_no_cov, y_test=residuals_test)
    test_predictions = res_model.predict(x_test_no_cov)
    train_predictions = res_model.predict(x_train_no_cov)
    
    test_r2_resmodel, test_rmse_resmodel = r2_score(residuals_test, test_predictions), mean_squared_error(residuals_test, test_predictions, squared=False)
    train_r2_resmodel, train_rmse_resmodel = r2_score(residuals_train, train_predictions), mean_squared_error(residuals_train, train_predictions, squared=False)
    print("step1 TRAIN r2: {0:+.3f} - TEST r2: {1:+.3f} \t TRAIN rmse: {2:+.3f} - TEST rmse: {3:+.3f}".format(train_r2_covmodel, test_r2_covmodel,
                                                                                                           train_rmse_covmodel, test_rmse_covmodel))
    print("step2 TRAIN r2: {0:+.3f} - TEST r2: {1:+.3f} \t TRAIN rmse: {2:+.3f} - TEST rmse: {3:+.3f}".format(train_r2_resmodel, test_r2_resmodel,
                                                                                                           train_rmse_resmodel, test_rmse_resmodel))
    if verbose:
        plot_training_summary(res_model, x_test_no_cov, test_predictions, residuals_test, model_name=model_name)
    
def fit_2s(cov_model, res_model, covariates: list, features, target, selected_features: list=None, standardize:bool=False, verbose=False):
    selected_columns = [ele for ele in features.columns]
    if selected_features is not None:
        selected_columns = [ele for ele in selected_features]
    if standardize:
        cat_variables = []
        for var in features.columns:
            if var.endswith(".0"):
                cat_variables.append(var)
        # cat_variables = [f"cohort_{i}.0" for i in range(1, 6)]
        scaler = StandardScaler()
        std_X_train = scaler.fit_transform(features)
        std_X_train = pd.DataFrame(data=std_X_train, columns=[ele for ele in features.columns], index=features.index)
        std_X_train[cat_variables] = features[cat_variables]
        features = std_X_train
        
    s1_features = features[covariates]
    s2_features = features[selected_columns].drop(covariates, axis=1, errors="ignore") # data step 2
    # updated_selected_cols = x_train_no_cov.columns
    
    # first step
    cov_model.fit(s1_features, target)
    y_pred1 = cov_model.predict(s1_features)
    # train_predictions = cov_model.predict(train_covariates)
    
    # second step
    residuals = target - y_pred1
    res_model.fit(s2_features, residuals)
    y_pred2 = res_model.predict(s2_features)
    y_pred_final = y_pred1 + y_pred2
    
    # perfs
    if verbose:
        s1_r2 = r2_score(target, y_pred1)
        s2_r2 = r2_score(residuals, y_pred2)
        final_r2 = r2_score(target, y_pred_final)
        print("step1 r2: {0:.3f} - step2 r2: {1:.3f} - final r2: {2:.3f}".format(s1_r2, s2_r2, final_r2))
        
        plot_training_summary(res_model, s2_features, y_pred2, residuals, model_name=None)
    
    return y_pred_final


def predict_2s(cov_model, res_model, covariates: list, features, target, verbose=False):
    s1_features = features[covariates]
    s2_features = features.drop(covariates, axis=1, errors="ignore")
    
    y_pred1 = cov_model.predict(s1_features)
    y_pred2 = res_model.predict(s2_features)
    
    return y_pred_final


def compute_lasso_feature_selection(lasso, features, target, verbose=False, two_steps=False): # train a lasso and extract feature selection
    columns = features.columns
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features = pd.DataFrame(data=features, columns=columns)
    
    if two_steps:
        lr_cov = LinearRegression()
        lr_cov.fit(x_train_covariates, y_train)
        covariate_list = ["cohort_1.0", "cohort_2.0", "cohort_3.0", 'cohort_4.0', 'cohort_5.0']
        x_covariates = features[covariate_list] # data step 1
        s1_y_pred = lr_cov.predict(x_covariates)
        residuals = target - s1_y_pred
        lasso.fit(features.drop(covariate_list, axis=1, errors="ignore").to_numpy(), residuals)
    else:     
        lasso.fit(features, target)
    
    
    if isinstance(lasso, TransformedTargetRegressor): lasso = lasso.regressor_
    results = pd.DataFrame(lasso.coef_.reshape(-1), index=features.columns).sort_values(by=0, ascending=False).transpose()
    for col in results.columns:
        if results[col].iloc[0] == 0:
            results.drop(col, axis=1, inplace=True)
    display(results)
    
    if verbose:
        coeffs = dict()
        for i, col in enumerate(features.columns):
            if lasso.coef_.reshape(-1)[i] != 0:
                coeffs[col] = lasso.coef_.reshape(-1)[i]

        coeffs = dict(sorted(coeffs.items(), key=lambda item: abs(item[1]), reverse=True))
        for keys in coeffs:
            print("{0:20}: {1:.3f}".format(keys, coeffs[keys]))
    return results.columns

def get_lasso_feature_selection(trained_lasso, columns, verbose=False): # model is pretrained
    if isinstance(trained_lasso, TransformedTargetRegressor): trained_lasso = trained_lasso.regressor_
    results = pd.DataFrame(trained_lasso.coef_.reshape(-1), index=columns).sort_values(by=0, ascending=False).transpose()
    for col in results.columns:
        if results[col].iloc[0] == 0:
            results.drop(col, axis=1, inplace=True)
    
    if verbose:
        coeffs = dict()
        for i, col in enumerate(columns):
            if trained_lasso.coef_.reshape(-1)[i] != 0:
                coeffs[col] = trained_lasso.coef_.reshape(-1)[i]

        coeffs = dict(sorted(coeffs.items(), key=lambda item: abs(item[1]), reverse=True))
        for keys in coeffs:
            print("{0:20}: {1:.3f}".format(keys, coeffs[keys]))
    return results.columns
    
        
def plot_dashboard(model, X_train, X_test, y_train, y_test):
    y_pred = model.predict(X_test)
    
    f, (ax0, ax1) = plt.subplots(2, 3, figsize=(10, 8))
    
    ax0[0].scatter(y_test, y_pred, s=8)
    
    p1 = max(max(y_pred), max(y_test))
    p2 = min(min(y_pred), min(y_test))
    # ax0[0].plot([p1, p2], [p1, p2], "--k")
    
    ax0[0].plot([-2, 5], [-2, 5], "--k")
    ax0[0].set_ylabel("Predicted target")
    ax0[0].set_xlabel("True target")
    ax0[0].text(
        s="Actual vs predicted",
        x=-1,
        y=6,
        fontsize=12,
        multialignment="center",
    )
    ax0[0].text(
        -1.5,
        4.5,
        r"$R^2$=%.2f, MAE=%.2f"
        % (r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred)),
    )
    ax0[0].set_xlim([-2, 5])
    ax0[0].set_ylim([-2, 5])
    ax0[0].ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

    # ax1[0].scatter(y_test, (y_pred - y_test), s=8)
    # ax1[0].set_ylabel("Residual")
    # ax1[0].set_xlabel("True target")
    # ax1[0].ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    
    ax0[1].scatter(y_pred, (y_test - y_pred), s=8)
    ax0[1].set_ylabel("Residual")
    ax0[1].set_xlabel("Predicted target")
    ax0[1].ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    
    ax1[1].scatter(y_test, (y_test - y_pred), s=8)
    ax1[1].set_ylabel("Residual")
    ax1[1].set_xlabel("True target")
    ax1[1].ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    
    ratio_MF = X_test["e3_sex"]
    bins, edges = pd.cut(y_pred, bins=10, retbins=True, labels=False)
    groups = X_test.groupby(['e3_sex', bins])
    male, female = [], []
    # for name, group in groups:
    #     print(name)
    for i in range(10):
        try:
            male.append(
                len(groups.get_group((1, i))['e3_sex'])
            )
        except:
            male.append(0)
    for i in range(10):
        try:
            female.append(
                len(groups.get_group((0, i))['e3_sex'])
            )
        except:
            female.append(0)
    res = [male[i]-female[i] for i in range(len(male))]
    
    
    # ax0[2].scatter(y_pred, , s=8)
    ax0[2].scatter(edges[1:], res, s=8)
    ax0[2].set_ylabel("sex diff M-F")
    ax0[2].set_xlabel("Predicted target")
    ax0[2].ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    
    bins, edges = pd.cut(y_test, bins=10, retbins=True, labels=False)
    # print(edges)
    groups = X_test.groupby(['e3_sex', bins])
    male, female = [], []
    # for name, group in groups:
    #     print(name)
    for i in range(10):
        try:
            male.append(
                len(groups.get_group((1, i))['e3_sex'])
            )
        except:
            male.append(0)
    for i in range(10):
        try:
            female.append(
                len(groups.get_group((0, i))['e3_sex'])
            )
        except:
            female.append(0)
    # print(male, female)
    res = [male[i]-female[i] for i in range(len(male))]
    # print(res)
    # print(len(groups.get_group((0, 0))['e3_sex']))
    
    # ratio_MF = 
    ax1[2].scatter(edges[1:], res, s=8)
    ax1[2].set_ylabel("sex diff M-F")
    ax1[2].set_xlabel("True target")
    ax1[2].ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    
    f.suptitle("HELIX data: p-factor", y=0.035)
    plt.show()

    
def extract_one_hot_column_names(column_list):
    one_hot_cols = []
    for col in column_list:
        if col.endswith('.0') and not col.startswith('log'):
            one_hot_cols.append(col)
    return one_hot_cols

def get_orig_x_oh_mapping(column_list):
    one_hot_cols = extract_one_hot_column_names(column_list)
    one_hot_mapping = dict()
    for one_hot_col in one_hot_cols:
        orig_name = one_hot_col.split('_')[:-1] # remove suffixes ('_1.0')
        orig_name = '_'.join(orig_name) # reverse split
        if orig_name not in one_hot_mapping.keys():
            one_hot_mapping[orig_name] = [one_hot_col]
        else:
            one_hot_mapping[orig_name].append(one_hot_col)
    return one_hot_mapping

def get_oh_x_orig_mapping(column_list):
    one_hot_cols = extract_one_hot_column_names(column_list)
    one_hot_mapping = dict()
    for one_hot_col in one_hot_cols:
        orig_name = one_hot_col.split('_')[:-1] # remove suffixes ('_1.0')
        orig_name = '_'.join(orig_name) # reverse split
        if orig_name not in one_hot_mapping.keys():
            one_hot_mapping[one_hot_col] = orig_name
    return one_hot_mapping
    
def get_display_feature_mapping(features: list):
    root_path = os.path.dirname(pathlib.Path(__file__).parent.parent.resolve())
    codebook_path = os.path.join(root_path, 'data/Helix data codebook.xlsx')
    codebook = pd.read_excel(codebook_path, na_values='NA')
    codebook = codebook.dropna(subset=['variable_name'])
    
    imputed_df_path = os.path.join(root_path, 'data/imputed/mf_all_omics.csv')
    feature_selection = pd.read_csv(imputed_df_path).columns
    
    cat_names_path = os.path.join(root_path, 'data/cat_names.xlsx')
    cat_names = pd.read_excel(cat_names_path).set_index('var_name')
    
    one_hot_cols = extract_one_hot_column_names(features)
    mapping_oh_x_orig = get_oh_x_orig_mapping(features)
    
    mappring_featurename_x_displayname = dict()
    for feature in features:
        orig_feature = feature
        if feature in one_hot_cols: # one hot encoded
            feature = mapping_oh_x_orig[feature]
            
            # dont add cat suffix for binary categorical variables
            count = 0
            for f in feature_selection:
                if feature in f:
                    count += 1
            # print(feature, count)
            if count > 1:
                if orig_feature in cat_names.index: # set the right category as suffix
                    # print(orig_feature, cat_names.loc[orig_feature]['category'])
                    display_name = list(codebook['name_for_table'].loc[
                        codebook['variable_name'] == feature
                    ])[0] + ' (' + str(cat_names.loc[orig_feature]['category']) + ')'
                else: # set generic category
                    display_name = list(codebook['name_for_table'].loc[
                        codebook['variable_name'] == feature
                    ])[0] + ' cat' + orig_feature[-3: -2]
            else:
                display_name = list(codebook['name_for_table'].loc[
                    codebook['variable_name'] == feature
                ])[0]
        else:
            display_name = list(codebook['name_for_table'].loc[
                codebook['variable_name'] == feature
            ])[0]
        if  isinstance(display_name, float) or len(display_name) == 0: # display name is nan
            print('error')
            assert(False)
        mappring_featurename_x_displayname[orig_feature] = display_name
    return mappring_featurename_x_displayname

def compute_category_x_variable_names_mapping(features_names, category: str):
    # print(features_names)
    if category == 'vartype':
        families_x_variable_names_mapping = compute_category_x_variable_names_mapping(features_names, category='family')
        vartype_x_families_mapping = compute_vartypes_x_families_mapping(list(families_x_variable_names_mapping.keys()))
        
        vartype_x_variable_names = {}
        for key in vartype_x_families_mapping.keys():
            families = vartype_x_families_mapping[key]
            vartype_x_variable_names[key] = []
            for fam in families:
                variables = families_x_variable_names_mapping[fam]
                vartype_x_variable_names[key].extend(variables)
        return vartype_x_variable_names
    
    results_path = os.path.join(os.path.dirname(pathlib.Path(__file__).parent.parent.resolve()), 'data/Helix data codebook.xlsx')
    # print(results_path)
    codebook = pd.read_excel(results_path, na_values='NA')
    one_hot_cols = extract_one_hot_column_names(features_names)
    mapping_orig_x_oh = get_orig_x_oh_mapping(features_names)
    
    variable_name_x_familly_mapping = dict()
    familly_x_variable_name_mapping = dict()
    for col in features_names:
        familly = None
        if col in one_hot_cols:
            founded = False
            for orig, oh_array in mapping_orig_x_oh.items():
                if col in oh_array:
                    founded = True
                    familly = codebook[category].loc[
                        (codebook['variable_name'] == orig)
                    ]
            if not founded:
                print("WARNING: couldnt found column:", col)
        else:
            familly = codebook[category].loc[
                (codebook['variable_name'] == col)
            ]
        # print(col)
        assert(len(familly) == 1)
        variable_name_x_familly_mapping[col] = familly.iloc[0]
        if familly.iloc[0] in familly_x_variable_name_mapping.keys():
            familly_x_variable_name_mapping[familly.iloc[0]].append(col)
        else:
            familly_x_variable_name_mapping[familly.iloc[0]] = [col]
    assert(len(list(itertools.chain.from_iterable(familly_x_variable_name_mapping.values()))) == len(features_names))
    assert(len(variable_name_x_familly_mapping.keys()) == len(features_names))
    assert(set(itertools.chain.from_iterable(familly_x_variable_name_mapping.values())) == set(variable_name_x_familly_mapping.keys()))
    assert(set(variable_name_x_familly_mapping.values()) == set(familly_x_variable_name_mapping.keys()))
    
    for key in familly_x_variable_name_mapping.keys():
        for col in familly_x_variable_name_mapping[key]:
            assert(key == variable_name_x_familly_mapping[col])
    for key in variable_name_x_familly_mapping.keys():
        assert(variable_name_x_familly_mapping[key] in familly_x_variable_name_mapping.keys())
        
    # test: 1. creates fake family mapping dictionaries. 
    #       2. creates fake shap_values.values data.
    # familly_x_variable_name_mapping = {
    #                                     'A' : ['v1', 'v2', 'v4'],
    #                                     'B' : ['v3', 'v5'],
    #                                     'C' : ['v6']
    #                                   }
    # variable_name_x_familly_mapping = {
    #                                     'v1' : 'A' , 'v2' : 'A', 'v3' : 'B',
    #                                     'v4' : 'A', 'v6' : 'C',
    #                                     'v5' : 'B'
    #                                   }
    # shap_values.feature_names = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    # print(shap_values.values.shape)
    # print(np.array([[1, 2, 3, 4, 5, 6], [2, -1, 1, 1, 1, 3], [-4, -1, -3, 0, 1, -2]]))
    # shap_values.values = np.array([[1, 2, 3, 4, 5, 6], [2, -1, 1, 1, 1, 3], [-4, -1, -3, 0, 1, -2]])
    # print(np.array([[1, 2, 3, 4, 5, 6], [2, -1, 1, 1, 1, 3], [-4, -1, -3, 0, 1, -2]]).shape)
        
    return familly_x_variable_name_mapping


def compute_vartypes_x_families_mapping(families: list, category=None):
    vartypes_x_families = {'Clinical factors': [],
                                 'Metabolites/Proteins': [],
                                 'Exposures': [],
                                 'Covariates': []}
    for index, familly in enumerate(families):
        if familly in ['Clinical factors (respi)', 'Clinical factors (mental)', 'Parental clinical factors (mental)', 'Clinical factors (cardio)', 'Lipids', 'Clinical factors', 'Parental clinical factors']:
            vartype = 'Clinical factors'
        elif familly in ['Proteome', 'Serum metabolome', 'Urine metabolome']:
            vartype = 'Metabolites/Proteins'
        elif familly == 'Covariates':
            vartype = 'Covariates'
        else:
            vartype = 'Exposures'
        
        # indices = [features_names.index(col) for col in familly_x_variable_name_mapping[familly]]
        # vartypes_x_variable_names[vartype].extend(indices)
        
        vartypes_x_families[vartype].append(familly)
    return vartypes_x_families
    
   

def compute_family_shap(shap_values):
    familly_shap_values = copy.deepcopy(shap_values)
    familly_x_variable_name_mapping = compute_category_x_variable_names_mapping(shap_values.feature_names, category='family')
    # print(familly_x_variable_name_mapping)
    
    columns = shap_values.feature_names
    familly_adjusted_shap = np.zeros((shap_values.shape[0], len(familly_x_variable_name_mapping.keys())))
    familly_names = []
    for index, familly in enumerate(familly_x_variable_name_mapping.keys()):
        indices = [columns.index(col) for col in familly_x_variable_name_mapping[familly]]
        familly_adjusted_shap[:, index] = np.sum(shap_values.values[:, indices], axis=1)
        familly_names.append(familly)

    familly_shap_values.values =  familly_adjusted_shap
    familly_shap_values.feature_names = familly_names
    return familly_shap_values


def compute_period_shap(shap_values):
    period_shap_values_object = copy.deepcopy(shap_values)
    period_x_variable_name_mapping = compute_category_x_variable_names_mapping(shap_values.feature_names, category='period')
    
    columns = shap_values.feature_names
    period_adjusted_shap = np.zeros((shap_values.shape[0], len(period_x_variable_name_mapping.keys())))
    period_names = []
    for index, period in enumerate(period_x_variable_name_mapping.keys()):
        indices = [columns.index(col) for col in period_x_variable_name_mapping[period]]
        period_adjusted_shap[:, index] = np.sum(shap_values.values[:, indices], axis=1)
        period_names.append(period)

    period_shap_values_object.values =  period_adjusted_shap
    period_shap_values_object.feature_names = period_names
    return period_shap_values_object


def compute_vartype_shap(shap_values):
    vartype_shap_values_object = copy.deepcopy(shap_values)
    vartypes_x_variable_names = compute_category_x_variable_names_mapping(shap_values.feature_names, category='vartype')
    # print(vartypes_x_variable_names)
    
    columns = shap_values.feature_names
    vartype_adjusted_shap = np.zeros((shap_values.shape[0], len(vartypes_x_variable_names.keys())))
    
    for index, vartype in enumerate(vartypes_x_variable_names.keys()):
        indices = [columns.index(col) for col in vartypes_x_variable_names[vartype]]
        vartype_adjusted_shap[:, index] = np.sum(shap_values.values[:, indices], axis=1)
      
    vartype_shap_values_object.values = vartype_adjusted_shap
    vartype_shap_values_object.feature_names = list(vartypes_x_variable_names.keys())
    return vartype_shap_values_object


def compute_oh_adjusted_shap(shap_values):
    one_hot_cols = extract_one_hot_column_names(shap_values.feature_names)
    one_hot_cols_indices = []
    mapping_orig_x_oh = get_orig_x_oh_mapping(shap_values.feature_names)

    size_diff = len(sum(list(mapping_orig_x_oh.values()), [])) - len(mapping_orig_x_oh.keys())
    adjusted_shap_values = np.zeros((shap_values.values.shape[0], shap_values.values.shape[1] - size_diff))
    col_list = []
    skip_list = []
    index_adjusted = 0
    for index, col in enumerate(shap_values.feature_names):
        if col in skip_list:
            continue

        indices = []
        for orig, oh_array in mapping_orig_x_oh.items():
            if col in oh_array:
                indices = [shap_values.feature_names.index(col) for col in mapping_orig_x_oh[orig]]
                col_list.append(orig)
                for c in oh_array:
                    skip_list.append(c)
                break

        if len(indices) == 0:
            adjusted_shap_values[:, index_adjusted] = shap_values.values[:, index]
            col_list.append(col)
        else:
            adjusted_shap_values[:, index_adjusted] = np.sum(shap_values.values[:, indices], axis=1)
        index_adjusted += 1

    one_hot_adjusted_shap_values = copy.deepcopy(shap_values)
    one_hot_adjusted_shap_values.values = adjusted_shap_values
    one_hot_adjusted_shap_values.feature_names = col_list
    return one_hot_adjusted_shap_values    
    
def standardize_shap_importance(shap_values): # make shap importances sum up to 1
    shap_values_cpy = copy.deepcopy(shap_values)
    abs_shap_values = abs(shap_values_cpy.values)
    shap_values_cpy.values = shap_values_cpy.values / abs_shap_values.mean(axis=0).sum(axis=0)
    # print(shap_values_cpy.values.mean(axis=0).sum(), abs(shap_values_cpy.values).mean(axis=0).sum())
    return shap_values_cpy
    
def plot_shap_feature_importance_ax(ax, shap_values, family_wise=False):
    importances = [np.mean(abs(shap_values.values[:,i])) for i in range(len(shap_values.values[0]))]
    indices = np.argsort(importances)[::-1]
    sorted_importances = [importances[indice] for indice in indices]
    sorted_labels = [shap_values.feature_names[indice] for indice in indices]
    
    # get display variable names
    if len(shap_values.feature_names) > 50: # family
        mapping_featurename_x_displayname = get_display_feature_mapping(sorted_labels)
        sorted_labels = [mapping_featurename_x_displayname[key] for key in sorted_labels]
    
    if family_wise:
        ax.set_title("Family wise feature importances (SHAP)")
    else:
        ax.set_title("Feature importances (SHAP)")
    ax.barh(range(len(sorted_importances[:10])), sorted_importances[:10][::-1], color="crimson", align="center")
    ax.set_yticks(ticks=range(len(sorted_importances[:10])))
    ax.set_yticklabels(sorted_labels[:10][::-1], fontsize=10)
    ax.set_ylim([-1, len(sorted_importances[:10])])
    ax.set_xlabel("mean(|SHAP value|)")
    for container in ax.containers:
        ax.bar_label(container, labels=[f'{x:.2f}' for x in container.datavalues], padding=2)

        
def plot_shap_feature_importance(shap_values, family_wise=False):
    fig, ax = plt.subplots(1, 1, figsize=(10,6))
    plot_shap_feature_importance_ax(ax, shap_values, family_wise=False)
    plt.show()

    
def plot_shap_dashboard(shap_values, figsize=(19,6)):
    one_hot_adjusted_shap_values = compute_oh_adjusted_shap(shap_values)
    familly_shap_values = compute_family_shap(shap_values)
    # one_hot_adjusted_shap_values = standardize_shap_importance(one_hot_adjusted_shap_values)
    # familly_shap_values = standardize_shap_importance(familly_shap_values)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios" : [1,1]})
    plot_shap_feature_importance_ax(ax1, one_hot_adjusted_shap_values)
    plot_shap_feature_importance_ax(ax2, familly_shap_values, family_wise=True)
    fig.tight_layout(pad=2.0)
    plt.show()

    
def extract_significant_interactions(shap_interactions, feature_names: list, percent:int=10, threshold:float=0.01, mode='mean'):
    # interaction importance
    if mode == 'max': # sort by max local interaction strength
        tmp = np.abs(shap_interactions).max(0)
    else: # sort by mean interaction strength
        tmp = np.abs(shap_interactions).mean(0)
    interaction_strengths_dict = {}
    marginal_effect_dict = {}
    for i in range(tmp.shape[0]):
        for j in range(i, tmp.shape[1]):
            if tmp[i, j] > 0:
                f1 = feature_names[i]
                f2 = feature_names[j]
                key = f1 + '_XXX_' + f2
                if f1==f2:
                    marginal_effect_dict[f1] = tmp[i, j]
                else:
                    interaction_strengths_dict[key] = 2 * tmp[i, j] # interaction(x1<=>x2) = interaction(x1.x2) + interaction(x2.x1)
    
    interaction_strengths_dict = {k: v for k, v in sorted(interaction_strengths_dict.items(), key=lambda item: item[1], reverse=True)} # sort by value
    
    # for key in interaction_strengths_dict.keys():
    #     print(key, interaction_strengths_dict[key])
        
    marginal_effect_dict = {k: v for k, v in sorted(marginal_effect_dict.items(), key=lambda item: item[1], reverse=True)} # sort by value
    significant_interactions = []
    for interaction in interaction_strengths_dict.keys():
        strength = interaction_strengths_dict[interaction]
        x1 = interaction.split('_XXX_')[0]
        x2 = interaction.split('_XXX_')[1]
        cumulated_marginal_effect = marginal_effect_dict[x1] + marginal_effect_dict[x2]
        # print(interaction, strength, cumulated_marginal_effect)
        # if strength > cumulated_marginal_effect/percent and strength > 0.01:
        if strength > threshold:
            significant_interactions.append(interaction)
    return significant_interactions


def plot_dependance_plots(shap_interactions, features, feature_names, significant_interactions, title=""):
    # significant_interactions = extract_significant_interactions(shap_interactions=shap_interactions, feature_names=feature_names)
    mapping_featurename_x_displayname = get_display_feature_mapping(feature_names)
    display_names = [mapping_featurename_x_displayname[key] for key in feature_names]
    
    # plot title
    fig, ax = plt.subplots(figsize=(19, 0.1))
    # fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.plot()
    
    if len(significant_interactions) > 1:
        n= math.ceil(len(significant_interactions)/2)
        fig, axs = plt.subplots(n, 2, figsize=(19, n*3))
        
        for i in range(0, len(significant_interactions)):
            # axs[i//2, i%2].plot([0, 1], [3, 3])
            x1 = significant_interactions[i].split('_XXX_')[0]
            x2 = significant_interactions[i].split('_XXX_')[1]
            if (len(significant_interactions)==2):
                axs[i%2].set_title("Pairwise interaction: " + mapping_featurename_x_displayname[x1] + ' - ' + mapping_featurename_x_displayname[x2])
                shap.dependence_plot(
                    (mapping_featurename_x_displayname[x1], mapping_featurename_x_displayname[x2]),
                    shap_interactions, features[feature_names].values,
                    feature_names = display_names,
                    ax=axs[i%2],
                    show=False,
                    x_jitter=1
                )
                axs[i%2].set_ylabel("SHAP interaction value")
            else:
                axs[i//2, i%2].set_title("Pairwise interaction: " + mapping_featurename_x_displayname[x1] + ' - ' + mapping_featurename_x_displayname[x2])
                shap.dependence_plot(
                    (mapping_featurename_x_displayname[x1], mapping_featurename_x_displayname[x2]),
                    shap_interactions, features[feature_names].values,
                    feature_names = display_names,
                    ax=axs[i//2, i%2],
                    show=False, 
                    x_jitter=1
                )
                axs[i//2, i%2].set_ylabel("SHAP interaction value")
        if (len(significant_interactions) % 2 ==1): # hide last subplot if odd number of plots.
            axs[-1, -1].axis('off')
        fig.tight_layout(h_pad=5.0)
    elif len(significant_interactions)==1:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        x1 = significant_interactions[0].split('_XXX_')[0]
        x2 = significant_interactions[0].split('_XXX_')[1]
        shap.dependence_plot(
                (mapping_featurename_x_displayname[x1], mapping_featurename_x_displayname[x2]),
                shap_interactions, features[feature_names].values,
                feature_names = display_names,
                ax=ax,
                show=False,
                x_jitter=1
            )
        ax.set_ylabel("SHAP interaction value")
        fig.tight_layout(h_pad=1.0)
    plt.savefig(f"../../../results/interactions/{title}.svg", format="svg")
    plt.savefig(f"../../../results/interactions/{title}.png", format="png")
        
        
def save_feature_importances(path, importances, importances_std, features_list, target_name, make_new=False):
    mapping_featurename_x_displayname = get_display_feature_mapping(features_list)
    if make_new:
        fi_results = pd.DataFrame(columns=['raw', 'exposure', target_name + ' mean feature importance', target_name + ' mean std'])
        for i, feature in enumerate(features_list):
            row = dict()
            row['raw'] = feature
            row['exposure'] = mapping_featurename_x_displayname[feature]
            row[target_name + ' mean feature importance'] = importances[i]
            row[target_name + ' mean std'] = importances_std[i]
            fi_results = fi_results.append(row, ignore_index=True)
            
        fi_results = fi_results.sort_values(by=target_name + ' mean feature importance', ascending=False)
        # display(fi_results)
        exposures = fi_results['raw']
        fi_results_test = fi_results.drop_duplicates(subset='exposure')

        fi_results = fi_results.set_index('raw')
        fi_results.to_excel(path)
    else:
        old_results = pd.read_excel(path)

        old_results = old_results.set_index('raw')
        for col in [target_name + ' mean feature importance', target_name + ' mean std']:
            try:
                old_results[col].values[:] = 0
            except:
                pass

        # display(old_results)
        for i, feature in enumerate(features_list):
            old_results.loc[feature, target_name + ' mean feature importance'] = importances[i]
            old_results.loc[feature, target_name + ' mean std'] = importances_std[i]
            old_results.loc[feature, 'exposure'] = mapping_featurename_x_displayname[feature]

        old_results.to_excel(path)
        # display(old_results)
        
        
def save_family_feature_importances(path, importances, importances_std, families, target_name, make_new=False):
    if make_new:
        fi_results = pd.DataFrame(columns=['family', target_name + ' mean feature importance', target_name + ' mean std'])
        
        for i, family in enumerate(families):
            row = dict()
            row['family'] = family
            row[target_name + ' mean feature importance'] = importances[i]
            row[target_name + ' mean std'] = importances_std[i]
            fi_results = fi_results.append(row, ignore_index=True)
            
        fi_results = fi_results.sort_values(by=target_name + ' mean feature importance', ascending=False)
        # display(fi_results)
        fi_results = fi_results.set_index('family')
        fi_results.to_excel(path)
    else:
        old_results = pd.read_excel(path)
        old_results = old_results.set_index('family')
        for col in [target_name + ' mean feature importance', target_name + ' mean std']:
            try:
                old_results[col].values[:] = 0
            except:
                pass

        # display(old_results)
        for i, family in enumerate(families):
            old_results.loc[family, target_name + ' mean feature importance'] = importances[i]
            old_results.loc[family, target_name + ' mean std'] = importances_std[i]
        old_results.to_excel(path)
        # display(old_results)
        
        
def cohorts_cv(model, features, target, standardize):
    cohort_cols = [f"cohort_{i}.0" for i in range(1, 7)]
    X = features.drop(columns=cohort_cols)
    # X = features
    scores_train, scores_test = [], []
    for cohort in cohort_cols:
        train_idx = X[features[cohort] == 0].index
        test_idx = X[features[cohort] == 1].index
        
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = target.loc[train_idx], target.loc[test_idx]
        print(X_test.shape)
        
        if standardize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            X_train = pd.DataFrame(data=X_train, columns=[ele for ele in X.columns])
            X_test = pd.DataFrame(data=X_test, columns=[ele for ele in X.columns])
        
        # display(X_train.head())
        # print(y_train)
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        score_train = metrics.r2_score(y_train, y_pred_train)
        score_test = metrics.r2_score(y_test, y_pred_test)
        scores_train.append(score_train)
        scores_test.append(score_test)
        
    # Compute and print performances and std
    mean_train_r2, std_train_r2 = np.mean(scores_train), np.std(scores_train)
    mean_test_r2, std_test_r2 = np.mean(scores_test), np.std(scores_test)
    print(f"Results:")
    print("\tMean train score: {0:.3f} std:{1:.2f}".format(mean_train_r2, std_train_r2))
    print("\tMean test score: {0:.3f} std:{1:.2f}".format(mean_test_r2, std_test_r2))
    
    print("Complete results")
    print(f"\tMean train scores: {scores_train}")
    print(f"\tMean test score: {scores_test}")

    
def covariates_2steps_target_adjustement_cohort_cv(cov_model, res_model, features, target, standardize):
    cohort_cols = [f"cohort_{i}.0" for i in range(1, 7)]
    step1_scores_train, step1_scores_test = [], []
    step2_scores_train, step2_scores_test = [], []
    final_scores_train, final_scores_test = [], []
    for cohort in cohort_cols:
        train_idx = features[features[cohort] == 0].index
        test_idx = features[features[cohort] == 1].index
        
        X_train, X_test = features.loc[train_idx], features.loc[test_idx]
        y_train, y_test = target.loc[train_idx], target.loc[test_idx]
        
        x_train_covariates, x_test_covariates = X_train[cohort_cols], X_test[cohort_cols] # data step 1
        if standardize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            X_train = pd.DataFrame(data=X_train, columns=[ele for ele in features.columns])
            X_test = pd.DataFrame(data=X_test, columns=[ele for ele in features.columns])
            
        x_train_no_cov, x_test_no_cov = X_train.drop(columns=cohort_cols), X_test.drop(columns=cohort_cols) # data step 2
        
        # Step 1
        cov_model.fit(x_train_covariates, y_train)
        s1_y_pred_train = cov_model.predict(x_train_covariates)
        s1_y_pred_test = cov_model.predict(x_test_covariates)
        score1_train = metrics.r2_score(y_train, s1_y_pred_train)
        score1_test = metrics.r2_score(y_test, s1_y_pred_test)
        step1_scores_train.append(score1_train)
        step1_scores_test.append(score1_test)
        
        # Compute residuals
        residuals_train = y_train - s1_y_pred_train
        residuals_test = y_test - s1_y_pred_test
        
        # step 2
        res_model.fit(x_train_no_cov.to_numpy(), residuals_train)
        s2_y_pred_train = res_model.predict(x_train_no_cov.to_numpy())
        s2_y_pred_test = res_model.predict(x_test_no_cov.to_numpy())
        score2_train = metrics.r2_score(residuals_train, s2_y_pred_train)
        score2_test = metrics.r2_score(residuals_test, s2_y_pred_test)
        step2_scores_train.append(score2_train)
        step2_scores_test.append(score2_test)
        
        # Compute final performance metrics (steps 1+2)
        final_pred_train = s1_y_pred_train + s2_y_pred_train
        final_pred_test = s1_y_pred_test + s2_y_pred_test
        final_score_train = metrics.r2_score(y_train, final_pred_train)
        final_score_test = metrics.r2_score(y_test, final_pred_test)
        final_scores_train.append(final_score_train)
        final_scores_test.append(final_score_test)
        
    # Compute and print performances and std
    mean_step1_train_r2, std_step1_train_r2 = np.mean(step1_scores_train), np.std(step1_scores_train)
    mean_step1_test_r2, std_step1_test_r2 = np.mean(step1_scores_test), np.std(step1_scores_test)
    mean_step2_train_r2, std_step2_train_r2 = np.mean(step2_scores_train), np.std(step2_scores_train)
    mean_step2_test_r2, std_step2_test_r2 = np.mean(step2_scores_test), np.std(step2_scores_test)
    mean_final_train_r2, std_final_train_r2 = np.mean(final_scores_train), np.std(final_scores_train)
    mean_final_test_r2, std_final_test_r2 = np.mean(final_scores_test), np.std(final_scores_test)
    print(f"Results:")
    print("\tstep 1 mean train score: {0:.3f} std:{1:.2f}".format(mean_step1_train_r2, std_step1_train_r2))
    print("\tstep 1 mean test score: {0:.3f} std:{1:.2f}".format(mean_step1_test_r2, std_step1_test_r2))
    print("\tstep 2 mean train score: {0:.3f} std:{1:.2f}".format(mean_step2_train_r2, std_step2_train_r2))
    print("\tstep 2 mean test score: {0:.3f} std:{1:.2f}".format(mean_step2_test_r2, std_step2_test_r2))
    print("\tfinal mean train score: {0:.3f} std:{1:.2f}".format(mean_final_train_r2, std_final_train_r2))
    print("\tfinal mean test score: {0:.3f} std:{1:.2f}".format(mean_final_test_r2, std_final_test_r2))
    
    print("Complete results")
    print(f"\tstep 1 mean train scores: {step1_scores_train}")
    print(f"\tstep 1 mean test score: {step1_scores_test}")
    print(f"\tstep 2 mean train score: {step2_scores_train}")
    print(f"\tstep 2 mean test score: {step2_scores_test}")
    print(f"\tfinal mean train score: {final_scores_train}")
    print(f"\tfinal mean test score: {final_scores_test}")