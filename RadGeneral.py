import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from scipy.stats import ttest_ind, chi2_contingency
import statsmodels.api as sm
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from pandas.core.common import SettingWithCopyWarning
from datetime import datetime

print(datetime.now())


def load_image(nifti_path):
    image = sitk.ReadImage(nifti_path)
    new_spacing = [1, 1, 1]
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(round(original_size[i] * (original_spacing[i] / new_spacing[i]))) for i in range(3)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampled_image = resampler.Execute(image)
    return resampled_image

def load_mask(mask_path):
    mask = sitk.ReadImage(mask_path)
    new_spacing = [1, 1, 1]
    original_spacing = mask.GetSpacing()
    original_size = mask.GetSize()
    new_size = [int(round(original_size[i] * (original_spacing[i] / new_spacing[i]))) for i in range(3)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputDirection(mask.GetDirection())
    resampler.SetOutputOrigin(mask.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampled_mask = resampler.Execute(mask)
    mask = sitk.GetArrayFromImage(resampled_mask)
    mask = np.where(mask == 2, 1, 0)
    filtered_mask = sitk.GetImageFromArray(mask)
    filtered_mask.CopyInformation(resampled_mask)
    return filtered_mask

def extract_radiomic(nifti_path, mask_path):
    config_path = "utils/pyrad_config.yaml"
    extractor = featureextractor.RadiomicsFeatureExtractor(config_path)
    features = extractor.execute(nifti_path, mask_path)
    features = {k: v for k, v in features.items() if "diagnostics" not in k}
    features = pd.DataFrame(features, index = [0])
    return features

def clinical_analysis(clinical_data):
    results = []
    features_to_keep = []
    for col in clinical_data.columns:
        if col not in ['target','ID','TTP']:
            num_unique_values = clinical_data[col].nunique()
            if num_unique_values > 10:
                group1 = clinical_data[clinical_data['target'] == 0][col]
                group2 = clinical_data[clinical_data['target'] == 1][col]
                t_stat, p_val = ttest_ind(group1, group2, nan_policy='omit')
                results.append({
                    'feature': col,
                    'type': 'Continuous',
                    'test': 'T-test',
                    'p-value': p_val
                })

            else:
                contingency_table = pd.crosstab(clinical_data[col], clinical_data['target'])
                chi2, p_val, dof, expected = chi2_contingency(contingency_table)
                results.append({
                    'feature': col,
                    'type': 'Categorical',
                    'test': 'Chi-squared',
                    'p-value': p_val
                })
            if p_val <= 0.1:
                features_to_keep.append(col)
            results_ua = pd.DataFrame(results)

    if len(features_to_keep) > 0:
        X = clinical_data[features_to_keep]
        y = clinical_data['target']
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        logit_model = sm.Logit(y,X)
        try:
            results_ma = logit_model.fit(disp=0)
            p_values = results_ma.pvalues
            p_values.index = features_to_keep
            significant_features = p_values[p_values <= 0.1].index
            final_significant_df = clinical_data[significant_features]
            final_significant_df = final_significant_df.copy()
            final_significant_df['target'] = clinical_data.iloc[:,-1]
        except sm.tools.sm_exceptions.PerfectSeparationError:
            final_significant_df = clinical_data[features_to_keep]
            final_significant_df = final_significant_df.copy()
            final_significant_df['target'] = clinical_data.iloc[:, -1]
        except np.linalg.LinAlgError:
            final_significant_df = clinical_data[features_to_keep]
            final_significant_df = final_significant_df.copy()
            final_significant_df['target'] = clinical_data.iloc[:, -1]

    return final_significant_df

def correlation_analysis(data, thrs):
    mat = data.corr().abs()
    high_correlation_pairs = []
    # for i in tqdm(range(len(mat.columns)), desc="Processing columns"):
    for i in range(len(mat.columns)):
        for j in range(i + 1, len(mat.columns)):
            if abs(mat.iloc[i, j]) > thrs:
                v1 = mat.columns[i]
                v2 = mat.columns[j]
                correlation_value = mat.iloc[i, j]
                high_correlation_pairs.append((v1, v2, correlation_value))
    pair_data = pd.DataFrame(high_correlation_pairs, columns=['feature1', 'feature2', 'correlation value'])
    drop_list = []
    for index, row in pair_data.iterrows():
        feature1 = row['feature1']
        feature2 = row['feature2']
        var1 = data[feature1].var()
        var2 = data[feature2].var()
        if var1 < var2:
            ftr = feature1
        else:
            ftr = feature2

        if ftr not in drop_list:
            drop_list.append(ftr)

    data = data.drop(columns=drop_list)
    return data

def lasso_analysis(data, target, cv):
    warnings.simplefilter("ignore", ConvergenceWarning)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    y = target
    lasso_cv = LassoCV(cv=10, random_state= 42).fit(data_scaled, y)
    a = lasso_cv.alpha_
    lasso_single = Lasso(alpha=a).fit(data_scaled, y)
    selected_features = pd.Series(lasso_single.coef_, index=data.columns)
    selected_features = selected_features[selected_features != 0]
    filtered_data = data[selected_features.index]
    return selected_features, filtered_data

def radscore(x, features):
    radscore_df = x * features
    radscore_df = radscore_df.sum(axis=1)
    radscore_df = pd.DataFrame(radscore_df.iloc[:], columns=['RadScore'])

    return radscore_df

def lr(df,y):
    param_grid = {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}
    grid = GridSearchCV(LogisticRegression(max_iter=10000), param_grid, scoring='accuracy')
    grid.fit(df.values, y)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(df.values)
    acc = accuracy_score(y, y_pred)
    return acc, best_model

def rf(df, y):
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20]}
    grid = GridSearchCV(RandomForestClassifier(), param_grid, scoring='accuracy')
    grid.fit(df.values, y)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(df.values)
    acc = accuracy_score(y, y_pred)
    return acc, best_model

def svm(df,y):
    param_grid = {'C': [0.1, 1, 10]}
    grid = GridSearchCV(SVC(), param_grid, scoring='accuracy')
    grid.fit(df.values, y)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(df.values)
    acc = accuracy_score(y, y_pred)
    return acc, best_model

def xgboost(df, y):
    param_grid = {'n_estimators': [50, 100, 200]}
    grid = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), param_grid, scoring='accuracy')
    grid.fit(df.values, y)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(df.values)
    acc = accuracy_score(y, y_pred)
    return acc, best_model

def mlp(df, y):
    param_grid = {'hidden_layer_sizes': [(8,), (16,), (32, 32)], 'alpha': [0.0001, 0.001]}
    grid = GridSearchCV(MLPClassifier(max_iter=10000), param_grid, scoring='accuracy')
    grid.fit(df.values, y)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(df.values)
    acc = accuracy_score(y, y_pred)
    return acc, best_model

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    image_mask = pd.read_csv("utils/image-mask-final.csv")
    clinical_data = pd.read_csv("utils/clinical-data-final.csv")

    #
    # Radiomics = pd.DataFrame()
    # for row, item in tqdm(image_mask.iterrows(), total=len(image_mask)):
    #     image_path = item[0].split(';')[0]
    #     mask_path = item[0].split(';')[1]
    #     image = load_image(image_path)
    #     mask = load_mask(mask_path)
    #     radiomics_features = extract_radiomic(image_path,mask_path)
    #     Radiomics = pd.concat([Radiomics, radiomics_features], ignore_index=True)
    # Radiomics.to_csv("file\\RadGeneral\\Radiomics.csv")

    radiomics_features = pd.read_csv("file/RadGeneral/Radiomics.csv")
    target = clinical_data['target'].values
    cv = True
    #
    if cv:
        print("\nStart Analysis in Cross-Validation: ")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        results = {
            'fold': [], 'train_acc_lr_c': [], 'test_acc_lr_c': [],
            'train_acc_lr_r': [], 'test_acc_lr_r': [],
            'train_acc_lr_rs': [], 'test_acc_lr_rs': [],
            'train_acc_lr_crs': [], 'test_acc_lr_crs': [],
            'train_acc_lr_cr': [], 'test_acc_lr_cr': [],

            'train_acc_rf_c': [], 'test_acc_rf_c': [],
            'train_acc_rf_r': [], 'test_acc_rf_r': [],
            'train_acc_rf_rs': [], 'test_acc_rf_rs': [],
            'train_acc_rf_crs': [], 'test_acc_rf_crs': [],
            'train_acc_rf_cr': [], 'test_acc_rf_cr': [],

            'train_acc_svm_c': [], 'test_acc_svm_c': [],
            'train_acc_svm_r': [], 'test_acc_svm_r': [],
            'train_acc_svm_rs': [], 'test_acc_svm_rs': [],
            'train_acc_svm_crs': [], 'test_acc_svm_crs': [],
            'train_acc_svm_cr': [], 'test_acc_svm_cr': [],

            'train_acc_xgb_c': [], 'test_acc_xgb_c': [],
            'train_acc_xgb_r': [], 'test_acc_xgb_r': [],
            'train_acc_xgb_rs': [], 'test_acc_xgb_rs': [],
            'train_acc_xgb_crs': [], 'test_acc_xgb_crs': [],
            'train_acc_xgb_cr': [], 'test_acc_xgb_cr': [],

            'train_acc_mlp_c': [], 'test_acc_mlp_c': [],
            'train_acc_mlp_r': [], 'test_acc_mlp_r': [],
            'train_acc_mlp_rs': [], 'test_acc_mlp_rs': [],
            'train_acc_mlp_crs': [], 'test_acc_mlp_crs': [],
            'train_acc_mlp_cr': [], 'test_acc_mlp_cr': []
        }

        for fold, (train_index, test_index) in tqdm(enumerate(skf.split(clinical_data, target)),
                                                    total=skf.get_n_splits(), desc="Processing folds"):
            train_clinical = clinical_data.iloc[train_index]
            train_radiomics = radiomics_features.iloc[train_index]
            train_target = train_clinical['target'].values

            train_clinical = clinical_analysis(train_clinical)
            train_clinical = train_clinical.drop(columns=['target'])

            train_radiomics = correlation_analysis(train_radiomics, 0.9)
            features, train_radiomics_lasso = lasso_analysis(train_radiomics, train_target, 5)

            if features.empty:
                print(f'\nNo radiomics features were selected in fold {fold}!')
                continue
            else:

                train_rad_score = radscore(train_radiomics_lasso, features)

                scaler = StandardScaler()

                scaled_radiomics_lasso = scaler.fit_transform(train_radiomics_lasso)
                train_radiomics_lasso.iloc[:, :] = scaled_radiomics_lasso

                scaled_rad_score = scaler.fit_transform(train_rad_score)
                train_rad_score.iloc[:, :] = scaled_rad_score

                print('train solo clinical')
                # Solo Clinical
                train_acc_lr_c, lr_clinical = lr(train_clinical, train_target)
                train_acc_rf_c, rf_clinical = rf(train_clinical, train_target)
                train_acc_svm_c, svm_clinical = svm(train_clinical, train_target)
                train_acc_xgb_c, xgb_clinical = xgboost(train_clinical, train_target)
                train_acc_mlp_c, mlp_clinical = mlp(train_clinical, train_target)

                print('train solo radiomics')
                # Solo Radiomics
                train_acc_lr_r, lr_radiomics = lr(train_radiomics_lasso, train_target)
                train_acc_rf_r, rf_radiomics = rf(train_radiomics_lasso, train_target)
                train_acc_svm_r, svm_radiomics = svm(train_radiomics_lasso, train_target)
                train_acc_xgb_r, xgb_radiomics = xgboost(train_radiomics_lasso, train_target)
                train_acc_mlp_r, mlp_radiomics = mlp(train_radiomics_lasso, train_target)

                print('train radscore')
                # Solo RadScore
                train_acc_lr_rs, lr_rad_score = lr(train_rad_score, train_target)
                train_acc_rf_rs, rf_rad_score = rf(train_rad_score, train_target)
                train_acc_svm_rs, svm_rad_score = svm(train_rad_score, train_target)
                train_acc_xgb_rs, xgb_rad_score = xgboost(train_rad_score, train_target)
                train_acc_mlp_rs, mlp_rad_score = mlp(train_rad_score, train_target)

                print('train clinical + radscore')
                # Clinical + RadScore
                train_crs = pd.concat([train_clinical, train_rad_score], axis=1)
                train_acc_lr_crs, lr_crs = lr(train_crs, train_target)
                train_acc_rf_crs, rf_crs = rf(train_crs, train_target)
                train_acc_svm_crs, svm_crs = svm(train_crs, train_target)
                train_acc_xgb_crs, xgb_crs = xgboost(train_crs, train_target)
                train_acc_mlp_crs, mlp_crs = mlp(train_crs, train_target)

                print('train clinical + radiomics')
                # Clinical + Radiomics
                train_cr = pd.concat([train_clinical, train_radiomics_lasso], axis=1)
                train_acc_lr_cr, lr_cr = lr(train_cr, train_target)
                train_acc_rf_cr, rf_cr = rf(train_cr, train_target)
                train_acc_svm_cr, svm_cr = svm(train_cr, train_target)
                train_acc_xgb_cr, xgb_cr = xgboost(train_cr, train_target)
                train_acc_mlp_cr, mlp_cr = mlp(train_cr, train_target)

                # Test Set
                test_clinical = clinical_data.iloc[test_index]
                test_radiomics = radiomics_features.iloc[test_index]
                test_target = test_clinical['target'].values

                test_clinical = test_clinical.reindex(columns=train_clinical.columns)
                test_radiomics_lasso = test_radiomics.reindex(columns=train_radiomics_lasso.columns)
                test_rad_score = radscore(test_radiomics_lasso, features)

                scaler = StandardScaler()
                scaled_radiomics_lasso = scaler.fit_transform(test_radiomics_lasso)
                test_radiomics_lasso.iloc[:, :] = scaled_radiomics_lasso
                scaled_rad_score = scaler.fit_transform(test_rad_score)
                test_rad_score.iloc[:, :] = scaled_rad_score

                # Test Models
                print('test lr')
                # lr
                test_clinical_pred = lr_clinical.predict(test_clinical)
                test_acc_lr_c = accuracy_score(test_target, test_clinical_pred)
                test_radiomics_pred = lr_radiomics.predict(test_radiomics_lasso)
                test_acc_lr_r = accuracy_score(test_target, test_radiomics_pred)
                test_rad_score_pred = lr_rad_score.predict(test_rad_score)
                test_acc_lr_rs = accuracy_score(test_target, test_rad_score_pred)
                test_crs = pd.concat([test_clinical, test_rad_score], axis=1)
                test_crs_pred = lr_crs.predict(test_crs)
                test_acc_lr_crs = accuracy_score(test_target, test_crs_pred)
                test_cr = pd.concat([test_clinical, test_radiomics_lasso], axis=1)
                test_cr_pred = lr_cr.predict(test_cr)
                test_acc_lr_cr = accuracy_score(test_target, test_cr_pred)

                results['fold'].append(fold)
                results['train_acc_lr_c'].append(train_acc_lr_c)
                results['test_acc_lr_c'].append(test_acc_lr_c)
                results['train_acc_lr_r'].append(train_acc_lr_r)
                results['test_acc_lr_r'].append(test_acc_lr_r)
                results['train_acc_lr_rs'].append(train_acc_lr_rs)
                results['test_acc_lr_rs'].append(test_acc_lr_rs)
                results['train_acc_lr_crs'].append(train_acc_lr_crs)
                results['test_acc_lr_crs'].append(test_acc_lr_crs)
                results['train_acc_lr_cr'].append(train_acc_lr_cr)
                results['test_acc_lr_cr'].append(test_acc_lr_cr)

                # rf
                print('test rf')
                test_clinical_pred = rf_clinical.predict(test_clinical)
                test_acc_rf_c = accuracy_score(test_target, test_clinical_pred)
                test_radiomics_pred = rf_radiomics.predict(test_radiomics_lasso)
                test_acc_rf_r = accuracy_score(test_target, test_radiomics_pred)
                test_rad_score_pred = rf_rad_score.predict(test_rad_score)
                test_acc_rf_rs = accuracy_score(test_target, test_rad_score_pred)
                test_crs = pd.concat([test_clinical, test_rad_score], axis=1)
                test_crs_pred = rf_crs.predict(test_crs)
                test_acc_rf_crs = accuracy_score(test_target, test_crs_pred)
                test_cr = pd.concat([test_clinical, test_radiomics_lasso], axis=1)
                test_cr_pred = rf_cr.predict(test_cr)
                test_acc_rf_cr = accuracy_score(test_target, test_cr_pred)

                results['train_acc_rf_c'].append(train_acc_rf_c)
                results['test_acc_rf_c'].append(test_acc_rf_c)
                results['train_acc_rf_r'].append(train_acc_rf_r)
                results['test_acc_rf_r'].append(test_acc_rf_r)
                results['train_acc_rf_rs'].append(train_acc_rf_rs)
                results['test_acc_rf_rs'].append(test_acc_rf_rs)
                results['train_acc_rf_crs'].append(train_acc_rf_crs)
                results['test_acc_rf_crs'].append(test_acc_rf_crs)
                results['train_acc_rf_cr'].append(train_acc_rf_cr)
                results['test_acc_rf_cr'].append(test_acc_rf_cr)

                # svm
                print('test svm')
                test_clinical_pred = svm_clinical.predict(test_clinical)
                test_acc_svm_c = accuracy_score(test_target, test_clinical_pred)
                test_radiomics_pred = svm_radiomics.predict(test_radiomics_lasso)
                test_acc_svm_r = accuracy_score(test_target, test_radiomics_pred)
                test_rad_score_pred = svm_rad_score.predict(test_rad_score)
                test_acc_svm_rs = accuracy_score(test_target, test_rad_score_pred)
                test_crs = pd.concat([test_clinical, test_rad_score], axis=1)
                test_crs_pred = svm_crs.predict(test_crs)
                test_acc_svm_crs = accuracy_score(test_target, test_crs_pred)
                test_cr = pd.concat([test_clinical, test_radiomics_lasso], axis=1)
                test_cr_pred = svm_cr.predict(test_cr)
                test_acc_svm_cr = accuracy_score(test_target, test_cr_pred)

                results['train_acc_svm_c'].append(train_acc_svm_c)
                results['test_acc_svm_c'].append(test_acc_svm_c)
                results['train_acc_svm_r'].append(train_acc_svm_r)
                results['test_acc_svm_r'].append(test_acc_svm_r)
                results['train_acc_svm_rs'].append(train_acc_svm_rs)
                results['test_acc_svm_rs'].append(test_acc_svm_rs)
                results['train_acc_svm_crs'].append(train_acc_svm_crs)
                results['test_acc_svm_crs'].append(test_acc_svm_crs)
                results['train_acc_svm_cr'].append(train_acc_svm_cr)
                results['test_acc_svm_cr'].append(test_acc_svm_cr)

                # xgb
                print('test xgb')
                test_clinical_pred = xgb_clinical.predict(test_clinical)
                test_acc_xgb_c = accuracy_score(test_target, test_clinical_pred)
                test_radiomics_pred = xgb_radiomics.predict(test_radiomics_lasso)
                test_acc_xgb_r = accuracy_score(test_target, test_radiomics_pred)
                test_rad_score_pred = xgb_rad_score.predict(test_rad_score)
                test_acc_xgb_rs = accuracy_score(test_target, test_rad_score_pred)
                test_crs = pd.concat([test_clinical, test_rad_score], axis=1)
                test_crs_pred = xgb_crs.predict(test_crs)
                test_acc_xgb_crs = accuracy_score(test_target, test_crs_pred)
                test_cr = pd.concat([test_clinical, test_radiomics_lasso], axis=1)
                test_cr_pred = xgb_cr.predict(test_cr)
                test_acc_xgb_cr = accuracy_score(test_target, test_cr_pred)

                results['train_acc_xgb_c'].append(train_acc_xgb_c)
                results['test_acc_xgb_c'].append(test_acc_xgb_c)
                results['train_acc_xgb_r'].append(train_acc_xgb_r)
                results['test_acc_xgb_r'].append(test_acc_xgb_r)
                results['train_acc_xgb_rs'].append(train_acc_xgb_rs)
                results['test_acc_xgb_rs'].append(test_acc_xgb_rs)
                results['train_acc_xgb_crs'].append(train_acc_xgb_crs)
                results['test_acc_xgb_crs'].append(test_acc_xgb_crs)
                results['train_acc_xgb_cr'].append(train_acc_xgb_cr)
                results['test_acc_xgb_cr'].append(test_acc_xgb_cr)

                # mlp
                print('test mlp')
                test_clinical_pred = mlp_clinical.predict(test_clinical)
                test_acc_mlp_c = accuracy_score(test_target, test_clinical_pred)
                test_radiomics_pred = mlp_radiomics.predict(test_radiomics_lasso)
                test_acc_mlp_r = accuracy_score(test_target, test_radiomics_pred)
                test_rad_score_pred = mlp_rad_score.predict(test_rad_score)
                test_acc_mlp_rs = accuracy_score(test_target, test_rad_score_pred)
                test_crs = pd.concat([test_clinical, test_rad_score], axis=1)
                test_crs_pred = mlp_crs.predict(test_crs)
                test_acc_mlp_crs = accuracy_score(test_target, test_crs_pred)
                test_cr = pd.concat([test_clinical, test_radiomics_lasso], axis=1)
                test_cr_pred = mlp_cr.predict(test_cr)
                test_acc_mlp_cr = accuracy_score(test_target, test_cr_pred)

                results['train_acc_mlp_c'].append(train_acc_mlp_c)
                results['test_acc_mlp_c'].append(test_acc_mlp_c)
                results['train_acc_mlp_r'].append(train_acc_mlp_r)
                results['test_acc_mlp_r'].append(test_acc_mlp_r)
                results['train_acc_mlp_rs'].append(train_acc_mlp_rs)
                results['test_acc_mlp_rs'].append(test_acc_mlp_rs)
                results['train_acc_mlp_crs'].append(train_acc_mlp_crs)
                results['test_acc_mlp_crs'].append(test_acc_mlp_crs)
                results['train_acc_mlp_cr'].append(train_acc_mlp_cr)
                results['test_acc_mlp_cr'].append(test_acc_mlp_cr)

        results_df = pd.DataFrame(results)
        for column in results_df.columns:
            mean_value = results_df[column].mean()
            print(f"The mean of {column} is {mean_value}")