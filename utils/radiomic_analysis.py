import os
from colorama import Fore, Back, Style
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV, ElasticNetCV, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import warnings
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
import seaborn as sns
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.exceptions import FitFailedWarning, ConvergenceWarning
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm

def PreprocessRadFeatures(radPath,clinicalPath, thrs):
    rad_features = pd.read_csv(radPath)
    rad_features = rad_features.dropna()
    cols_to_drop = [col for col in rad_features.columns if 'diagnostic' in col]
    cols_to_drop.append("Mask")
    rad_features = rad_features.drop(columns=cols_to_drop)
    rad_features["Image"] = rad_features["Image"].str[35:42]

    clinical_feature = pd.read_csv(clinicalPath)
    clinical_feature['target'] = (clinical_feature['TTP'] > 14).astype(float)

    rad_features['target'] = (clinical_feature['TTP'] > 14).astype(float)
    target = rad_features['target']
    mat = rad_features.drop(columns='target').corr(method='pearson')
    cluster_map = False
    if cluster_map:
        sns.set(style='white')
        clustermap = sns.clustermap(mat, method='average', cmap='vlag', linewidths=0.75, figsize=(200, 200))
        plt.setp(clustermap.ax_heatmap.get_yticklabels(), rotation=0)  # Keep y-labels horizontal
        plt.setp(clustermap.ax_heatmap.get_xticklabels(), rotation=90)  # Rotate x-labels for readability
        clustermap.savefig('clustermap.png', format='png')
        plt.show()
    cutoff_corr = thrs
    high_correlation_pairs = []
    for i in tqdm(range(len(mat.columns)), ascii=True, desc="Processing correlations"):
        # print(i)
        # Print feature pair being processed without interfering with tqdm bar
        for j in range(i + 1, len(mat.columns)):
            if abs(mat.iloc[i, j]) > cutoff_corr:
                v1 = mat.columns[i]
                v2 = mat.columns[j]
                correlation_value = mat.iloc[i, j]
                high_correlation_pairs.append((v1, v2, correlation_value))
    pair_data = pd.DataFrame(high_correlation_pairs, columns=['feature1','feature2','correlation value'])
    drop_list = []

    for index, row in tqdm(pair_data.iterrows(), total=pair_data.shape[0], ascii = True, desc=f"Evaluating features "):
        feature1 = row['feature1']
        feature2 = row['feature2']


        X_train1, X_test1, y_train, y_test = train_test_split(rad_features[feature1], target, test_size=0.3, random_state=seed)
        X_train2, X_test2, _, _ = train_test_split(rad_features[feature2], target, test_size=0.3, random_state=seed)

        model1 = LogisticRegression(max_iter=10000)
        model2 = LogisticRegression(max_iter=10000)

        model1.fit(X_train1.values.reshape(-1,1), y_train.values)
        model2.fit(X_train2.values.reshape(-1,1), y_train.values)

        y_pred1 = model1.predict_proba(X_test1.values.reshape(-1, 1))[:, 1]
        y_pred2 = model2.predict_proba(X_test2.values.reshape(-1, 1))[:, 1]

        auc1 = roc_auc_score(y_test, y_pred1)
        auc2 = roc_auc_score(y_test, y_pred2)

        if auc1 > auc2:
            drop_list.append(feature2)
        else:
            drop_list.append(feature1)

    drop_list = set(drop_list)
    rad_features = rad_features.drop(columns=drop_list)
    rad_features = rad_features.set_index('Image', drop=True).rename_axis(None)
    rad_features = rad_features.dropna(axis=0)
    print('Finished Preprocess of ', radPath)
    print('Features maintained: ', len(rad_features.columns))
    return rad_features
def PreprocessClinicalFeatures(csvPath):
    clinical_features = pd.read_csv(csvPath)
    clinical_features['target'] = (clinical_features['TTP'] > 14).astype(float)
    for feature in clinical_features:
        new_feature_name = feature.replace(" ","_")
        col = {feature:new_feature_name}
        clinical_features = clinical_features.rename(columns = col)
    clinical_features = clinical_features.drop(columns=['age', 'AFP'])
    clinical_features = clinical_features.rename(columns={'PS_bclc_0_0_1-2_1_3-4_3':'PS_bclc'})
    clinical_features = EncodeDF(clinical_features)
    clinical_features = clinical_features.drop(columns = 'Tr_Size')
    clinical_features = clinical_features.dropna(axis = 0)
    return clinical_features
def EncodeDF(df):
    elab_features = []
    for feature in df:
        if df[feature].unique().size <= 10:
            # print(feature)
            elab_features.append(feature)
            values = df[feature].unique()
            try:
                values = np.sort(values)
            except:
                print(Back.RED+'Array not sorted for feature:',feature,Style.RESET_ALL)
            replace = range(0,values.size)
            dict_replace = dict(zip(values,replace))
            df[feature] = df[feature].replace(dict_replace)
    return df
def class_features(df):
    cat_feat = []
    num_feat = []
    for feature in df:
        if df[feature].unique().size <= 10:
            cat_feat.append(feature)
        else:
            num_feat.append(feature)
    return cat_feat,num_feat
def IDMerge(df1,df2):
    id_c = df1['ID']
    id_r = df2['ID']
    ids = set(id_c).intersection(id_r)
    df1 = df1[df1['ID'].isin(ids)]
    df2 = df2[df2['ID'].isin(ids)]
    return df1,df2
def plot_coefficients(coefs, n_highlight):
    _, ax = plt.subplots(figsize=(9, 6))
    n_features = coefs.shape[0]
    alphas = coefs.columns
    for row in coefs.itertuples():
        ax.semilogx(alphas, row[1:], ".-", label=row.Index)

    alpha_min = alphas.min()
    top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
    for name in top_coefs.index:
        coef = coefs.loc[name, alpha_min]
        plt.text(alpha_min, coef, name + "   ", horizontalalignment="right", verticalalignment="center")

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("coefficient")
    plt.show()
def LassoAnalysis(x,y):
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FitFailedWarning)
    alphas = np.linspace(0.0001, 1, 1000)
    find_alpha = LassoCV(cv = 10, random_state = seed,
                         max_iter = 10000,
                         verbose = 0,
                         alphas=alphas).fit(x,y)
    lasso = Lasso(alpha=find_alpha.alpha_).fit(x,y)
    coef = list(zip(lasso.coef_, x))
    features = []
    coeff = []
    count = 0
    for value, item in coef:
        if value != 0:
            count += 1
    print(Fore.WHITE + Style.BRIGHT + "LASSO Analysis" + Style.RESET_ALL)
    print(Fore.GREEN + Style.BRIGHT + f"Number of non-zero coefficients: {count}" + Style.RESET_ALL)
    print(Fore.RED + Style.BRIGHT + f"Rad_Score: " + Style.RESET_ALL)
    print(Fore.CYAN + str('%.4f' % lasso.intercept_) + Style.RESET_ALL + ' + ')
    for value, item in coef:
        if value != 0:
            print(Fore.CYAN + str('%.4f' % value) + Style.RESET_ALL + ' * ' + Fore.YELLOW + str(item) + Style.RESET_ALL)
            features.append(item)
            coeff.append(value)
    features.append('intercept')
    coeff.append(lasso.intercept_)
    df = pd.DataFrame({"feature":features,"coeff":coeff})
    return df
def ElasticNetAnalysis(x,y):
    alphas = np.linspace(0.0001, 1, 1000)
    elastic_net = ElasticNetCV(cv=10, random_state=seed, alphas = alphas, max_iter= 100000, verbose=0).fit(x, y)
    optimal_alpha = elastic_net.alpha_
    optimal_l1_ratio = elastic_net.l1_ratio_
    elastic_net_optimal = ElasticNet(alpha=optimal_alpha, l1_ratio=optimal_l1_ratio)
    elastic_net_optimal.fit(x, y)
    coef = list(zip(elastic_net_optimal.coef_, x))
    features = []
    coeff = []
    count = 0
    for value, item in coef:
        if value != 0:
            count += 1
    print(Fore.WHITE + Style.BRIGHT + "ElasticNet Analysis" + Style.RESET_ALL)
    print(Fore.GREEN + Style.BRIGHT + f"Number of non-zero coefficients: {count}" + Style.RESET_ALL)
    print(Fore.RED + Style.BRIGHT + f"Rad_Score: " + Style.RESET_ALL)
    print(Fore.CYAN + str('%.4f' % elastic_net_optimal.intercept_) + Style.RESET_ALL + ' + ')
    for value, item in coef:
        if value != 0:
            print(Fore.CYAN + str('%.4f' % value) + Style.RESET_ALL + ' * ' + Fore.YELLOW + str(item) + Style.RESET_ALL)
            features.append(item)
            coeff.append(value)
    features.append('intercept')
    coeff.append(elastic_net_optimal.intercept_)
    df = pd.DataFrame({"feature": features, "coeff": coeff})
    return df
def RadScore(x,y,features):

    f = x[features['feature'][:-1]]
    c = features['coeff'][:-1].values
    rad_score = f * c
    rad_score['sum'] = rad_score.sum(axis=1)
    rad_score['score'] = rad_score['sum'] + features['coeff'].values[-1]
    x['target'] = y
    x['rad_score'] = rad_score['score']
    return x
def Tsne2D(df):
    warnings.simplefilter("ignore", FutureWarning)
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    features = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    tsne_2d = TSNE(n_components=2, random_state=1969)
    tsne_2d_results = tsne_2d.fit_transform(features)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(tsne_2d_results[:, 0], tsne_2d_results[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(scatter)
    plt.title('t-SNE 2D')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    plt.show()
def Tsne3D(df):
    warnings.simplefilter("ignore", FutureWarning)
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    features = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    tsne_3d = TSNE(n_components=3, random_state=1969)
    tsne_3d_results = tsne_3d.fit_transform(features)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(tsne_3d_results[:, 0], tsne_3d_results[:, 1], tsne_3d_results[:, 2], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(scatter)
    ax.set_title('t-SNE 3D')
    ax.set_xlabel('t-SNE component 1')
    ax.set_ylabel('t-SNE component 2')
    ax.set_zlabel('t-SNE component 3')
    plt.show()
def lr(df, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    x = df[[name for name in df if 'target' not in name]]
    param_grid = {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}
    grid = GridSearchCV(LogisticRegression(max_iter=10000), param_grid, cv=skf, scoring='accuracy')
    grid.fit(x, y)
    best_model = grid.best_estimator_
    log_reg_accuracy = cross_val_score(best_model, x, y, cv=skf, scoring='accuracy')
    log_reg_auc = cross_val_score(best_model, x, y, cv=skf, scoring='roc_auc')

    print(f"Logistic Regression --> {log_reg_accuracy.mean():.4f}      {log_reg_auc.mean():.4f}")
def svm(df, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    x = df[[name for name in df if 'target' not in name]]
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    grid = GridSearchCV(SVC(probability=True), param_grid, cv=skf, scoring='accuracy')
    grid.fit(x, y)
    best_model = grid.best_estimator_
    svm_accuracy = cross_val_score(best_model, x, y, cv=skf, scoring='accuracy')
    svm_auc = cross_val_score(best_model, x, y, cv=skf, scoring='roc_auc')
    print(f"SVM                 --> {svm_accuracy.mean():.4f}      {svm_auc.mean():.4f}")
def rf(df, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    x = df[[name for name in df if 'target' not in name]]
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
    grid = GridSearchCV(RandomForestClassifier(random_state=1969), param_grid, cv=skf, scoring='accuracy')
    grid.fit(x, y)
    best_model = grid.best_estimator_
    rf_accuracy = cross_val_score(best_model, x, y, cv=skf, scoring='accuracy')
    rf_auc = cross_val_score(best_model, x, y, cv=skf, scoring='roc_auc')
    print(f"Random Forest       --> {rf_accuracy.mean():.4f}      {rf_auc.mean():.4f}")
def xgb(df, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    x = df[[name for name in df if 'target' not in name]]
    param_grid = {'n_estimators': [50, 100, 200]}
    grid = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_grid, cv=skf, scoring='accuracy')
    grid.fit(x, y)
    best_model = grid.best_estimator_
    xgb_accuracy = cross_val_score(best_model, x, y, cv=skf, scoring='accuracy')
    xgb_auc = cross_val_score(best_model, x, y, cv=skf, scoring='roc_auc')
    print(f"XGBoost             --> {xgb_accuracy.mean():.4f}      {xgb_auc.mean():.4f}")
def mlp(df, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    x = df[[name for name in df if 'target' not in name]]
    param_grid = {'hidden_layer_sizes': [(32,), (32,16), (64,)]}
    grid = GridSearchCV(MLPClassifier(max_iter=10000, random_state=1969), param_grid, cv=skf, scoring='accuracy')
    grid.fit(x, y)
    best_model = grid.best_estimator_
    mlp_accuracy = cross_val_score(best_model, x, y, cv=skf, scoring='accuracy')
    mlp_auc = cross_val_score(best_model, x, y, cv=skf, scoring='roc_auc')
    print(f"MLP                 --> {mlp_accuracy.mean():.4f}      {mlp_auc.mean():.4f}")

if __name__ == '__main__':
    seed = 1969
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category= ConvergenceWarning)
    type = "liver"
    thr = 0.85
    thr_s = f"{int(thr * 100):03d}"
    if type == "liver":
        file = "AUC_feature_liver_"+thr_s+".csv"
    else:
        file = "AUC_feature_" + thr_s + ".csv"
    clinical_path = "C:\\Users\\marvu\\Desktop\\GitHub\\RadTACE\\file\\clinical_data.csv"
    if file not in os.listdir("C:\\Users\\marvu\\Desktop\\GitHub\\RadTACE\\file\\reducted\\"):
        if type == "liver":
            radiomics_csv = "radiomics_features_liver.csv"
        else:
            radiomics_csv = "radiomics_features.csv"
        print('Preprocessing of file ',radiomics_csv)
        full_path = "C:\\Users\\marvu\\Desktop\\GitHub\\RadTACE\\file\\" + radiomics_csv
        RadFeatures = pd.read_csv(full_path)
        RadFeatures = PreprocessRadFeatures(full_path,clinical_path, thr)
        RadFeatures.to_csv("C:\\Users\\marvu\\Desktop\\GitHub\\RadTACE\\file\\reducted\\AUC_feature_liver"+thr_s+".csv")
        RadFeatures = RadFeatures.reset_index()
        RadFeatures = RadFeatures.rename(columns={"index": "ID"})
    else:
        print("Read file ",file)
        if type == "liver":
            RadFeatures = pd.read_csv("C:\\Users\\marvu\\Desktop\\GitHub\\RadTACE\\file\\reducted\\AUC_feature_liver_"+thr_s+".csv")
        else:
            RadFeatures = pd.read_csv("C:\\Users\\marvu\\Desktop\\GitHub\\RadTACE\\file\\reducted\\AUC_feature_"+thr_s+".csv")
        RadFeatures = RadFeatures.rename(columns={"Unnamed: 0": "ID"})

    _ , num_cols = class_features(RadFeatures)
    num_cols = num_cols[1:]
    scaler = StandardScaler().fit(RadFeatures[num_cols])
    RadFeatures[num_cols] = scaler.transform(RadFeatures[num_cols])
    ClinicalFeatures = PreprocessClinicalFeatures(clinical_path)
    RadFeatures, ClinicalFeatures = IDMerge(RadFeatures,ClinicalFeatures)

    RadFeatures = RadFeatures.reset_index()
    RadFeatures = RadFeatures.drop( columns = 'index')
    ClinicalFeatures = ClinicalFeatures.reset_index()
    ClinicalFeatures = ClinicalFeatures.drop( columns = 'index')
    # Divided the dataset in train and test
    clinic = ClinicalFeatures['hepatitis']
    target = RadFeatures['target'].values
    features = RadFeatures.drop(columns=['ID','target'])
    lasso1 = LassoAnalysis(features,target)
    en1 = ElasticNetAnalysis(features,target)
    data = RadScore(features,target,lasso1)
    x_train,x_test,y_train,y_test = train_test_split(features,target,test_size=0.2,random_state=seed)



    data_score=RadScore(x_train,y_train,lasso1)
    data_train = data_score[['target','rad_score']]
    data_train_sort = data_train.sort_values(by = 'rad_score')
    fpr, tpr, thresholds = roc_curve(data_train_sort['target'],data_train_sort['rad_score'])
    youden_index = tpr + (1-fpr) -1
    optimal_threshold = thresholds[np.argmax(youden_index)]
    y_pred = (data_train['rad_score'] >= optimal_threshold).astype(int)
    accuracy_train = np.mean(y_pred == data_train['target'])
    auc_train = auc(fpr, tpr)

    data_score_test = RadScore(x_test,y_test,lasso1)
    data_test = data_score_test[['target','rad_score']]
    data_test_sort = data_test.sort_values(by = 'rad_score')
    fpr, tpr, thresholds = roc_curve(data_test_sort['target'], data_test_sort['rad_score'])
    youden_index = tpr + (1 - fpr) - 1
    optimal_threshold = thresholds[np.argmax(youden_index)]
    y_pred = (data_test['rad_score'] >= optimal_threshold).astype(int)
    accuracy_test = np.mean(y_pred == data_test['target'])
    auc_test = auc(fpr, tpr)


    # Clinical variable retained after UA and MA is only 'hepatitis'
    c_train,c_test,cy_train,cy_test = train_test_split(clinic,target,test_size=0.2,random_state=seed)
    clin_radscore = pd.DataFrame({"hepatitis":clinic.values,"rad_score":data['rad_score'].values})
    clin_radscore_train = pd.DataFrame({"hepatitis":c_train.values,"rad_score":data_train['rad_score'].values})
    clin_radscore_test = pd.DataFrame({"hepatitis":c_test.values,"rad_score":data_test['rad_score'].values})

    # Radiomcs varibale foundend in LASSO analysis
    radiomics_feature = features[lasso1['feature'][:-1]]
    radiomics_feature_train = x_train[lasso1['feature'][:-1]]
    radiomics_feature_test = x_test[lasso1['feature'][:-1]]

    #Combined Radiomics + Clinical features
    combined = radiomics_feature.copy()
    combined['hepatitis'] = clinic.values
    combined_train = radiomics_feature_train.copy()
    combined_train['hepatitis'] = c_train
    combined_test = radiomics_feature_test.copy()
    combined_test['hepatitis'] = c_test

    #train and test models
    print("                        Accuracy    AUC")
    print(Fore.LIGHTWHITE_EX + Style.BRIGHT + "Clinic + Rad_Score" + Style.RESET_ALL)
    lr(clin_radscore,target)
    svm(clin_radscore,target)
    rf(clin_radscore,target)
    xgb(clin_radscore,target)
    mlp(clin_radscore,target)
    print(Fore.LIGHTWHITE_EX + Style.BRIGHT + "Radiomics Features" + Style.RESET_ALL)
    lr(radiomics_feature,target)
    svm(radiomics_feature,target)
    rf(radiomics_feature,target)
    xgb(radiomics_feature,target)
    mlp(radiomics_feature,target)
    print(Fore.LIGHTWHITE_EX + Style.BRIGHT + "Clinic + Radiomics Features" + Style.RESET_ALL)
    lr(combined,target)
    svm(combined,target)
    rf(combined,target)
    xgb(combined,target)
    mlp(combined,target)

    a = 0

