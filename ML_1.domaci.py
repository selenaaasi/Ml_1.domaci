import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import statistics
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


def split_test_train(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train,X_test,y_train,y_test

def standard_scaler(X_train,X_test):
    sc_X = StandardScaler()
    X_train_scale = sc_X.fit_transform(X_train)
    X_test_scale = sc_X.transform(X_test)
    return X_train_scale,X_test_scale

#Univariate Selection - 2,3,4,5
def univariate_selection(X_train,y_train):
    bestfeatures = SelectKBest(score_func=chi2, k=5)
    fit = bestfeatures.fit(X_train, y_train)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X_train.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    print(featureScores.nlargest(4, 'Score'))

#RFE-2,5,6,15
def RFE_col(X_train,y_train):
    classifier = LogisticRegression(random_state=0)
    rfe = RFE(classifier, 4)
    fit = rfe.fit(X_train, y_train)
    print("Num Features: %d" % fit.n_features_)
    print("Selected Features: %s" % fit.support_)
    print("Feature Ranking: %s" % fit.ranking_)
    rfe_rank = fit.ranking_

#PCA 4,11,9,17
def PCA_col(X_train,y_train):
    pca_example=PCA().fit(X_train)
    print("Explained_variance_ratio")
    print(pca_example.explained_variance_ratio_)
    pca = PCA(n_components=4).fit(X_train)
    n_pcs = pca.components_.shape[0]
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
    initial_feature_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
                             '17']
    most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
    # using LIST COMPREHENSION HERE AGAIN
    dic = {'PC{}'.format(i + 1): most_important_names[i] for i in range(n_pcs)}
    # build the dataframe
    df = pd.DataFrame(sorted(dic.items()))
    print(df)

#Decision Tree-2,3,8,4
def decision_tree(X_train,y_train):
    model = ExtraTreesClassifier()
    model.fit(X_train, y_train)
    print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers
    # plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    print("Feature importance")
    print(feat_importances)
    plt.show()

#Logistic regression
def logistic_regression(X_train,y_train,X_test,y_test):
    classifier = LogisticRegression(random_state=42)
    classifier.fit(X_train, y_train)
    score = cross_val_score(classifier, X_test, y_test, cv=10)
    print("SCORE:")
    print(score)
    print("Average score:")
    print(statistics.mean(score))

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    #plt.rcParams['font.size'] = 12
    #plt.hist(score, bins=8)

    # x-axis limit from 0 to 1
    #plt.xlim(0, 1)
    #plt.title('Histogram of predicted probabilities')
    #plt.xlabel('Predicted probability of diabetes')
    #plt.ylabel('Frequency')
    #plt.show()

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

#Process data
def process_data(X,y):
    X_train, X_test, y_train, y_test = split_test_train(X, y)
    X_train_scale, X_test_scale = standard_scaler(X_train, X_test)
    logistic_regression(X_train_scale, y_train, X_test_scale, y_test)

if __name__ == "__main__":
    dataset = pd.read_csv("csv_result-diabetic_dataset.csv")

    # all col
    X = dataset.iloc[:, 1:19]
    y = dataset.iloc[:, 20]
    # random
    array = random.sample(range(0, 17), 4)
    X_random = dataset.iloc[:, [array[0], array[1], array[2],array[3]]]
    y_random = dataset.iloc[:, 20]

    # Univariate Selection
    X_unvariate_selection = dataset.iloc[:, [2, 3, 4, 5]]
    y_unvariate_selection = dataset.iloc[:, 20]

    # RFE
    X_rfe = dataset.iloc[:, [2,5, 6, 15]]
    y_rfe = dataset.iloc[:, 20]

    # PCA
    X_PCA = dataset.iloc[:, [4,9,11, 17]]
    y_PCA = dataset.iloc[:, 20]

    # DecisionTree
    X_dt = dataset.iloc[:, [2, 3, 4, 8]]
    y_dt = dataset.iloc[:, 20]

    X_train, X_test, y_train, y_test=split_test_train(X,y)
    univariate_selection(X_train, y_train)
    X_train_scale,X_test_scale=standard_scaler(X_train,X_test)
    RFE_col(X_train_scale,y_train)
    PCA_col(X_train_scale,y_train)
    decision_tree(X_train_scale,y_train)

    print("Sve kolone>>>>>>>>>")
    logistic_regression(X_train_scale,y_train,X_test_scale,y_test)
    print("Nasumicno izabranih 5 kolona>>>>>>")
    process_data(X_random,y_random)
    print("Univariate Selection>>>>>>")
    process_data(X_unvariate_selection, y_unvariate_selection)
    print("RFE>>>>>>")
    process_data(X_rfe,y_rfe)
    print("PCA>>>>>>")
    process_data(X_PCA,y_PCA)
    print("Decision tree>>>>>")
    process_data(X_dt,y_dt)


