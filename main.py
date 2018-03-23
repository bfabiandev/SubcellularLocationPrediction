import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RepeatedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, f1_score, precision_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from time import time

from preprocess import prepare_all_data

blind_ids = ["SEQ677", "SEQ231", "SEQ871", "SEQ388", "SEQ122", "SEQ758", "SEQ333", "SEQ937", "SEQ351",
             "SEQ202", "SEQ608", "SEQ402", "SEQ433", "SEQ821", "SEQ322", "SEQ982", "SEQ951", "SEQ173",
             "SEQ862", "SEQ224"]

names = ['Cytosolic', 'Mitochondrial', 'Nuclear', 'Secreted']


def visualize_data(x, y):
    pca = PCA(n_components=2)

    order = np.random.permutation(range(x.shape[0]))
    x_shuf = x[order]
    y_shuf = y[order]
    principalComponents = pca.fit_transform(x_shuf)
    print('These two axes account for {} of the variance'.format(
        sum(pca.explained_variance_ratio_)))
    principalDf = pd.DataFrame(data=principalComponents, columns=[
                               'principal component 1', 'principal component 2'])
    df = pd.DataFrame(data=y_shuf, columns=['target'])
    finalDf = pd.concat([principalDf, df[['target']]], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = [0, 1, 2, 3]
    colors = ['r', 'g', 'b', 'black']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                   finalDf.loc[indicesToKeep, 'principal component 2'], c=color, s=10)
    ax.legend(names)
    ax.grid()
    plt.show()
    fig.savefig('pca.png')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def evaluate_model(train_x, train_y, model=LogisticRegression(n_jobs=-1), kfolds=5, verbose=False):
    print(model)

    kf = KFold(n_splits=kfolds, shuffle=True)
    losses = []
    accs = []
    f1s = []
    print_conf_matrix = True
    for train_index, dev_index in kf.split(train_x):
        x_train, x_dev = train_x[train_index], train_x[dev_index]
        y_train, y_dev = train_y[train_index], train_y[dev_index]

        model.fit(x_train, y_train.ravel())  # Train classifier
        pred_proba = model.predict_proba(x_dev)  # Make predictions
        preds = model.predict(x_dev)
        train_preds = model.predict(x_train)

        # Evaluate log loss on this split
        loss = log_loss(y_dev, pred_proba, labels=[0, 1, 2, 3])
        print("Log loss = {}".format(loss))

        acc = accuracy_score(y_dev.ravel(), preds)
        print("Accuracy = {}".format(acc))

        f1 = f1_score(y_dev.ravel(), preds, average='weighted')
        print("F1 = {}".format(f1))

        acc_train = accuracy_score(y_train.ravel(), train_preds)
        print("Train acc = {}".format(acc_train))
        print()

        if verbose and print_conf_matrix:
            print_conf_matrix = False
            cm = confusion_matrix(y_dev, preds, labels=[0, 1, 2, 3])
            fig = plt.figure()
            plot_confusion_matrix(
                cm, names, normalize=True)
            plt.show()
            fig.savefig('cm.png')

        losses.append(loss)
        accs.append(acc)
        f1s.append(f1)

    print("Mean loss = {}".format(np.mean(losses)))
    print("Mean acc = {}".format(np.mean(accs)))
    print("Mean f1 = {}".format(np.mean(f1s)))

    model.fit(x_train, y_train.ravel())


def get_predictions(model, test_x):
    return model.predict_proba(test_x)


def plot_counts(y):
    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts)))

    fig, ax = plt.subplots()
    plt.bar(unique, counts)
    plt.xticks(unique, names)
    plt.show()
    fig.savefig('counts.png')


def main(use_features_if_available=True, verbose=False):
    if not use_features_if_available:
        print("Started collecting all features...")
        df_final_train_x, df_final_train_y, df_final_test_x = prepare_all_data(
            verbose=verbose)

        df_final_train_x.to_pickle('df_final_train_x.pkl')
        df_final_train_y.to_pickle('df_final_train_y.pkl')
        df_final_test_x.to_pickle('df_final_test_x.pkl')
    else:
        try:
            print('Trying to load from saved data...')

            df_final_train_x = pd.read_pickle('df_final_train_x.pkl')
            df_final_train_y = pd.read_pickle('df_final_train_y.pkl')
            df_final_test_x = pd.read_pickle('df_final_test_x.pkl')
        except IOError:
            print('Couldn\'t find the saved feature matrices... Started collecting it.')
            df_final_train_x, df_final_train_y, df_final_test_x = prepare_all_data(
                verbose=verbose)

            df_final_train_x.to_pickle('df_final_train_x.pkl')
            df_final_train_y.to_pickle('df_final_train_y.pkl')
            df_final_test_x.to_pickle('df_final_test_x.pkl')

    train_x = df_final_train_x.as_matrix()
    train_y = df_final_train_y.as_matrix()
    test_x = df_final_test_x.as_matrix()

    if verbose:
        plot_counts(train_y)
        visualize_data(train_x, train_y)

    model0 = KNeighborsClassifier(n_neighbors=10)
    model1 = LogisticRegression(penalty='l1', C=1, random_state=0)
    model2 = GradientBoostingClassifier(
        n_estimators=200, max_features='auto', random_state=0)
    model3 = AdaBoostClassifier(
        n_estimators=200, learning_rate=1.0, random_state=0)
    model4 = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=3, n_jobs=-1, random_state=0)
    model5 = ExtraTreesClassifier(
        n_estimators=200, max_features=None, min_samples_split=3, n_jobs=-1, random_state=0)

    model6 = SVC(probability=True, random_state=0)
    model7 = MLPClassifier(hidden_layer_sizes=(
        64, 32, 32), early_stopping=True, random_state=0)

    modelVoting1 = VotingClassifier(
        [('gb', model2), ('et', model5), ('lr', model1)], voting='soft', n_jobs=-1)

    modelVoting = VotingClassifier(
        [('gb', model2), ('et', model5)], voting='soft', n_jobs=-1)

    model = modelVoting

    evaluate_model(model=model, train_x=train_x,
                   train_y=train_y, verbose=verbose)

    preds = get_predictions(model, test_x)

    names_short = ['Cyto', 'Mito', 'Nucl', 'Secr']
    for idx, row in enumerate(preds):
        print('{} {} Confidence {}%'.format(
            blind_ids[idx], names_short[np.argmax(row)], round(100*max(row), 2)))

    try:
        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")

        X = train_x

        for f in range(X.shape[1]):
            print("%d. feature %s (%f)" % (
                f + 1, df_final_train_x.columns[indices[f]], importances[indices[f]]))

        # Plot the feature importances of the forest
        fig = plt.figure()
        plt.title("Feature importances")
        plt.bar(range(20), importances[indices][:20],
                color="b", yerr=std[indices][:20], align="center")
        plt.xticks(
            range(20), df_final_train_x.columns[indices][:20], rotation=90)
        plt.xlim([-1, 20])
        plt.tight_layout()
        if verbose:
            plt.show()
        fig.savefig('importances.png')
    except:
        pass


if __name__ == '__main__':
    main(use_features_if_available=True, verbose=True)
