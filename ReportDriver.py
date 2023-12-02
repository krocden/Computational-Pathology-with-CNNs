import copy
import torch
import torchvision.models as models
from torch.utils.data import TensorDataset
import _init
import ImgLoader
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def choose_pth(seeds):
    model = models.resnet18(weights=None)
    fc_in = model.fc.in_features
    model.fc = torch.nn.Linear(fc_in, out_features=3)
    model.load_state_dict(torch.load(f"Model/Saved/resnet18_Seed={seeds}.pth"))
    model = model.to(config.device)
    model.eval()
    return model


def choose_pertrain():
    # TODO 2 use other model maybe, you can try for the best pertrained weight and architecture
    model = models.resnet18(weights="IMAGENET1K_V1")
    fc_in = model.fc.in_features
    model.fc = torch.nn.Linear(fc_in, out_features=3)
    model = model.to(config.device)
    model.eval()
    return model


def giveClassificationReport(y_trn, train_pred, y_tst, test_pred):
    print("Classification Report on Training Data")
    print(classification_report(y_trn, train_pred, zero_division=0.0))

    print("Classification Report on Testing Data")
    print(classification_report(y_tst, test_pred, zero_division=0.0))


def give_feature_set(model, test_data):
    features = []
    labels = []
    model_feature_extract = copy.deepcopy(model)
    model_feature_extract.fc = torch.nn.Identity()
    model_feature_extract.eval()
    with torch.no_grad():
        for X, y in test_data:
            X = X.to(config.device)
            features.extend(model_feature_extract(X).cpu().numpy())
            labels.extend(y.cpu().numpy())
    features = np.array(features)
    labels = np.array(labels)
    return features, labels


def visualize_TSNE(X, y):
    print("now generating t-SNE output")
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(X)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y, cmap='viridis')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("t-SNE of Model Features")
    plt.show()


def draw_confusing_Matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()


def use_Kmean(X_trn, y_trn, k, X_tst, y_tst):
    print("now using K-mean to predict")
    kmean = KNeighborsClassifier(n_neighbors=k)
    kmean.fit(X_trn, y_trn)
    trn_pred = kmean.predict(X_trn)
    tst_pred = kmean.predict(X_tst)
    giveClassificationReport(y_trn, trn_pred, y_tst, tst_pred)


def use_random_forest(X_trn, y_trn, k, X_tst, y_tst):
    print("now using random forest to predict")
    forest = RandomForestClassifier(n_estimators=k)
    forest.fit(X_trn, y_trn)
    trn_pred = forest.predict(X_trn)
    tst_pred = forest.predict(X_tst)
    giveClassificationReport(y_trn, trn_pred, y_tst, tst_pred)


def use_SVM(X_trn, y_trn, X_tst, y_tst):
    print("now using SVM to predict")
    svm = SVC()
    svm.fit(X_trn, y_trn)
    trn_pred = svm.predict(X_trn)
    tst_pred = svm.predict(X_tst)
    giveClassificationReport(y_trn, trn_pred, y_tst, tst_pred)


def use_adaBoost(X_trn, y_trn, k, X_tst, y_tst):
    print("now using adaBoost to predict")
    boost = AdaBoostClassifier(n_estimators=k)
    boost.fit(X_trn, y_trn)
    trn_pred = boost.predict(X_trn)
    tst_pred = boost.predict(X_tst)
    giveClassificationReport(y_trn, trn_pred, y_tst, tst_pred)


if __name__ == '__main__':
    os.environ['LOKY_MAX_CPU_COUNT'] = '8'  # this set the concurrent core for generating the graph explicitly
    config = _init.Config()
    # TODO 1 to have a reasonable model from task1 (you may need to import the .pth manually)
    seed = 6436  # change the seed to any Saved seed you want to load
    config.set_seed(seed)

    model_tsk1 = choose_pth(seed)
    model_pertrained = choose_pertrain()

    dataset_path1 = "datasets/Dataset 1/Dataset 1/Colorectal Cancer"  # shouldn't use as conclusion, only used in checking
    dataset_path2 = "datasets/Dataset 2/Dataset 2/Prostate Cancer"  # for scenario 1 and 2
    dataset_path3 = "datasets/Dataset 3/Dataset 3/Animal Faces"  # for scenario 3 and 4
    # TODO 3 change the loader to load the dataset 2 and 3 for our 4 different scenarios
    loader = ImgLoader.Loader(config, dataset_path1, shuffle=True)
    train_set, test_set = loader.get_dataloader(train_ratio=0.7)
    # TODO 3.5 this part can be commented since this is only for testing
    model_tsk1.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for X, y in test_set:
            X = X.to(config.device)
            outputs = model_tsk1(X)
            _, tensor_pred = torch.max(outputs.data, 1)
            labels.extend(y.cpu().numpy())
            predictions.extend(tensor_pred.cpu().numpy())

    # print("Length of true labels:", len(labels))
    # print("Length of predicted labels:", len(predictions))
    print(classification_report(labels, predictions, zero_division=0.0))
    draw_confusing_Matrix(labels, predictions)

    # TODO 4 in 4 different scenarios, you will need to get the feature from both our model and the pertrained one then visualize them
    trn_feature, trn_label = give_feature_set(model_pertrained, train_set)
    tst_feature, tst_label = give_feature_set(model_pertrained, test_set)
    visualize_TSNE(trn_feature, trn_label)

    # TODO 5 choose 1 from the 4 scenarios use the extracted features and labels in TODO4 then in 2 different
    #  traditional ML models, one must be K_mean and the other one could be any. Though, 3 different ML models are
    #  already defined above, use them


    # TODO 6 get the result and finish our report