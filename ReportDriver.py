import copy
import torch
import torchvision.models as models
from torch.utils.data import TensorDataset
import _init
from os import walk
import ImgLoader
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
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
    model = models.resnet18(weights="IMAGENET1K_V1")
    fc_in = model.fc.in_features
    model.fc = torch.nn.Linear(fc_in, out_features=3)
    model = model.to(config.device)
    model.eval()
    return model

def giveClassificationReport(y_trn, train_pred, y_tst, test_pred, title):
    print("Classification Report on Training Data")
    train_report = classification_report(y_trn, train_pred, zero_division=0.0)
    print(train_report)

    print("Classification Report on Testing Data")
    test_report = classification_report(y_tst, test_pred, zero_division=0.0)
    print(test_report)

    with open("Model/Saved/perf_metrics", 'a') as file:
        perf_metrics = f"{title}: \n {train_report} \n {test_report}"
        file.write(perf_metrics + '\n\n')


def give_feature_set(model, test_data):
    features = []
    labels = []
    model_feature_extract = copy.deepcopy(model)
    model_feature_extract.fc = torch.nn.Identity()
    model
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
    print("Generating t-SNE output...")
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(X)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y, cmap='viridis')
    plt.legend(handles=scatter.legend_elements(num=[0,1,2])[0], labels=datasetClasses,  title="Classes")
    if usingPretrained:
        plt.title(f"t-SNE of pretrained model features on dataset {loader.directory.split('/')[1][-1]}: {loader.directory.split('/')[2]}")
    else:
        plt.title(f"t-SNE of model{seed} features on dataset {loader.directory.split('/')[1][-1]}: {loader.directory.split('/')[2]}")
    plt.show()

def draw_confusion_Matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='.1f', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title(f'Confusion Matrix: {title}')
    plt.show()

def use_KNN(X_train, y_train, k, X_test, y_test):
    print('Running KNN...')
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_predtrain = knn.predict(X_train)
    y_predtest = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_predtest)
    print(f"Accuracy: {accuracy:.4f}")
    giveClassificationReport(y_train, y_predtrain, y_test, y_predtest,f"KNN - Pretrained - D{loader.directory.split('/')[1][-1]}" if  usingPretrained else f"KNN - {seed} - D{loader.directory.split('/')[1][-1]}")
    draw_confusion_Matrix(y_train, y_predtrain, f"train set on dataset {loader.directory.split('/')[1][-1]}: {loader.directory.split('/')[2]}")
    draw_confusion_Matrix(y_test, y_predtest, f"test set on dataset {loader.directory.split('/')[1][-1]}: {loader.directory.split('/')[2]}")
    print('Finished running KNN')

def use_SVM(X_train, y_train, X_test, y_test):
    print("Running SVM...")
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
    X_train = scaling.transform(X_train)
    X_test = scaling.transform(X_test)
    svm = SVC()
    svm.fit(X_train, y_train)
    trn_pred = svm.predict(X_train)
    tst_pred = svm.predict(X_test)
    giveClassificationReport(y_train, trn_pred, y_test, tst_pred, f"SVM - Pretrained - D{loader.directory.split('/')[1][-1]}" if  usingPretrained else f"SVM - {seed} - D{loader.directory.split('/')[1][-1]}")
    draw_confusion_Matrix(y_train, trn_pred, f"train set on dataset {loader.directory.split('/')[1][-1]}: {loader.directory.split('/')[2]}")
    draw_confusion_Matrix(y_test, tst_pred, f"test set on dataset {loader.directory.split('/')[1][-1]}: {loader.directory.split('/')[2]}")
    print('Finished running SVM')

if __name__ == '__main__':
    os.environ['LOKY_MAX_CPU_COUNT'] = '8'  # this set the concurrent core for generating the graph explicitly
    config = _init.Config()
    # change the seed to any Saved seed you want to load (model/saved/)
    seed = 6436  
    config.set_seed(seed)

    model_tsk1 = choose_pth(seed)
    model_pertrained = choose_pertrain()

    dataset_path1 = "datasets/Dataset 1/Colorectal Cancer"
    dataset_path2 = "datasets/Dataset 2/Prostate Cancer"
    dataset_path3 = "datasets/Dataset 3/Animal Faces"

    print('Loading images...')
    # Change dataset to Dataset 2 or Dataset 3
    datasetClasses = [f.name for f in os.scandir(dataset_path3) if f.is_dir()]
    loader = ImgLoader.Loader(config, dataset_path3, shuffle=True)
    # Change to pretrained model else use self trained model
    usingPretrained = False
    train_set, test_set = loader.get_dataloader(train_ratio=0.7)
    print('Finished loading images')

    # For Testing
    """
    model_tsk1.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for X, y in train_set:
            X = X.to(config.device)
            outputs = model_tsk1(X)
            _, tensor_pred = torch.max(outputs.data, 1)
            labels.extend(y.cpu().numpy())
            predictions.extend(tensor_pred.cpu().numpy())

    print("Length of true labels:", len(labels))
    print("Length of predicted labels:", len(predictions))

    classif_report = classification_report(labels, predictions, zero_division=0.0)
    draw_confusion_Matrix(labels, predictions, f"train set on dataset {loader.directory.split('/')[1][-1]}: {loader.directory.split('/')[2]}")

    with open("Model/Saved/perf_metrics", 'a') as file:
        perf_metrics = f"Model - {seed} - TrainD{loader.directory.split('/')[1][-1]}: \n {classif_report}"
        file.write(perf_metrics + '\n\n')

    predictions = []
    labels = []
    with torch.no_grad():
        for X, y in test_set:
            X = X.to(config.device)
            outputs = model_tsk1(X)
            _, tensor_pred = torch.max(outputs.data, 1)
            labels.extend(y.cpu().numpy())
            predictions.extend(tensor_pred.cpu().numpy())

    print("Length of true labels:", len(labels))
    print("Length of predicted labels:", len(predictions))

    classif_report = classification_report(labels, predictions, zero_division=0.0)
    draw_confusion_Matrix(labels, predictions, f"test set on dataset {loader.directory.split('/')[1][-1]}: {loader.directory.split('/')[2]}")

    with open("Model/Saved/perf_metrics", 'a') as file:
        perf_metrics = f"Model - {seed} - TestD{loader.directory.split('/')[1][-1]}: \n {classif_report}"
        file.write(perf_metrics + '\n\n')
    """

    print("Getting model features...")
    trn_feature, trn_label = give_feature_set(model_pertrained if usingPretrained else model_tsk1, train_set)
    tst_feature, tst_label = give_feature_set(model_pertrained if usingPretrained else model_tsk1, test_set)
    print("Got model features")

    # t-SNE
    visualize_TSNE(trn_feature, trn_label)

    # KNN
    use_KNN(trn_feature, trn_label, 3, tst_feature, tst_label)

    # SVM
    use_SVM(trn_feature, trn_label, tst_feature, tst_label)
