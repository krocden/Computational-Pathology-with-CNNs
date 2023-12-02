import copy
import torch
import torchvision.models as models
import _init
import ImgLoader
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os


if __name__ == '__main__':
    os.environ['LOKY_MAX_CPU_COUNT'] = '8'  # this set the concurrent core for generating the graph explicitly
    config = _init.Config()
    seed = 28506  # change the seed to any Saved seed you want to load
    config.set_seed(seed)
    model = models.resnet18(weights=None)
    fc_in = model.fc.in_features
    model.fc = torch.nn.Linear(fc_in, out_features=3)
    model.load_state_dict(torch.load(f"Model/Saved/resnet18_Seed={seed}.pth"))
    model = model.to(config.device)
    model.eval()
    true_labels = []

    image_path = "datasets/Dataset 1/Dataset 1/Colorectal Cancer"
    loader = ImgLoader.Loader(config, image_path, shuffle=True)
    _, test_data = loader.get_dataloader(train_ratio=0.7)

    predictions = []
    with torch.no_grad():
        for images, labels in test_data:
            images = images.to(config.device)
            outputs = model(images)
            _, tensor_pred = torch.max(outputs.data, 1)
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(tensor_pred.cpu().numpy())

    print("Length of true labels:", len(true_labels))
    print("Length of predicted labels:", len(predictions))
    print(classification_report(true_labels, predictions))

    features = []
    model_feature_extract = copy.deepcopy(model)
    model_feature_extract.fc = torch.nn.Identity()
    with torch.no_grad():
        for images, _ in test_data:
            images = images.to(config.device)
            features.extend(model_feature_extract(images).cpu().numpy())

    print("now generating t-SNE output")
    features_array = np.array(features)
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(features_array)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=true_labels, cmap='viridis')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("t-SNE of Model Features")
    plt.show()

