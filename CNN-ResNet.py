import os
import torch
import torchvision.models as model
import _init
import ImgLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    glb = _init.Config()
    seed = glb.set_seed()
    resnet = model.resnet18(weights=None)
    fc_in = resnet.fc.in_features
    resnet.fc = torch.nn.Linear(fc_in, out_features=3)

    resnet = resnet.to(glb.device)

    # should change this to tweak the model
    hyperparam = {
        'learning rate': 1e-5,
        'epoch number': 7,
        'batch size': 30,
    }

    image_path = "datasets/Dataset 1/Colorectal Cancer"

    #current_script_path = os.path.dirname(os.path.abspath(__file__))
    #parent_dir = os.path.dirname(current_script_path)
    #datasets_path = os.path.join(parent_dir, image_path)

    loader = ImgLoader.Loader(glb, image_path, shuffle=True, batch_size=hyperparam['batch size'])

    train_data, test_data = loader.get_dataloader(train_ratio=0.7)

    optimizer = torch.optim.Adam(resnet.parameters(), lr=hyperparam['learning rate'])
    loss = torch.nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    train_accuracy = []
    val_accuracy = []

    num_epochs = hyperparam['epoch number']
    for epoch in range(num_epochs):
        resnet.train()
        for inputs, labels in train_data:
            inputs = inputs.to(glb.device)
            labels = labels.to(glb.device)
            optimizer.zero_grad()
            outputs = resnet(inputs)
            l = loss(outputs, labels)
            train_losses.append(l.item())
            l.backward()
            optimizer.step()
        print(f"in epoch {epoch}, the loss is: {l.item()}")
        
        #torch.save(resnet.state_dict(), f'Model/Saved/resnet18_Seed={seed}_epoch={epoch}.pth')
        
        # Accuracy
        with torch.no_grad():
            resnet.eval()
            all_train_labels = []
            all_train_preds = []
            for inputs, labels in train_data:
                inputs = inputs.to(glb.device)
                labels = labels.to(glb.device)
                outputs = resnet(inputs)
                _, predicted = torch.max(outputs, 1)
                all_train_labels.extend(labels.cpu().numpy())
                all_train_preds.extend(predicted.cpu().numpy())

        trainAccuracy = accuracy_score(all_train_labels, all_train_preds)
        train_accuracy.append(trainAccuracy)
        print(f"Training, in epoch {epoch}, Loss: {l.item()}, Accuracy: {trainAccuracy}")

        # Validation
        resnet.eval()
        with torch.no_grad():
            all_val_labels = []
            all_val_preds = []
            for inputs, labels in test_data:
                inputs = inputs.to(glb.device)
                labels = labels.to(glb.device)
                outputs = resnet(inputs)
                test_loss = loss(outputs, labels)
                val_losses.append(test_loss.item())
                _, predicted = torch.max(outputs, 1)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(predicted.cpu().numpy())
            valAccuracy = accuracy_score(all_val_labels, all_val_preds)
            val_accuracy.append(valAccuracy)
            print(f"Validation, in epoch {epoch}, Loss: {test_loss.item()}, Accuracy: {valAccuracy}")

    plt.figure()
    plt.title("Training and Validation Loss")
    plt.plot(train_losses, label = 'Training')
    plt.plot(val_losses, label = 'Validation')
    plt.ylabel('Loss')
    plt.xlabel('Step')
    plt.legend(loc='best')
    plt.show()

    plt.figure()
    plt.title("Training and Validation Accuracy")
    plt.plot(train_accuracy, label = 'Training')
    plt.plot(val_accuracy, label = 'Validation')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(loc='best')
    plt.show()

    with open("Model/Saved/hyperparameter_log", 'a') as file:
        hyperparam_str = f"seed={seed}, hyperparameter:{{learning rate: {hyperparam['learning rate']}, number of epoch: {hyperparam['epoch number']}, batch size: {hyperparam['batch size']} }}"
        file.write(hyperparam_str + '\n')

    torch.save(resnet.state_dict(), f'Model/Saved/resnet18_Seed={seed}.pth')

    resnet.eval()
