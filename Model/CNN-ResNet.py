import os
import torch
import torchvision.models as model
import ImgLoader
import _init

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

    image_path = "datasets/Dataset 1/Dataset 1/Colorectal Cancer"

    current_script_path = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_script_path)
    datasets_path = os.path.join(parent_dir, image_path)

    loader = ImgLoader.Loader(glb, datasets_path, shuffle=True, batch_size=hyperparam['batch size'])

    train_data, _ = loader.get_dataloader(train_ratio=0.7)

    optimizer = torch.optim.Adam(resnet.parameters(), lr=hyperparam['learning rate'])
    loss = torch.nn.CrossEntropyLoss()

    num_epochs = hyperparam['epoch number']
    resnet.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_data:
            inputs = inputs.to(glb.device)
            labels = labels.to(glb.device)
            optimizer.zero_grad()
            outputs = resnet(inputs)
            l = loss(outputs, labels)
            l.backward()
            optimizer.step()
        print(f"in epoch {epoch}, the loss is: {l.item()}")

    with open("Saved/hyperparameter_log", 'a') as file:
        hyperparam_str = f"seed={seed}, hyperparameter:{{learning rate: {hyperparam['learning rate']}, number of epoch: {hyperparam['epoch number']}, batch size: {hyperparam['batch size']} }}"
        file.write(hyperparam_str + '\n')

    torch.save(resnet.state_dict(), f'Saved/resnet18_Seed={seed}.pth')

    resnet.eval()
