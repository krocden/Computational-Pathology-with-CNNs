import os
import torch
import torchvision.models as model
import ImgLoader
import _init

if __name__ == '__main__':
    glb = _init.Config()
    resnet = model.resnet18(weights=None)
    fc_in = resnet.fc.in_features
    resnet.fc = torch.nn.Linear(fc_in, out_features=3)

    resnet = resnet.to(glb.device)

    image_path = "datasets/Dataset 1/Dataset 1/Colorectal Cancer"

    current_script_path = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_script_path)
    datasets_path = os.path.join(parent_dir, image_path)

    loader = ImgLoader.Loader(datasets_path, shuffle=True)

    data = loader.get_dataloader()

    optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001)
    loss = torch.nn.CrossEntropyLoss()

    num_epochs = 10
    resnet.train()
    for epoch in range(num_epochs):
        for inputs, labels in data:
            inputs = inputs.to(glb.device)
            labels = labels.to(glb.device)
            optimizer.zero_grad()
            outputs = resnet(inputs)
            l = loss(outputs, labels)
            l.backward()
            optimizer.step()
        print(f"in epoch {epoch}, the loss is: {l.item()}")

    torch.save(resnet.state_dict(), 'resnet18.pth')

    resnet.eval()
