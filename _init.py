import torch


class Config:
    def __init__(self):
        self.device_id = torch.cuda.current_device()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    x = torch.tensor([1, 2, 3], device=torch.device('cuda'))
    print(x)

