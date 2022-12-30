import torch
class CNNRNN(torch.nn.Module):

    def __init__(self, class_num) -> None:
        super().__init__()
        self.lstm = torch.nn.LSTM(2048, 512, 1, batch_first = True)
        self.logits = torch.nn.ReLU()
        self.dense = torch.nn.Linear(512, class_num)
        self.softmax = torch.nn.Softmax(-1)
        self.class_num = class_num

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.logits(x)
        x = self.dense(x)
        x = self.softmax(x)
        return x