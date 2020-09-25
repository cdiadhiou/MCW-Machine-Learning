import numpy as np
import torch
import torch.nn as nn
import json
import joblib

from azureml.core.model import Model

# The number of nodes in the hidden layers
hidden_layer_size = 150


class LSTMPredictor(nn.Module):
    def __init__(self):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTMCell(1, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, 1)

    def forward(self, input, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), hidden_layer_size, dtype=torch.double)
        c_t = torch.zeros(input.size(0), hidden_layer_size, dtype=torch.double)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm(input_t, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
        for i in range(future):  # if we should predict the future
            h_t, c_t = self.lstm(output, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


def init():
    global model
    # model_path = Model.get_model_path('pytorch-hymenoptera')
    model_path = './model.pt'
    model = torch.load(model_path)

    model.eval()


def run(input_data):
    input_data = torch.from_numpy(np.array(json.loads(input_data)))
    future_days = 30

    # get prediction
    with torch.no_grad():
        y_t = model(input_data, future=future_days)
        y = y_t.detach().numpy()

    result = {"future_days": future_days, "forecast": y[0, -future_days:]}
    return result


if __name__ == '__main__':
    init()
    with open('data.json', 'r', encoding='utf-8') as f:
        data = f.read()

    print(run(data))
