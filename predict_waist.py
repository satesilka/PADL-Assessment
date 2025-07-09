import torch
import torch.nn as nn
import numpy as np


class PredictWaist(nn.Module):
    def __init__(self, input_size=5):
        super(PredictWaist, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.output_layer = nn.Linear(16, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output_layer(x)
        return x


def predict(measurements):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PredictWaist(input_size=5).to(device)
    model.load_state_dict(torch.load('q4_model.pth', map_location=device))
    model.eval()

    feature_means = torch.tensor([982.398174, 1023.80899, 1721.67697, 74.3073034, 0.481039326], device=device)
    feature_stds = torch.tensor([109.33545829, 84.94810067, 106.07447632, 16.19013007, 0.49964036], device=device)

    measurements = measurements.to(device)
    measurements = (measurements - feature_means) / feature_stds

    with torch.no_grad():
        scaled_predictions = model(measurements)
        target_mean = 854.9901685393259
        target_std = 120.18641262533329
        predictions = scaled_predictions * target_std + target_mean

    return predictions.cpu()
