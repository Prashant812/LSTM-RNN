from torch import nn
import torch


class LSTMModel(nn.Module):
    def __init__(self, stateDimension: int, featureDim: int, nOutputFeatures: int, numLayers=2):
        super().__init__()
        self.nFeature = featureDim
        self.nState = stateDimension
        self.nLayers = numLayers
        self.nOutput = nOutputFeatures
        self.rnn = nn.LSTM(self.nFeature, self.nState,
                           numLayers, batch_first=True)
        self.register_buffer("h0", torch.randn(numLayers, 1, self.nState))
        self.register_buffer("c0", torch.randn(numLayers, 1, self.nState))
        self.fcnn = nn.Linear(self.nState, self.nOutput)

    def forward(self, x: torch.Tensor):
        if type(x) == torch.nn.utils.rnn.PackedSequence:
            bs = x.batch_sizes.max().item()
        else:
            bs = x.shape[0]
        h0 = self.h0.repeat(1, bs, 1)
        c0 = self.c0.repeat(1, bs, 1)
        _, (hn, _) = self.rnn(x, (h0, c0))
        hn = hn[-1]
        output = self.fcnn(hn)
        return output
