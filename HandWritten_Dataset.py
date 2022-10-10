import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from pathlib import Path
from typing import List
import os
import numpy as np
from torch import nn


class HandWritingDataset(Dataset):
    def __init__(self, path: str, group: int, train=True, nGap=1, nPlots=2):
        self.nGap = nGap
        self.nPlots = nPlots
        if train:
            subpath = "train"
        else:
            subpath = "dev"
        self.dir = Path(path)
        self.group = group
        with open(self.dir/"Mapping.txt") as fhand:
            for i in fhand:
                if i.startswith(f"Group {group}"):
                    line = i
                    break
        chars = self.getChars(line)
        self.numChars = len(chars)
        self.char2int = {chars[i]: i for i in range(len(chars))}
        self.int2char = {i: chars[i] for i in range(len(chars))}
        HandData = []
        plotData = []
        for char in chars:
            ploti = 0
            folder = self.dir/char/subpath
            files = os.listdir(folder)
            for file in files:
                with open(folder/file) as fhand:
                    data = fhand.read()
                data = data.split()
                data = data[1:]
                data = list(map(float, data))
                data = np.array(data)
                data = torch.from_numpy(data)
                data = data.reshape((-1, 2))
                data = data/data.max(dim=0)[0]
                data = data[::nGap, :]
                data = data.float()
                label = torch.tensor(self.char2int[char])
                if ploti < nPlots:
                    plotData.append((data.detach().numpy(), char))
                ploti += 1
                HandData.append((data, label))
        self.plotData = plotData
        self.data = HandData
        self.chars = chars

    def showChars(self):
        for i, data in enumerate(self.plotData):
            x = data[0][:, 0]
            y = data[0][:, 1]
            label = data[1]
            plt.subplot(self.nPlots, self.numChars, i+1)
            plt.plot(x, y, c="r")
            plt.xticks([])
            plt.yticks([])
            plt.title(f"{label}", c="g")

        plt.show()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def getChars(self, line: str):
        colonIndex = line.find(":")
        line = line[colonIndex+1:]
        chars: List = line.split(",")
        chars = list(map(lambda x: x.strip(), chars))
        return chars
