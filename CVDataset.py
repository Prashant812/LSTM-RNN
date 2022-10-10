import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from pathlib import Path
from typing import List
import os
import numpy as np


class CVDataset(Dataset):
    def __init__(self, path: str, group: int, train=True):
        if train:
            subpath = "Train"
        else:
            subpath = "Test"
        self.dir = Path(path)
        self.group = group
        with open(self.dir/"Mapping_CV.txt") as fhand:
            for i in fhand:
                if i.startswith(f"Group-{group}"):
                    line = i
                    break
        cvs = self.getCVs(line)
        self.numcvs = len(cvs)
        self.char2int = {cvs[i]: i for i in range(len(cvs))}
        self.int2char = {i: cvs[i] for i in range(len(cvs))}
        CVData = []
        for cv in cvs:
            folder = self.dir/cv/subpath
            files = os.listdir(folder)
            for file in files:
                with open(folder/file) as fhand:
                    data = fhand.readlines()
                data = list(map(lambda x: list(map(float, x.split())), data))
                data = np.array(data)
                data = torch.from_numpy(data)
                data = data.float()
                label = torch.tensor(self.char2int[cv])
                CVData.append((data, label))
        self.data = CVData
        self.chars = cvs

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def getCVs(self, line: str):
        gText = f"Group-{self.group}"
        groupIndex = line.find(gText)
        line = line[groupIndex+len(gText)+1:]
        cvs: List = line.split(",")
        cvs = list(map(lambda x: x.strip(), cvs))
        return cvs
