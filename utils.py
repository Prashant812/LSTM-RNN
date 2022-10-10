from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn


def plotLosses(trainLosses, testLosses):
    xs = np.arange(0, len(trainLosses))
    plt.plot(xs, trainLosses, label="Train Loss")
    xs = np.arange(0, len(testLosses))
    plt.plot(xs, testLosses, alpha=0.65, label="Test Loss")
    plt.legend()
    plt.title("Losses vs Epoch")
    plt.show()


def getActualAndPredictedOutput(model, device, dataloader) -> Tuple[List, List]:
    pred = []
    act = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            yhat = model(x)
            _, ypred = yhat.max(dim=1)
            ypred = ypred.detach().cpu().numpy()
            ypred = list(ypred)
            y = list(y.detach().cpu().numpy())
            pred.extend(ypred)
            act.extend(y)
    return pred, act


def trainModel(device, model, criterion, optimizer, threshold, trainDataloader, trainDataset, testDataloader, testDataset, verbose=True,):
    rnn = model.to(device)
    epoch = 0
    trainLoss = 0
    prevTrainLoss = 0
    trainLosses = []
    testLosses = []
    while True:
        epoch += 1
        trainLoss = 0
        rnn.train()
        if verbose:
            print()
            print(f"Epoch #{epoch} {'-'*30}")
        for x, y in trainDataloader:
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            pred = rnn(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            if type(x) == torch.nn.utils.rnn.PackedSequence:
                nInput = x.batch_sizes.max().item()
            else:
                nInput = x.shape[0]
            trainLoss += nInput*loss.item()
        trainLoss = trainLoss/len(trainDataset)
        trainLosses.append(trainLoss)
        if verbose:
            print(f"Train Loss: {trainLoss}")
            print(f"Previous Training Loss: {prevTrainLoss}")
        rnn.eval()
        with torch.no_grad():
            testLoss = 0
            for x, y in testDataloader:
                x = x.to(device)
                y = y.to(device)
                pred = rnn(x)
                loss = criterion(pred, y)
                if type(x) == torch.nn.utils.rnn.PackedSequence:
                    nInput = x.batch_sizes.max().item()
                else:
                    nInput = x.shape[0]
                testLoss += nInput*loss.item()
            testLoss = testLoss/len(testDataset)
            testLosses.append(testLoss)
        if verbose:
            print(f"Test Loss: {testLoss}")

        if ((abs(trainLoss-prevTrainLoss) < threshold) and prevTrainLoss != 0) or epoch > 1000:
            break
        else:
            prevTrainLoss = trainLoss
    if verbose:
        print(
            f"Training continued till {epoch} number of epochs.\n The final training loss being {trainLoss} and test loss being {testLoss}.")
    return rnn, trainLosses, testLosses


def collate_fn(listOfData):
    x = list(map(lambda x: x[0], listOfData))
    y = list(map(lambda x: x[1], listOfData))
    xxs = nn.utils.rnn.pack_sequence(x, enforce_sorted=False)
    yys = torch.stack(y)
    return xxs, yys
