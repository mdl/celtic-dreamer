import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import yaml
import os
import attridict
import gymnasium as gym
import csv
import pandas as pd
import plotly.graph_objects as pgo


def seedEverything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def findFile(filename):
    currentDir = os.getcwd()
    for root, dirs, files in os.walk(currentDir):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"File '{filename}' not found in subdirectories of {currentDir}")


def loadConfig(config_path):
    if not config_path.endswith(".yml"):
        config_path += ".yml"
    config_path = findFile(config_path)
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return attridict(config)


def saveLossesToCSV(filename, metrics):
    fileAlreadyExists = os.path.isfile(filename + ".csv")
    with open(filename + ".csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        if not fileAlreadyExists:
            writer.writerow(metrics.keys())
        writer.writerow(metrics.values())


def plotMetrics(filename, title="", savePath="metricsPlot", window=10):
    if not filename.endswith(".csv"):
        filename += ".csv"

    data = pd.read_csv(filename)
    fig = pgo.Figure()

    colors = [
        "gold", "gray", "beige", "blueviolet", "cadetblue",
        "chartreuse", "coral", "cornflowerblue", "crimson", "darkorange",
        "deeppink", "dodgerblue", "forestgreen", "aquamarine", "lightseagreen",
        "lightskyblue", "mediumorchid", "mediumspringgreen", "orangered", "violet"]
    num_colors = len(colors)

    for idx, column in enumerate(data.columns):
        if column in ["envSteps", "gradientSteps"]:
            continue

        fig.add_trace(pgo.Scatter(
            x=data["gradientSteps"], y=data[column], mode='lines',
            name=f"{column} (original)",
            line=dict(color='gray', width=1, dash='dot'),
            opacity=0.5, visible='legendonly'))

        smoothed_data = data[column].rolling(window=window, min_periods=1).mean()
        fig.add_trace(pgo.Scatter(
            x=data["gradientSteps"], y=smoothed_data, mode='lines',
            name=f"{column} (smoothed)",
            line=dict(color=colors[idx % num_colors], width=2)))

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=30),
            yanchor='top'
        ),
        xaxis=dict(
            title="Gradient Steps",
            showgrid=True,
            zeroline=False,
            position=0
        ),
        yaxis_title="Value",
        template="plotly_dark",
        height=1080,
        width=1920,
        margin=dict(t=60, l=40, r=40, b=40),
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="White",
            borderwidth=2,
            font=dict(size=12)
        )
    )

    if not savePath.endswith(".html"):
        savePath += ".html"
    fig.write_html(savePath)


def sequentialModel1D(inputSize, hiddenSizes, outputSize, activationFunction="Tanh", finishWithActivation=False):
    activationFunction = getattr(nn, activationFunction)()
    layers = []
    currentInputSize = inputSize

    for hiddenSize in hiddenSizes:
        layers.append(nn.Linear(currentInputSize, hiddenSize))
        layers.append(activationFunction)
        currentInputSize = hiddenSize

    layers.append(nn.Linear(currentInputSize, outputSize))
    if finishWithActivation:
        layers.append(activationFunction)

    return nn.Sequential(*layers)


# def computeLambdaValues(rewards, values, continues, lambda_=0.95):
#     returns = torch.zeros_like(rewards)
#     bootstrap = values[:, -1]
#     for i in reversed(range(rewards.shape[-1])):
#         returns[:, i] = rewards[:, i] + continues[:, i] * ((1 - lambda_) * values[:, i] + lambda_ * bootstrap)
#         bootstrap = returns[:, i]
#     return returns

def compute_gae(rewards, values, gamma, lam):
    """
    Computes GAE advantages and returns.
    Args:
        rewards (Tensor): Shape [batch, T]
        values (Tensor): Shape [batch, T+1] (V(s_0) to V(s_T))
        gamma (float): Scalar discount factor.
        lam (float): Scalar GAE lambda parameter.
    Returns:
        advantages (Tensor): Shape [batch, T]
        returns (Tensor): Shape [batch, T]
    """
    T = rewards.size(1)
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(rewards.size(0), device=rewards.device)

    for t in reversed(range(T)):
        # The value of the next state is V(s_{t+1})
        next_value = values[:, t + 1]

        # Calculate the TD-error (delta)
        delta = rewards[:, t] + gamma * next_value - values[:, t]

        # Update the advantage using the recursive GAE formula
        last_adv = delta + gamma * lam * last_adv
        advantages[:, t] = last_adv

    # The returns are the advantages plus the original values
    returns = advantages + values[:, :T]

    return advantages, returns

def ensureParentFolders(*paths):
    for path in paths:
        parentFolder = os.path.dirname(path)
        if parentFolder and not os.path.exists(parentFolder):
            os.makedirs(parentFolder, exist_ok=True)