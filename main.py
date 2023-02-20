import os
import random

import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from scipy.spatial.distance import euclidean


def total_distance(points, median):
    """
    Sum of euclidean distances between each point and the median point
    """
    return torch.sum((points - median).pow(2).sum(1).sqrt())


def random_range(min, max):
    """
    Generate random float number within range
    """
    return random.random() * (max - min) + min


def init_median(points):
    """
    Initialize median point from given points
    """
    # TODO try init with median in each dimension
    assert len(points) > 0
    dimension = len(points[0])
    median = [
        random_range(min(p[d] for p in points), max(p[d] for p in points))
        for d in range(dimension)
    ]
    return median


def geometric_median(points):
    epochs = 10
    epoch_losses = list()

    median = init_median(points)
    medians = [median]
    # dimension change to facilitate matrix computation
    median = torch.tensor([median], requires_grad=True)
    points = torch.tensor(points)

    for i in tqdm(range(epochs)):
        loss = total_distance(points, median)
        loss.backward()
        epoch_losses.append(loss.item())

        with torch.no_grad():
            median -= median.grad * 1e-1
            median.grad.zero_()

        medians.append(median.detach().numpy()[0])

    # save plots
    plt.plot(epoch_losses)
    plt.savefig("images/losses.png")
    plot_animation_2d(points.numpy(), medians)

    return median.detach().numpy()[0]


def plot_animation_2d(points, medians):
    """
    Plot 2D plot animation
    """
    plot_files = list()
    colors = ["blue"] * len(points) + ["red"]
    X = [p[0] for p in points]
    Y = [p[1] for p in points]

    for i, median in enumerate(medians):
        plt.figure()
        plt.scatter(X + [median[0]], Y + [median[1]], c=colors)

        fname = f"images/points_{i}.png"
        plt.savefig(fname)
        plot_files.append(fname)

    with imageio.get_writer("images/optimization.gif", mode="I") as writer:
        for fname in plot_files:
            image = imageio.imread(fname)
            writer.append_data(image)

    for fname in set(plot_files):
        os.remove(fname)


def check_dist(points, median):
    dists = [euclidean(p, median) for p in points]
    dist = sum(dists)
    print(dist)


if __name__ == "__main__":
    points = [(1, 0), (2, 0), (5, 0), (9, 0), (11, 0)]
    median = geometric_median(points)
    print(median)
