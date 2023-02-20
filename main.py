import random
import torch

import matplotlib.pyplot as plt
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
    epochs = 100
    epoch_losses = list()
    median = init_median(points)
    # dimension change to facilitate matrix computation
    median = torch.tensor([median], requires_grad=True)
    points = torch.tensor(points)

    for i in range(epochs):
        loss = total_distance(points, median)
        print("median", median)
        print("loss", loss)
        loss.backward()
        epoch_losses.append(loss.item())
        with torch.no_grad():
            print("grad", median.grad)
            median -= median.grad
            median.grad.zero_()
        print(median)

    plt.plot(epoch_losses)
    plt.savefig('losses.png')


def check_dist(points, median):
    dists = [euclidean(p, median) for p in points]
    dist = sum(dists)
    print(dist)


if __name__ == "__main__":
    points = [(1, 0), (2, 0), (5, 0), (9, 0), (11, 0)]
    median = geometric_median(points)
    print(median)
