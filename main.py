import random
import torch

# from math import dist
from scipy.spatial.distance import euclidean
from torchmetrics.functional import pairwise_euclidean_distance


def total_distance(points, median):
    # return torch.sum(torch.tensor([euclidean(p, median) for p in points]))
    # print(median)
    # print(pairwise_euclidean_distance(points, median))
    # return torch.sum(pairwise_euclidean_distance(points, median))
    dist = points - median
    # print('diff', dist)
    dist = dist.pow(2)
    # print("pow", dist)
    dist = dist.sum(1).sqrt()
    # print(dist)
    return torch.sum(dist)


def random_range(min, max):
    return random.random() * (max - min) + min


def init_median(points):
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

    median = init_median(points)
    median = torch.tensor([median], requires_grad=True)
    # median = median.reshape(1, -1)
    points = torch.tensor(points)
    # print(points)
    # print(median)
    # print(median.view(1, -1))
    # loss = torch.nn.MSELoss()

    # TODO do it without duplicating medians
    for i in range(epochs):
        Y = [median for _ in range(len(points))]
        # loss = total_distance(points, median)
        loss = total_distance(points, median)
        print("median", median)
        print("loss", loss)
        # output = loss(points, Y)
        loss.backward()
        with torch.no_grad():
            print("grad", median.grad)
            median -= median.grad * 1e-1
            median.grad.zero_()
        print(median)


def check_dist(points, median):
    dists = [euclidean(p, median) for p in points]
    dist = sum(dists)
    print(dist)


if __name__ == "__main__":
    points = [(1, 0), (2, 0), (5, 0), (9, 0), (11, 0)]
    median = geometric_median(points)
    import pdb

    pdb.set_trace()
    print(median)
