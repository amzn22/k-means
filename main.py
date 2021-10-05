import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import math


class Point:
    def __init__(self, x, y, cluster=-1):
        self.x = x
        self.y = y
        self.cluster = cluster

    def __str__(self):
        return self.x + self.y + self.cluster


def dist(a, b):
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def rand_points(n):
    points = []
    for i in range(n):
        point = Point(np.random.randint(0, 100), np.random.randint(0, 100))
        points.append(point)
    return points


def centroids(points, k):
    x_centre = np.mean(list(map(lambda p: p.x, points)))
    y_centre = np.mean(list(map(lambda p: p.y, points)))
    center = Point(x_centre, y_centre)
    R = max(map(lambda r: dist(r, center), points))
    centers = []
    for i in range(k):
        x_c = x_centre + R * np.cos(2 * np.pi * i / k)
        y_c = y_centre + R * np.sin(2 * np.pi * i / k)
        centers.append(Point(x_c, y_c, i))
    return centers


def nearest_centroid(points, centroids):
    for p in points:
        min_dist = dist(p, centroids[0])
        p.cluster = 0
        for i in range(len(centroids)):
            temp = dist(p, centroids[i])
            if temp < min_dist:
                min_dist = temp
                p.cluster = centroids[i].cluster


def show_clusters(points, centroids, title=None):
    clusters = set(list(map(lambda p: p.cluster, points)))
    colors = cm.rainbow(np.linspace(0, 1, len(clusters)))
    plt.scatter(
        list(map(lambda p: p.x, points)),
        list(map(lambda p: p.y, points)),
        color=list(map(lambda p: colors[p.cluster], points)),
    )
    plt.scatter(list(map(lambda p: p.x, centroids)), list(map(lambda p: p.y, centroids)), color='pink')
    plt.title(title)
    plt.show()
    plt.close()


def recalc_centroids(points, centers):
    new_centroids = []
    for center in centers:
        y_center = 0
        x_center = 0
        count = 0
        for point in points:
            if (point.cluster == center.cluster):
                y_center += point.y
                x_center += point.x
                count += 1
        # print(center.cluster, x_center, y_center, count)

        new_centroids.append(Point(x_center / count, y_center / count, center.cluster))
    return new_centroids


def centroids_changed(old_centroids, new_centroids):
    old_centroids.sort(key=lambda c: c.cluster)
    new_centroids.sort(key=lambda c: c.cluster)
    # print(old_centroids, new_centroids)
    if (len(old_centroids) != len(new_centroids)):
        return False
    for i in range(len(old_centroids)):
        if (old_centroids[i].x != new_centroids[i].x) or (old_centroids[i].y != new_centroids[i].y):
            return False
    return True


def j_count(points, centroids):
    j_value = 0
    for i in range(len(points)):
        for j in range(len(centroids)):
            if points[i].cluster == centroids[j].cluster:
                j_value += dist(points[i], centroids[j]) ** 2
    return j_value


def k_means_with_k(points, k):
    centers = centroids(points, k)
    nearest_centroid(points, centers)
    new_centroids = recalc_centroids(points, centers)
    while True:
        centers = new_centroids
        new_centroids = recalc_centroids(points, centers)
        nearest_centroid(points, new_centroids)
        show_clusters(points, centers)
        if centroids_changed(centers, new_centroids):
            break
    return centers


def d_count(prev, current, next):
    return abs(current - next) / abs(prev - current)


def cluster_count(points):
    d_m = []
    j_m = []
    k_max = math.sqrt(len(points))
    for i in range(1, int(k_max)):
        j_m.append(j_count(points, k_means_with_k(points, i)))
    for i in range(1, len(j_m) - 1):
        d_m.append(d_count(j_m[i - 1], j_m[i], j_m[i + 1]))
    print(d_m)
    index = d_m.index(min(d_m))
    return index + 1


def k_means(points):
    cluster_amount = cluster_count(points)
    return k_means_with_k(points, cluster_amount)