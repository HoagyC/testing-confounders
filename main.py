from dataclasses import dataclass
import math
import random
import time

from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


@dataclass
class Point:
    point_id: int
    x: int
    y: int
    parents: list
    children: list
    error: float
    value: float = 0


@dataclass
class Link:
    point: Point
    strength: float


def build_simple_regression(n_inputs):
    # add n_nodes points below the output node
    points = []
    points.append(Point(
        x=50,
        y=50,
        parents=[],
        children=[],
        point_id=0,
        error=random.normalvariate(0, 1)
    ))

    for i in range(n_inputs):
        conx_str = random.normalvariate(0, 1)
        points.append(Point(x=50 + (i - (n_inputs / 2)) * 10,
                            y=20,
                            parents=[],
                            children=[Link(points[0], conx_str)],
                            point_id=i+1,
                            error=random.normalvariate(0, 1)))

        points[0].parents.append(Link(points[i+1], conx_str))

    return points


def build_random_diagonal(n_nodes, scale, density):
    # add n_nodes points randomly
    points = []
    for i in range(n_nodes):
        points.append(Point(x=random.randrange(scale),
                            y=random.randrange(scale),
                            parents=[],
                            children=[],
                            point_id=i,
                            error=random.uniform(0, 1)))

    # sort points from bottom-left to top-right
    points.sort(key=lambda p: math.hypot(p.x, p.y))

    # connect points by density, with direction from bottom-left to top-right
    for n1, p1 in enumerate(points):
        for n2, p2 in enumerate(points[:n1]):
            distance = math.hypot(p1.x - p2.x, p1.y - p2.y)
            if distance < scale * density:
                conx_str = random.uniform(-1, 1)
                p1.parents.append(Link(p2, conx_str))
                p2.children.append(Link(p1, conx_str))

    return points


def make_directed_graph(points_list):
    # add a point on the graph for each point
    plt.scatter([p.x for p in points_list], [p.y for p in points_list], s=2)

    # add arrow from parent to child to show the directed graph
    for p1 in points_list:
        for p2 in [l.point for l in p1.children]:
            plt.arrow(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y, head_width=3, length_includes_head=True)

    plt.show()


def forward_pass(points_list):
    # takes each point and assigns it a value based on the value of its predecessors, the connection strengths,
    # and the size of the error.
    # requires that the points form a directed acyclic graph ordered such that a parent precedes all of its children

    for p in points_list:
        p.value = sum([link.strength * link.point.value for link in p.parents]) + np.random.normal(p.error)
    return [p.value for p in points_list]


def test_subset(outcome_data, selected_var, n_confounds=5):
    # outcome_data has shape (n_trials, n_nodes)
    dependent_var = outcome_data[:, -1]
    independent_var = outcome_data[:, selected_var]

    # puts the selected variable first as a column vector, and then adds the chosen number of confounders
    if n_confounds > outcome_data.shape[1] - 2:
        confounds = np.concatenate((outcome_data[:, :selected_var], outcome_data[:, selected_var + 1:-1]),  axis=1)
    elif n_confounds == 0:
        confounds = None
    else:
        # sample confounders randomly from n_nodes except the final node and selected var
        var_range = list(range(outcome_data.shape[1] - 1))
        var_range.remove(selected_var)
        confounds_xs = random.sample(var_range, k=n_confounds)
        confounds = outcome_data[:, confounds_xs]

    if n_confounds:
        X = np.insert(confounds, 0, independent_var, axis=1)
    else:
        # X must be a 2D array to feed into the LinearRegression
        X = independent_var.reshape((len(independent_var), -1))

    regr = LinearRegression().fit(X, dependent_var)
    return regr.coef_[0]


def run_regressions(graph, n_trials, selected_variable, n_samples=40):
    n_nodes = len(graph)
    outcomes = np.array([forward_pass(graph) for _ in range(n_trials)])

    impacts = []
    for n_confounds in range(n_nodes - 1):
        n_confounds_results = []
        for _ in range(n_samples):
            coefficient = test_subset(outcomes, selected_variable, n_confounds=n_confounds)
            n_confounds_results.append(coefficient)
        impacts.append((n_confounds, n_confounds_results))

    full_impact = test_subset(outcomes, selected_variable, n_confounds=n_nodes)

    return impacts, full_impact


def plot_results_by_n_confounds(results, full_impact):
    for r in results:
        plt.plot(r[1], len(r[1]) * [r[0]])
        plt.plot(np.mean(r[1]), r[0], markersize=3, marker='x', color='red')

    plt.axvline(full_impact, color='green')
    plt.show()


def get_abs_errors(results):
    errors = []
    for r in results:
        errors.append(np.mean(r[1]))
    return errors


def main():
    n_nodes = 40
    scale = 100
    density = 0.25
    n_trials = 200
    n_samples = 40

    selected_variable = n_nodes - 2

    n_graphs = 50

    abs_errors = []
    for _ in range(n_graphs):
        graph = build_random_diagonal(n_nodes, scale, density)

        impacts, full_impact = run_regressions(graph, n_trials, selected_variable, n_samples=n_samples)

        abs_errors.append([abs(np.mean(r[1] - full_impact)) for r in impacts])

    abs_errors = np.array(abs_errors)
    mean_average_errors = np.mean(abs_errors, axis=0)
    print(mean_average_errors)
    plt.plot(np.linspace(0, 1, num=len(mean_average_errors)), mean_average_errors)
    plt.show()


if __name__ == "__main__":
    main()
