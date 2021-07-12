from copy import deepcopy
from dataclasses import dataclass
import math
import random
import time
from typing import List

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


@dataclass
class Graph:
    points: List[Point]


def build_simple_regression(n_inputs: int):
    # adds the output node
    points = [Point(
        x=50,
        y=50,
        parents=[],
        children=[],
        point_id=n_inputs - 1,
        error=random.uniform(0, 1)
    )]

    # adds the input nodes at the beginning of the points list
    for i in range(n_inputs - 1):
        conx_str = random.normalvariate(0, 1)
        points.insert(-1, Point(x=50 + ((i + 1) - (n_inputs / 2)) * 10,
                                y=20,
                                parents=[],
                                children=[Link(points[-1], conx_str)],
                                point_id=i,
                                error=random.uniform(0, 1)))
        points[-1].parents.append(Link(points[i], conx_str))

    return Graph(points)


def build_random_diagonal(n_nodes: int, density: float):
    scale = 100  # just for graphing, has no material impact

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
        p1.point_id = n1
        for n2, p2 in enumerate(points[:n1]):
            distance = math.hypot(p1.x - p2.x, p1.y - p2.y)
            if distance < scale * density:
                conx_str = random.uniform(-1, 1)
                p1.parents.append(Link(p2, conx_str))
                p2.children.append(Link(p1, conx_str))

    return Graph(points)


def plot_directed_graph_paths(graph: Graph, source: int = 5, labels=[]):
    # add a point on the graph for each point
    plt.scatter([p.x for p in graph.points], [p.y for p in graph.points], s=2)

    if labels:
        assert len(graph.points) == len(labels)
        for p in graph.points:
            plt.text(p.x, p.y + 10, round(labels[p.point_id], 2))

    assert source < len(graph.points)
    is_reachable = reachable(source, graph)

    # add arrow from parent to child to show the directed graph
    for p1 in graph.points:
        if source == p1.point_id:
            colour = 'green'
        elif is_reachable[p1.point_id]:
            colour = 'red'
        else:
            colour = 'blue'

        for p2 in [l.point for l in p1.children]:
            plt.arrow(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y, head_width=3, length_includes_head=True, color=colour)

    plt.show()


def plot_directed_graph(graph: Graph):

    # add a point on the graph for each point
    plt.scatter([p.x for p in graph.points[:-1]], [p.y for p in graph.points[:-1]], s=2)
    plt.scatter(graph.points[-1].x, graph.points[-1].y, s=50, marker="*")

    # add arrow from parent to child to show the directed graph
    for p1 in graph.points:
        for l in p1.children:
            color = 'red' if l.strength < 0 else 'black'
            gap_x = l.point.x - p1.x
            gap_y = l.point.y - p1.y
            plt.arrow(x=p1.x + 0.1 * gap_x,
                      y=p1.y + 0.1 * gap_y,
                      dx=(l.point.x - p1.x) * 0.8,
                      dy=(l.point.y - p1.y) * 0.8,
                      head_width=2,
                      length_includes_head=True,
                      width=l.strength * 0.7,
                      color=color)

    plt.show()


def get_causal_impact(graph: Graph, selected_var: int, final_var: int):
    selected_point = graph.points[selected_var]
    assert selected_point.point_id == selected_var
    downstream = reachable(selected_var, graph)
    upstream = reachable(final_var, graph, reverse=True)
    midpoints = [p1 and p2 for p1, p2 in zip(downstream, upstream)]
    midpoints[final_var] = 1
    # print(midpoints)
    traversed = [0] * len(graph.points)
    traversed[selected_var] = 1
    impacts = [0] * len(graph.points)
    impacts[selected_var] = 1
    for n, p in enumerate(graph.points):
        if not midpoints[n]:
            continue
        # this checks that all of the parents have been traversed
        # if they have then their impact values are set, and so the impact value of the child can be calculated
        impacts[n] = sum([x.strength * impacts[x.point.point_id] for x in p.parents])

    # plot_directed_graph_paths(graph, source=selected_var, labels=impacts)
    return impacts


def reachable(p1: int, graph: Graph, reverse: bool = False):
    visited = [0]*len(graph.points)
    queue = [p1]
    while queue:
        n = queue.pop(0)
        if not reverse:
            for child in graph.points[n].children:
                child_id = child.point.point_id
                if not visited[child_id]:
                    queue.append(child_id)
                    visited[child_id] = 1
        else:
            for parent in graph.points[n].parents:
                parent_id = parent.point.point_id
                if not visited[parent_id]:
                    queue.append(parent_id)
                    visited[parent_id] = 1

    return visited


def forward_pass(graph: Graph, selected_var: int = 0):
    # takes each point and assigns it a value based on the value of its predecessors, the connection strengths,
    # and the size of the error.
    # requires that the points form a directed acyclic graph ordered such that a parent precedes all of its children

    selected_point = graph.points[selected_var]
    assert selected_point.point_id < len(graph.points)

    for p in graph.points:
        p.value = 0

    for p in graph.points:
        p.value = sum([link.strength * link.point.value for link in p.parents]) + np.random.normal(0, p.error)

        if p.value > 1000:
            print([(link.strength, link.point.value) for link in p.parents], p.error)
    return [p.value for p in graph.points]


def forward_causal_impact(graph: Graph, selected_var: int = 0, seed: int = 0, final_var: int = 0):
    # takes each point and assigns it a value based on the value of its predecessors, the connection strengths,
    # and the size of the error.
    # requires that the points form a directed acyclic graph ordered such that a parent precedes all of its children
    if not final_var:
        final_var = len(graph.points) - 1

    deltas = list(np.arange(-1, 1, 0.1))
    finals = []

    for delta in deltas:
        np.random.seed(seed)
        selected_point = graph.points[selected_var]
        assert selected_point.point_id < len(graph.points)

        for p in graph.points:
            p.value = 0

        for p in graph.points:
            p.value = sum([link.strength * link.point.value for link in p.parents]) + np.random.normal(0, p.error)
            if p.point_id == selected_var:
                p.value += delta

            if p.value > 1000:
                print([(link.strength, link.point.value) for link in p.parents], p.error)

        finals.append(graph.points[final_var].value)

    deltas_input = np.array(deltas)
    deltas_input = deltas_input.reshape(-1, 1)
    finals_input = np.array(finals)
    regr = LinearRegression().fit(deltas_input, finals_input)
    impact = regr.coef_[0]    # regression unnecessary for linear models but important for sensitive, non-linear cases
    return deltas, finals, impact


def test_subset(outcome_data, selected_var, n_confounds=5):
    # outcome_data has shape (n_trials, n_nodes)
    dependent_var = outcome_data[:, -1]
    independent_var = outcome_data[:, selected_var]

    # puts the selected variable first as a column vector, and then adds the chosen number of confounders
    if n_confounds > outcome_data.shape[1] - 2:
        confounds = np.concatenate((outcome_data[:, :selected_var], outcome_data[:, selected_var + 1:-1]), axis=1)
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


def run_regressions(graph: Graph, n_trials: int, selected_variable: int, n_samples: int = 40):
    n_nodes = len(graph.points)
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


def run_trials(n_nodes=40, density=0.25, n_trials=200, n_samples=40):
    selected_variable = n_nodes - 2

    n_graphs = 50

    abs_errors = []

    for i in range(n_graphs):
        graph = build_random_diagonal(n_nodes, density)
        if i == 0:
            final_var = len(graph.points) - 2
            impacts = get_causal_impact(graph, 5, final_var)
            plot_directed_graph_paths(graph, labels=impacts)

        impacts, full_impact = run_regressions(graph, n_trials, selected_variable, n_samples=n_samples)

        abs_errors.append([abs(np.mean(r[1] - full_impact)) for r in impacts])
        # print(abs_errors, impacts, full_impact)
        # print(i)

    abs_errors = np.array(abs_errors)
    mean_average_errors = np.mean(abs_errors, axis=0)
    print(mean_average_errors)
    plt.plot(np.linspace(0, 1, num=len(mean_average_errors)), mean_average_errors)
    plt.show()


def main():
    n_nodes = 40
    n_trials = 200
    density = 0.25

    graph = build_random_diagonal(n_nodes, density=0.25)
    plot_directed_graph(graph)
    print([x.point_id for x in graph.points])
    # plot_directed_graph(graph)
    final_var = len(graph.points) - 1
    impacts = [get_causal_impact(graph, i, final_var)[-1] for i in range(n_nodes)]

    # d, f, i = forward_causal_impact(graph, 5, 0, final_var)
    # # plt.scatter(d, f)

    outcomes = np.array([forward_pass(graph) for _ in range(n_trials)])
    full_controlled_impacts = [test_subset(outcomes, i, n_confounds=n_nodes) for i in range(n_nodes)]
    tested_impacts = [forward_causal_impact(graph, selected_var=i, final_var=final_var)[-1] for i in range(n_nodes)]

    # print(impacts, full_controlled_impacts)

    plt.scatter(impacts, full_controlled_impacts, c=tested_impacts)
    plt.show()


if __name__ == "__main__":
    main()
