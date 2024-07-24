from typing import List, Tuple
import sys

INF = float('inf')

class Edge:
    def __init__(self, weight: int = INF, to: int = -1):
        self.weight = weight
        self.to = to

def prim(n: int, adj: List[List[int]]) -> None:
    total_weight = 0
    selected = [False] * n
    min_e = [Edge() for _ in range(n)]
    min_e[0].weight = 0

    for _ in range(n):
        v = -1
        for j in range(n):
            if not selected[j] and (v == -1 or min_e[j].weight < min_e[v].weight):
                v = j

        if min_e[v].weight == INF:
            print("No MST!")
            return

        selected[v] = True
        total_weight += min_e[v].weight
        if min_e[v].to != -1:
            print(f"{v} {min_e[v].to}")

        for to in range(n):
            if adj[v][to] < min_e[to].weight:
                min_e[to] = Edge(adj[v][to], v)

    print(total_weight)

def test_prim():
    n = 4
    adj = [
        [0, 1, 4, INF],
        [1, 0, 2, 6],
        [4, 2, 0, 3],
        [INF, 6, 3, 0]
    ]
    prim(n, adj)

if __name__ == "__main__":
    test_prim()
