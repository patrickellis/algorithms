from typing import List, Tuple

INF = 10**9

class Edge:
    def __init__(self, src: int, dst: int, weight: int):
        self.src = src
        self.dst = dst
        self.weight = weight

def bellman_ford(n: int, edges: List[Edge], v: int) -> None:
    dist = [INF] * n
    dist[v] = 0
    p = [-1] * n
    x = -1

    for _ in range(n):
        x = -1
        for e in edges:
            if dist[e.src] < INF and dist[e.dst] > dist[e.src] + e.weight:
                dist[e.dst] = max(-INF, dist[e.src] + e.weight)
                p[e.dst] = e.src
                x = e.dst

    if x == -1:
        print(f"No negative cycle from {v}")
    else:
        y = x
        for _ in range(n):
            y = p[y]

        path = []
        cur = y
        while True:
            path.append(cur)
            if cur == y and len(path) > 1:
                break
            cur = p[cur]
        path.reverse()

        print("Negative cycle: ", end="")
        for u in path:
            print(u, end=" ")
        print()

def test_bellman_ford():
    edges = [
        Edge(0, 1, 1),
        Edge(1, 2, 1),
        Edge(2, 3, -3),
        Edge(3, 1, 1)
    ]
    n = 4
    source = 0
    bellman_ford(n, edges, source)

if __name__ == "__main__":
    test_bellman_ford()
