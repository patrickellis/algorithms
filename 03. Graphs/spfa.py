from collections import deque
from typing import List, Tuple, Optional

adj = [
    [(1, 2), (2, 5)],  # edges from vertex 0
    [(2, 1), (3, 4)],  # edges from vertex 1
    [(3, 3), (4, 8)],  # edges from vertex 2
    [(4, 2), (5, 2)],  # edges from vertex 3
    [(5, 10)],  # edge creating a negative cycle
    [(3, 1)],  # edge creating a negative cycle
]
n = len(adj)
dist = [float("inf")] * n
cnt = [0] * n
inqueue = [False] * n
p = [-1] * n  # predecessor list to reconstruct paths


def spfa(
    s: int, adj: List[List[Tuple[int, int]]]
) -> Tuple[bool, List[int], Optional[List[int]]]:
    dist[s] = 0
    queue = deque()
    queue.append(s)
    inqueue[s] = True

    while queue:
        v = queue.popleft()
        inqueue[v] = False

        for to, weight in adj[v]:
            if dist[v] + weight < dist[to]:
                dist[to] = dist[v] + weight
                p[to] = v
                if not inqueue[to]:
                    queue.append(to)
                    inqueue[to] = True
                    cnt[to] += 1
                    if cnt[to] > n - 1:
                        return False, dist, None

    return True, dist, p


def reconstructPath(s: int, e: int, p: list[int]):
    path = []
    at = e
    while at != s:
        path.append(at)
        at = p[at]
    path.append(s)
    path.reverse()
    return path


def test_spfa():
    source = 0
    has_no_negative_cycle, distances, p = spfa(source, adj)
    print("No negative cycle detected:", has_no_negative_cycle)
    if has_no_negative_cycle:
        for target in range(len(adj)):
            path = reconstructPath(source, target, p)
            print(f"Path from {source} to {target}: {path}")
    print(f"Predecessors: {p}")


if __name__ == "__main__":
    test_spfa()
