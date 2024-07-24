from collections import deque
from typing import List, Tuple, Optional

INF = float('inf')

def spfa(s: int, adj: List[List[Tuple[int, int]]]) -> Tuple[bool, List[int], Optional[List[int]]]:
    n = len(adj)
    dist = [INF] * n
    cnt = [0] * n
    inqueue = [False] * n
    p = [-1] * n  # predecessor list to reconstruct paths
    q = deque()

    dist[s] = 0
    q.append(s)
    inqueue[s] = True

    while q:
        v = q.popleft()
        inqueue[v] = False

        for to, length in adj[v]:
            if dist[v] + length < dist[to]:
                dist[to] = dist[v] + length
                p[to] = v
                if not inqueue[to]:
                    q.append(to)
                    inqueue[to] = True
                    cnt[to] += 1
                    if cnt[to] > n: # Negative cycle
                        return False, dist, None

    return True, dist, p

def reconstructPath(s: int, e: int, p: list[int]):
    path = []
    at = e
    while at:
        path.append(at)
        at = p[at]
    path.reverse()
    # If s and e are connected return the path
    if path[0] == s:
        return path
    return []

def test_spfa():
    adj = [
        [(1, 2), (2, 5)],      # edges from vertex 0
        [(2, 1), (3, 4)],      # edges from vertex 1
        [(3, 3), (4, 8)],      # edges from vertex 2
        [(4, 2), (5, 2)],      # edges from vertex 3
        [(5, 10)],            # edge creating a negative cycle
        [(3, 1)],              # edge creating a negative cycle
    ]
    source = 0
    has_no_negative_cycle, distances, p = spfa(source, adj)
    print("No negative cycle detected:", has_no_negative_cycle)
    if has_no_negative_cycle:
        for target in range(len(adj)):
            path = reconstruct_path(source, target, p)
            print(f"Path from {source} to {target}: {path}")

if __name__ == "__main__":
    test_spfa()
