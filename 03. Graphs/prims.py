from typing import List, Tuple
import heapq
import sys

def prim(adj: List[List[Tuple[int,int]]]) -> None:
    n = len(adj)
    p = [-1]*n
    pq = [(0,0)]
    total_weight = 0
    visited = [False]*n

    for _ in range(n):
        weight, v = heapq.heappop(pq)
        if visited[v]:
            continue
        visited[v] = True
        total_weight += weight

        for to,weight in adj[v]:
            if not visited[to]:
                p[to] = v
                heapq.heappush(pq,(weight,to))

    return total_weight

def test_prim():
    adj = [
        [(1, 2), (3, 6)],
        [(0, 2), (2, 3), (3, 8), (4, 5)],
        [(1, 3), (4, 7)],
        [(0, 6), (1, 8)],
        [(1, 5), (2, 7)]
    ]
    weight = prim(adj)
    print(f"MST weight: {weight}")

if __name__ == "__main__":
    test_prim()
