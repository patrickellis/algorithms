import heapq

adj = []
n = len(adj)


def prim() -> None:
    pq = [(0, 0)]
    total_weight = 0
    visited = [False] * n

    while pq:
        w, v = heapq.heappop(pq)
        if visited[v]:
            continue
        visited[v] = True
        total_weight += w

        for to, weight in adj[v]:
            heapq.heappush(pq, (weight, to))

    return total_weight


def test_prim():
    adj = [
        [(1, 2), (3, 6)],
        [(0, 2), (2, 3), (3, 8), (4, 5)],
        [(1, 3), (4, 7)],
        [(0, 6), (1, 8)],
        [(1, 5), (2, 7)],
    ]
    weight = prim(adj)
    print(f"MST weight: {weight}")


if __name__ == "__main__":
    test_prim()
