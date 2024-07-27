import heapq

# Adjacency list representing the graph
adj = [
    [(1, 2), (2, 4)],   # edges from node 0 to node 1 with weight 2, and to node 2 with weight 4
    [(2, 1), (3, 7)],   # edges from node 1 to node 2 with weight 1, and to node 3 with weight 7
    [(3, 3)],           # edge from node 2 to node 3 with weight 3
    []                  # node 3 has no outgoing edges
]

n = len(adj)  # Number of nodes in the graph
dist = [float('inf')] * n  # Distance from source to each node
p = [-1] * n  # Predecessors
visited = [False] * n  # Visited nodes

def dijkstra(s):
    dist[s] = 0
    pq = [(0, s)]  # Priority queue of (distance, node)

    while pq:
        _, v = heapq.heappop(pq)
        visited[v] = True

        for to, weight in adj[v]:
            if dist[v] + weight < dist[to]:
                dist[to] = dist[v] + weight
                p[to] = v
                heapq.heappush(pq, (dist[to], to))

def run_dijkstra_tests():
    test_cases = [
        {
            "name": "Basic functionality test",
            "adj": [
                [(1, 2), (2, 4)],
                [(2, 1), (3, 7)],
                [(3, 3)],
                []
            ],
            "source": 0,
            "expected_dist": [0, 2, 3, 6]
        },
        {
            "name": "Single node graph",
            "adj": [
                []
            ],
            "source": 0,
            "expected_dist": [0]
        },
        {
            "name": "Disconnected graph",
            "adj": [
                [(1, 1)],
                [],
                [(3, 1)],
                []
            ],
            "source": 0,
            "expected_dist": [0, 1, float('inf'), float('inf')]
        },
        {
            "name": "Graph with zero weight edges",
            "adj": [
                [(1, 0), (2, 2)],
                [(2, 0)],
                [(3, 1)],
                []
            ],
            "source": 0,
            "expected_dist": [0, 0, 0, 1]
        },
        {
            "name": "Graph with all edges having the same weight",
            "adj": [
                [(1, 1), (2, 1)],
                [(2, 1), (3, 1)],
                [(3, 1)],
                []
            ],
            "source": 0,
            "expected_dist": [0, 1, 1, 2]
        },
        {
            "name": "Graph with different weights",
            "adj": [
                [(1, 10), (2, 5)],
                [(2, 2), (3, 1)],
                [(1, 3), (3, 9)],
                [(2, 6)]
            ],
            "source": 0,
            "expected_dist": [0, 8, 5, 9]
        },
    ]

    for test in test_cases:
        # Reset variables
        n = len(test["adj"])
        dist = [float('inf')] * n
        p = [-1] * n
        visited = [False] * n

        # Redefine dijkstra to use the current test case's adj
        def dijkstra(s, adj=test["adj"]):
            dist[s] = 0
            pq = [(0, s)]
            while pq:
                _, v = heapq.heappop(pq)
                visited[v] = True
                for to, weight in adj[v]:
                    if dist[v] + weight < dist[to]:
                        dist[to] = dist[v] + weight
                        p[to] = v
                        if not visited[to]:
                            heapq.heappush(pq, (dist[to], to))

        dijkstra(test["source"])
        result = dist

        assert result == test["expected_dist"], f"Test {test['name']} failed: expected {test['expected_dist']}, got {result}"
        print(f"Test {test['name']} passed.")

run_dijkstra_tests()


