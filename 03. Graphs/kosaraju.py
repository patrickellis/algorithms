
from typing import List

def dfs(v: int, adj: List[List[int]], visited: List[bool], output: List[int], collect: bool) -> None:
    visited[v] = True
    if collect:
        output.append(v)
    for u in adj[v]:
        if not visited[u]:
            dfs(u, adj, visited, output, collect)
    if not collect:
        output.append(v)

def strongly_connected_components(adj: List[List[int]]) -> List[List[int]]:
    n = len(adj)
    order = []
    visited = [False] * n

    # First pass: Order vertices by finish time
    for i in range(n):
        if not visited[i]:
            dfs(i, adj, visited, order, False)

    # Reverse the graph
    adj_rev = [[] for _ in range(n)]
    for v in range(n):
        for u in adj[v]:
            adj_rev[u].append(v)

    visited = [False] * n
    components = []

    # Second pass: Collect strongly connected components
    while order:
        v = order.pop()
        if not visited[v]:
            component = []
            dfs(v, adj_rev, visited, component, True)
            component.sort()
            components.append(component)

    return components

# Test case
def test_strongly_connected_components():
    adj = [
        [1],         # edges from vertex 0
        [2],         # edges from vertex 1
        [0, 3],      # edges from vertex 2
        [4],         # edges from vertex 3
        [5, 7],      # edges from vertex 4
        [6],         # edges from vertex 5
        [4],         # edges from vertex 6
        [8],         # edges from vertex 7
        [9],         # edges from vertex 8
        [7]          # edges from vertex 9
    ]
    adj = [
        [1, 3], [1, 4], [2, 1], [3, 2], [4, 5]
    ]
    adj = [[2,3],[0],[1],[4]]
    components = strongly_connected_components(adj)
    print("Strongly Connected Components:", components)

if __name__ == "__main__":
    test_strongly_connected_components()
