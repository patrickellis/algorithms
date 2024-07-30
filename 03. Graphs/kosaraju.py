"""
1. Perform DFS on input graph G and push finished vertices onto a stack
   (NOTE: This is the same as in DFS Topological Sort, but be aware
   the graphs may be cyclic).
2. Create the transpose of G.
3. Iterate through the stack of vertices on the stack and perform DFS on each.
   Collect the vertices visited in each DFS as an SCC.
"""
from typing import List

def kosaraju(adj: List[List[int]]) -> List[List[int]]:
    components = []
    n = len(adj)
    stack = []
    visited = [False]*n

    def dfs(v: int, adj: List[List[int]], visited: List[bool], output: List[int]):
        visited[v] = True
        for edge in adj[v]:
            if not visited[edge]:
                dfs(edge, adj, visited, output)
        output.append(v)

    def transpose_graph(adj: List[List[int]]) -> List[List[int]]:
        adj_transpose = [[] for _ in range(n)]

        for v in range(n):
            for edge in adj[v]:
                adj_transpose[edge].append(v)

        return adj_transpose

    for i in range(n):
        if not visited[i]:
            dfs(i, adj, visited, stack)
    stack.reverse()

    adj_transpose = transpose_graph(adj)
    visited = [False]*n
    for v in stack:
        if not visited[v]:
            component = []
            dfs(v, adj_transpose, visited, component)
            components.append(component)

    return components

adjacency_list = [
    [1],        # 0 -> 1
    [2],        # 1 -> 2
    [0, 3],     # 2 -> 0, 2 -> 3
    [4],        # 3 -> 4
    [5],        # 4 -> 5
    [3],        # 5 -> 3
]

components = kosaraju(adjacency_list)
print("Strongly connected components:", components)

