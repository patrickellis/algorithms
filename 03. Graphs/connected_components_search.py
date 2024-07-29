"""DFS algorithm to find connected components.

Connected components are defined for those in an UNDIRECTED graph.
"""
from typing import List

adj = [[1],[0,2],[1],[4],[3]]
n = len(adj)
visited = [False]*n

def dfs(v: int, component: List[int]) -> None:
    visited[v] = True
    component.append(v)
    for edge in adj[v]:
        if not visited[edge]:
            dfs(edge, component)

def find_components(n: int, adj: List[List[int]]) -> List[List[int]]:
    components = []
    for v in range(n):
        if not visited[v]:
            component = []
            dfs(v, component)
            components.append(component)
    return components

if __name__ == "__main__":
    find_components(n, adj)
    for component in find_components(n, adj):
        print(component)
