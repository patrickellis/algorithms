from collections import deque

adj = {"A": ["B", "D"], "B": ["A", "C"], "C": ["B", "D"], "D": ["A", "C"]}
n = len(adj)


def is_bipartite():
    color = {}
    for v in range(n):
        if v not in color:
            queue = deque([v])
            color[v] = 0
            while queue:
                v = queue.popleft()
                for edge in adj[v]:
                    if edge not in color:
                        color[edge] = 1 - color[v]
                        queue.append(edge)
                    elif color[edge] == color[v]:
                        return False
    return True
