
from collections import defaultdict, deque

def kahn_topological_sort(graph):
    in_degree = {u: 0 for u in graph}
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    queue = deque([u for u in in_degree if in_degree[u] == 0])

    top_order = []

    while queue:
        u = queue.popleft()
        top_order.append(u)

        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(top_order) == len(in_degree):
        return top_order
    else:
        return []

# Example usage:
if __name__ == "__main__":
    # Define the graph as an adjacency list
    graph = {
        'A': ['C'],
        'B': ['C', 'D'],
        'C': ['E'],
        'D': ['F'],
        'E': ['H', 'F'],
        'F': ['G'],
        'G': [],
        'H': []
    }

    result = kahn_topological_sort(graph)
    if result:
        print("Topological Order:", result)
    else:
        print("The graph has a cycle.")
