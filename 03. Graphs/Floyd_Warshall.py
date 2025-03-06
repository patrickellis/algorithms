def floyd_warshall(graph):
    nodes = list(graph.keys())
    dist = {u: {v: float('inf') for v in nodes} for u in nodes}

    for u in nodes:
        dist[u][u] = 0

    for u in graph:
        for v, w in graph[u].items():
            dist[u][v] = w

    for k in nodes:
        for i in nodes:
            for j in nodes:
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist

def find_diameter(dist):
    diameter = 0
    for u in dist:
        for v in dist[u]:
            if dist[u][v] != float('inf') and dist[u][v] > diameter:
                diameter = dist[u][v]
    return diameter

def graph_diameter(graph):
    dist = floyd_warshall(graph)
    diameter = find_diameter(dist)
    return diameter
