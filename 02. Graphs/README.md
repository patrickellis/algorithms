# Graphs

## Glossary

- [Depth First Search](#depth-first-search)
  - [Iterative](#dfs-iterative)
  - [Recursive](#dfs-recursive)
- [Breadth First Search](#breadth-first-search)
  - [Iterative](#bfs-iterative)
  - [Recursive](#bfs-recursive)

## Depth First Search

> [!NOTE]  
> Time: O(V+E) V: vertex count, E: edge count  
> Space O(V+E)  

### Iterative

```Python
g = [] # Adjacency list representing graph
n = len(g) # Number of nodes in the graph

def dfs(at: int):
    stack = [at]
    visited = [False]*n

    while stack:
        node = stack.pop()
        visited.add(node)
        neighbours = g[node]

        for next in neighbours:
            if next not in visited:
                stack.append(next)
```

### Recursive

```Python
g = [] # Adjacency list representing graph
n = len(g) # Number of nodes in the graph
visited = [False]*n

def dfs(node, visited: set):
    if visited[node]: return
    visited[node] = True
    neighbours = g[node]

    for next in neighbours:
        dfs(next)
```

## Breadth First Search

> [!NOTE]  
> Time: O(V+E) V: vertex count, E: edge count  
> Space O(V)  

### Iterative

```Python
from collections import deque

g = [] # Adjacency list representing graph
n = len(g) # Number of nodes in the graph

def reconstructPath(s: int, e: int, prev: list[int]):
    path = []
    at = e
    while at:
        path.append(at)
        at = prev[at]

    path.reverse()

    # If s and e are connected return the path
    if path[0] == s:
        return path
    return []

def solve(at: int):
    queue = deque([at])
    visited = [False]*n
    prev = [None]*n

    while queue:
        node = queue.popleft()
        visited[node] = True
        neighbours = g[node]

        for next in neighbours:
            if !visited[next]:
                queue.append(next)
                prev[next] = node
    return prev

def bfs(s: int, e: int):
    prev = solve(s) # Do a BFS starting at node s

    return reconstructPath(s, e, prev) # return reconstructed path from s->e
```

### Recursive

- Uncommon

<details>
<summary>Click to view code</summary>

```Python
from collections import deque

g = [] # Adjacency list representing graph
n = len(g) # Number of nodes in the graph

def reconstructPath(s: int, e: int, prev: list[int]):
    path = []
    at = e
    while at:
        path.append(at)
        at = prev[at]

    path.reverse()

    # If s and e are connected return the path
    if path[0] == s:
        return path
    return []

def solve(queue: deque, visited: list[Bool], prev: list[int]):
    if not queue:
        return

    node = queue.popleft()
    visited[node] = True
    neighbours = g[node]

    for next in neighbours:
        if !visited[next]:
            queue.append(next)
            prev[next] = node

def bfs(s: int, e: int):
    queue = deque([s])
    visited = [False]*n
    prev = [None]*n

    solve(s) # Do a BFS starting at node s

    return reconstructPath(s, e, prev) # return reconstructed path from s->e

```

</details>
