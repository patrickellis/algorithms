# Graphs

## Glossary

- [Depth First Search](#depth-first-search)
  - [Iterative](#dfs-iterative)
    - [Iterative: Shortest path with Matrix Input](#iterative-shortest-path-with-matrix-input)
  - [Recursive](#dfs-recursive)
- [Breadth First Search](#breadth-first-search)
  - [Iterative](#bfs-iterative)
  - [Recursive](#bfs-recursive)

## Depth First Search

> [!NOTE]  
> Time: O(V+E) V: vertex count, E: edge count  
> Space O(V+E)  

### Iterative DFS

```Python
adj = [] # Adjacency list representing graph
n = len(adj) # Number of vs in the graph

def dfs(at: int):
    stack = [at]
    visited = [False]*n

    while stack:
        v = stack.pop()
        visited[v] = True
        for next in adj[v]:
            if !visited[next]:
                stack.append(next)
```

### Recursive DFS

```Python
adj = [] # Adjacency list representing graph
n = len(adj) # Number of vs in the graph
visited = [False]*n

def dfs(v):
    visited[v] = True

    for next in adj[v]:
        if !visited[next]:
            dfs(next)
```

It can be useful to compute the entry and exit times and vertex color.
These changes are required:

```diff
adj = [] # Adjacency list representing graph
n = len(adj) # Number of vs in the graph
- visited = [False]*n
+ timer = 0
+ time_in = [None]*n
+ time_out = [None]*n
+ color = [0]*n # 0=unvisited, 1=visited, 2=exited

def dfs(v):
-   visited[v] = True
+   time_in[v] = timer
+   timer += 1
+   color[v] = 1

    for next in adj[v]:
+       if color[next] == 0:
-       if !visited[next]:
            dfs(next)
+   color[v] = 2
+   time_out[v] = timer
+   timer += 1
```

<details>
<summary>Click for the non-diff version with Python syntax highlighting.</summary>

```Python
adj = [] # Adjacency list representing graph
n = len(adj) # Number of vs in the graph
timer = 0
time_in = [None]*n
time_out = [None]*n
color = [0]*n # 0=unvisited, 1=visited, 2=exited

def dfs(v):
    time_in[v] = timer
    timer += 1
    color[v] = 1

    for next in adj[v]:
        if color[next] == 0:
            dfs(next)
    color[v] = 2
    time_out[v] = timer
    timer += 1
```

</details>

## Breadth First Search

> [!NOTE]  
> Time: O(V+E) V: vertex count, E: edge count  
> Space O(V)  

### Iterative

```Python
from collections import deque

adj = [] # Adjacency list representing graph
n = len(adj) # Number of vs in the graph

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
        v = queue.popleft()
        visited[v] = True
        for next in adj[v]:
            if !visited[next]:
                queue.append(next)
                prev[next] = v
    return prev

def bfs(s: int, e: int):
    prev = solve(s) # Do a BFS starting at v s
    return reconstructPath(s, e, prev) # return reconstructed path from s->e
```

#### Iterative: Shortest path with Matrix Input

Watch the accompanying video [here](https://www.youtube.com/watch?v=KiCBXu4P-2Y&list=PLDV1Zeh2NRsDGO4--qE8yH72HFL1Km93P&index=6).

```Python
m = [] # Input matrix of size R x C
R, C = len(m), len(m[0])
directions = ((0,1),(0,-1),(1,0),(-1,0))
reached_end = False
nodes_left_in_layer = 0
nodes_in_next_layer = 0
move_count = 0
queue = deque([s])
visited = set()

def explore_neighbours(v: tuple[int,int]):
    r,c = v
    for d in directions:
        rr = r+d[0]
        cc = c+d[1]

        # Skip out of bounds
        if rr < 0 or cc < 0: continue
        if rr >= R  or cc >= C: continue

        # Skip visited / blocked cells
        if (rr,cc) in visited: continue
        if m[rr][cc] == '#': continue

        queue.append((rr,cc))
        nodes_in_next_layer += 1

def solve(s: tuple[int,int], e: tuple[int,int]):
    while queue:
        v = queue.popleft()
        visited.add(v)

        if v == e:
            reached_end = True
            break

        explore_neighbours(v)
        nodes_left_in_layer -= 1

        if nodes_left_in_layer == 0:
            nodes_left_in_layer = nodes_in_next_layer
            nodes_in_next_layer = 0
            move_count += 1

        if reached_end:
            return move_count

    return -1

def bfs(s: tuple(int,int), e: tuple(int,int)):
    return solve(s,e) 

bfs((0,0),(15,12))
```

### Recursive

- Uncommon

<details>
<summary>Click to view code</summary>

```Python
from collections import deque

adj = [] # Adjacency list representing graph
n = len(adj) # Number of vs in the graph

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

    v = queue.popleft()
    visited[v] = True

    for next in adj[v]:
        if !visited[next]:
            queue.append(next)
            prev[next] = v

def bfs(s: int, e: int):
    queue = deque([s])
    visited = [False]*n
    prev = [None]*n

    solve(queue, visited, prev) # Do a BFS starting at v s

    return reconstructPath(s, e, prev) # return reconstructed path from s->e

```

</details>
