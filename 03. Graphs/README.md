# Graphs

## Glossary

- [Depth First Search](#depth-first-search)
  - [Iterative](#dfs-iterative)
    - [Iterative: Shortest path with Matrix Input](#iterative-shortest-path-with-matrix-input)
  - [Recursive](#dfs-recursive)
- [Breadth First Search](#breadth-first-search)
  - [Iterative](#bfs-iterative)
  - [Recursive](#bfs-recursive)
- [Topological Sort](#topological-sort)
- [Dijkstra](#dijkstra)

## Depth First Search

> [!NOTE]  
> Time: O(V+E) V: vertex count, E: edge count  
> Space O(V+E)  

### Iterative DFS

```Python
adj = [] # Adjacency list representing graph
n = len(adj) # Number of nodes in the graph

def dfs(at: int):
    stack = [at]
    visited = [False]*n

    while stack:
        v = stack.pop()
        visited[v] = True
        for edge in adj[v]:
            if !visited[edge]:
                stack.append(edge)
```

### Recursive DFS

```Python
adj = [] # Adjacency list representing graph
n = len(adj) # Number of nodes in the graph
visited = [False]*n

def dfs(v):
    visited[v] = True

    for edge in adj[v]:
        if !visited[edge]:
            dfs(edge)
```

It can be useful to compute the entry and exit times and vertex color.
These changes are required:

```diff
adj = [] # Adjacency list representing graph
n = len(adj) # Number of nodes in the graph
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

    for edge in adj[v]:
+       if color[edge] == 0:
-       if !visited[edge]:
            dfs(edge)
+   color[v] = 2
+   time_out[v] = timer
+   timer += 1
```

<details>
<summary>Click for the non-diff version with Python syntax highlighting.</summary>

```Python
adj = [] # Adjacency list representing graph
n = len(adj) # Number of nodes in the graph
timer = 0
time_in = [None]*n
time_out = [None]*n
color = [0]*n # 0=unvisited, 1=visited, 2=exited

def dfs(v):
    time_in[v] = timer
    timer += 1
    color[v] = 1

    for edge in adj[v]:
        if color[edge] == 0:
            dfs(edge)
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
n = len(adj) # Number of nodes in the graph


def reconstructPath(s: int, e: int, p: list[int]):
    path = []
    at = e
    while at:
        path.append(at)
        at = p[at]
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
        for edge in adj[v]:
            if !visited[edge]:
                queue.append(edge)
                p[edge] = v
    return p

def bfs(s: int, e: int):
    prev = solve(s) # Do a BFS starting at v s
    return reconstructPath(s, e, p) # return reconstructed path from s->e
```

#### Iterative: Shortest path with Matrix Input

Watch the accompanying video [here](https://www.youtube.com/watch?v=KiCBXu4P-2Y&list=PLDV1Zeh2NRsDGO4--qE8yH72HFL1Km93P&index=6).

```Python
m = [] # Input matrix of size R x C
R, C = len(m), len(m[0])
directions = ((0,1),(0,-1),(1,0),(-1,0))
reached_end = False
nodes_left_in_layer = 0
nodes_in_edge_layer = 0
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
        nodes_in_edge_layer += 1

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
            nodes_left_in_layer = nodes_in_edge_layer
            nodes_in_edge_layer = 0
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
n = len(adj) # Number of nodes in the graph

def reconstructPath(s: int, e: int, p: list[int]):
    path = []
    at = e
    while at:
        path.append(at)
        at = p[at]

    path.reverse()

    # If s and e are connected return the path
    if path[0] == s:
        return path
    return []

def solve(queue: deque, visited: list[Bool], p: list[int]):
    if not queue:
        return

    v = queue.popleft()
    visited[v] = True

    for edge in adj[v]:
        if !visited[edge]:
            queue.append(edge)
            p[edge] = v

def bfs(s: int, e: int):
    queue = deque([s])
    visited = [False]*n
    p = [None]*n # predecessors

    solve(queue, visited, p) # Do a BFS starting at v s

    return reconstructPath(s, e, p) # return reconstructed path from s->e

```

</details>

## Topological Sort

- [Reference](https://cp-algorithms.com/graph/topological-sort.html)

1. Starting from some vertex $v$, DFS attempts to traverse along all edges outgoing from $v$.  
   By the time $dfs(v)$ has finished, all nodes that are reachable from $v$ have been visited.
2. After $dfs(v)$ completes, append the vertex $v$ to the result.
3. Repeat for every vertex in the graph, with one or multiple depth-first search runs.  
4. Reverse the resulting list.

For every directed edge $v → u$, $u$ will appear earlier in the list than $v$, because $u$ is reachable from $v$.  
If we simply label the vertices in this list with $n-1,n-2,...,1,0$, we have found a topological order of the graph.

**I.e. The list represents the reversed topological order.**  

<details>
<summary>Click to view an explanation in terms of exit times.</summary>

- The exit time for a vertex $v$ is the time at which the function call $dfs(v)$ finished.  
- The exit time of any vertex $v$ is always greater than the exit time of any vertex reachable from it  
*(since they were visited either before the call $dfs(v)$ or during it)*.  
- **Thus, the desired topological ordering is the list of vertices ordered according to descending exit times**.

</details>

```Python
adj = [] # Adjacency list representing graph
n = len(adj) # Number of nodes in the graph
visited = [False]*n
res = []

def dfs(v):
    visited[v] = True
    for edge in adj[v]:
        if !visited[edge]:
            dfs(edge)
    res.append(v)

def topological_sort():
    for i in range(n):
        if !visited[i]:
            dfs(i)
    res.reverse()
```

## Dijkstra

- [Reference](https://cp-algorithms.com/graph/dijkstra.html)

1. Create an array $dist[]$ where for each vertex $v$ we store the current lenth of the shortest path from $s$ to $v$ in $dist[v]$.
2. Initially, $dist[s] = 0$, and for all other vertices this length equals infinity.
$$d[v]=∞,v\neq s$$
3. In addition, maintain a Boolean array $u[]$ which stores for each vertex $v$ whether it's marked.  
   Initially all vertices are **unmarked**:
   $$u[v] = False, \forall v \in[0..n]$$   
   The main assertion on which Dijkstra's algorithm correctness is based is the following:  
   **After any vertex $v$  becomes marked, the current distance to it $d[v]$  is the shortest, and will no longer change.**  
4. The Dijkstra's algorithm runs for $n$ iterations.  
   At each iteration, it selects an unmarked vertex $v$ with the lowest value $dist[v]$.  
5. $v$ is marked. 
6. All of the edges of the form $(v,to)$ are checked, attempting to improve the value $d[to]$ for each vertex $to$.  
$$d[to]=\min(d[to],d[v]+len)$$
7. After $n$ iterations, all vertices will be marked, and the algorithm terminates.  
   The values $d[v]$ are the lengths of shortest paths from $s$ to all vertices $v$.

```Python
adj = [] # Adjacency list representing graph
n = len(adj) # Number of nodes in the graph
visited = [False]*n
dist = [float('inf')]*n
p = [-1]*n # predecessors

def dijkstra(s):
    u = [False]*n
    dist[s] = 0
    
    for i in range(n):
        v = -1
        for j in range(n):
            if (!u[j] and (v == -1 or dist[j] < dist[v])):
                v = j

        if dist[v] == float('inf'):
            break

        u[v] = True
        for edge in adj[v]:
            to, len = edge
            if dist[v] + len < dist[to]:
                dist[to] = dist[v]+len
                p[to] = v
```

### Restoring Shortest Paths

Build an array of predecessors like so: for each sucecssful relaxation, i.e. when for some  selected vertex $v$, there is an improvement in the distance to some vertex $to$, we update the predecessor vertex for $to$ with vertex $v$: 

$$p[to]=v$$

The shortest path to any vertex `e` can be restored like so:

```Python
def reconstructPath(s: int, e: int, p: list[int]):
    path = []
    at = e
    while at:
        path.append(at)
        at = p[at]
    path.reverse()
    # If s and e are connected return the path
    if path[0] == s:
        return path
    return []
```

### Runtime

The running time consists of:

- $n$ searches for a vertex with the smallest value $d[v]$ among $O(n)$ unmarked vertices.
- $m$ relaxation attempts.

For the simplest implementation of the *vertex search*, it requires $O(n)$ operations, and each relaxation can be performed in $O(1)$, thus:

$$O(n^2+m)$$
