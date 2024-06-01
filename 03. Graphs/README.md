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


## Single-source Shortest Path

### Topological Sort
    
1. Perform Topological sort.
2. Traverse graph in sorted order. I.e. pick starting vertex from the sorted output and explore its neighbours,  maintaining shortest distance.
3. Time Complexity: O(V+E)

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

### Dijkstra
    
    - Used only if weights are non-negative (positive).
    - Use Priority Queue with Integer Array intead of Quue with Integer. (?)
    - Use Distance array instead of boolean visited array.
    - Time Complexity is $O(ElogE) -> O(Elog V^2) -> O(E*2 log v) -> O(ElogV)$
    - https://leetcode.com/problems/network-delay-time
    - https://leetcode.com/problems/cheapest-flights-within-k-stops/ - Imp as slight variation (?)

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

#### Restoring Shortest Paths

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

#### Runtime

The running time consists of:

- $n$ searches for a vertex with the smallest value $d[v]$ among $O(n)$ unmarked vertices.
- $m$ relaxation attempts.

For the simplest implementation of the *vertex search*, it requires $O(n)$ operations, and each relaxation can be performed in $O(1)$, thus:

$$O(n^2+m)$$

#### Bellman Ford

- [Video](https://www.youtube.com/watch?v=lyw4FaxrwHg&list=PLDV1Zeh2NRsDGO4--qE8yH72HFL1Km93P&index=20)
- Able to find negative cycles.
- Used with negative weights, also.
- Not preferred over Dijkstra, because time complexity is O(VE).
- Run this algorithm a second time if a negative cycle check is required. If the shortest distance of a vertex is reduced, then the graph has a negative cycle.

##### Edge List (fast and optimized) - O(VE)

- [Code Source](https://github.com/williamfiset/Algorithms/blob/master/src/main/java/com/williamfiset/algorithms/graphtheory/BellmanFordEdgeList.java)

```Python
class Edge:
    def __init__(self, from_node, to_node, cost):
        self.from_node = from_node
        self.to_node = to_node
        self.cost = cost

def bellman_ford(edges, V, start):
    dist = [float('inf')] * V
    dist[start] = 0

    # Only in the worst case does it take V-1 iterations for the Bellman-Ford
    # algorithm to complete. Another stopping condition is when we're unable to
    # relax an edge, this means we have reached the optimal solution early.
    relaxed_an_edge = True

    # For each vertex, apply relaxation for all the edges
    for _ in range(V - 1):
        if not relaxed_an_edge:
            break
        relaxed_an_edge = False
        for edge in edges:
            if dist[edge.from_node] + edge.cost < dist[edge.to_node]:
                dist[edge.to_node] = dist[edge.from_node] + edge.cost
                relaxed_an_edge = True

    # Run algorithm a second time to detect which nodes are part
    # of a negative cycle. A negative cycle has occurred if we
    # can find a better path beyond the optimal solution.
    relaxed_an_edge = True
    for _ in range(V - 1):
        if not relaxed_an_edge:
            break
        relaxed_an_edge = False
        for edge in edges:
            if dist[edge.from_node] + edge.cost < dist[edge.to_node]:
                dist[edge.to_node] = float('-inf')
                relaxed_an_edge = True

    return dist

if __name__ == "__main__":
    E, V, start = 10, 9, 0
    edges = [
        Edge(0, 1, 1),
        Edge(1, 2, 1),
        Edge(2, 4, 1),
        Edge(4, 3, -3),
        Edge(3, 2, 1),
        Edge(1, 5, 4),
        Edge(1, 6, 4),
        Edge(5, 6, 5),
        Edge(6, 7, 4),
        Edge(5, 7, 3)
    ]

    distances = bellman_ford(edges, V, start)

    for i in range(V):
        print(f"The cost to get from node {start} to {i} is {distances[i]:.2f}")

    # Output:
    # The cost to get from node 0 to 0 is 0.00
    # The cost to get from node 0 to 1 is 1.00
    # The cost to get from node 0 to 2 is -Infinity
    # The cost to get from node 0 to 3 is -Infinity
    # The cost to get from node 0 to 4 is -Infinity
    # The cost to get from node 0 to 5 is 5.00
    # The cost to get from node 0 to 6 is 5.00
    # The cost to get from node 0 to 7 is 8.00
    # The cost to get from node 0 to 8 is Infinity

```

##### Adjacency List - O(VE)

- [Code Source](https://github.com/williamfiset/Algorithms/blob/master/src/main/java/com/williamfiset/algorithms/graphtheory/BellmanFordAdjacencyMatrix.java).

```
class Edge:
    def __init__(self, from_node, to_node, cost):
        self.from_node = from_node
        self.to_node = to_node
        self.cost = cost

def create_graph(V):
    return [[] for _ in range(V)]

def add_edge(graph, frm, to, cost):
    graph[frm].append(Edge(frm, to, cost))

def bellman_ford(graph, V, start):
    # Initialize the distance to all nodes to be infinity except for the start node which is zero.
    dist = [float('inf')] * V
    dist[start] = 0

    # For each vertex, apply relaxation for all the edges
    for _ in range(V - 1):
        for edges in graph:
            for edge in edges:
                if dist[edge.from_node] + edge.cost < dist[edge.to_node]:
                    dist[edge.to_node] = dist[edge.from_node] + edge.cost

    # Run algorithm a second time to detect which nodes are part of a negative cycle.
    for _ in range(V - 1):
        for edges in graph:
            for edge in edges:
                if dist[edge.from_node] + edge.cost < dist[edge.to_node]:
                    dist[edge.to_node] = float('-inf')

    return dist

if __name__ == "__main__":
    E, V, start = 10, 9, 0
    graph = create_graph(V)
    add_edge(graph, 0, 1, 1)
    add_edge(graph, 1, 2, 1)
    add_edge(graph, 2, 4, 1)
    add_edge(graph, 4, 3, -3)
    add_edge(graph, 3, 2, 1)
    add_edge(graph, 1, 5, 4)
    add_edge(graph, 1, 6, 4)
    add_edge(graph, 5, 6, 5)
    add_edge(graph, 6, 7, 4)
    add_edge(graph, 5, 7, 3)

    distances = bellman_ford(graph, V, start)

    for i in range(V):
        print(f"The cost to get from node {start} to {i} is {distances[i]:.2f}")

    # Output:
    # The cost to get from node 0 to 0 is 0.00
    # The cost to get from node 0 to 1 is 1.00
    # The cost to get from node 0 to 2 is -Infinity
    # The cost to get from node 0 to 3 is -Infinity
    # The cost to get from node 0 to 4 is -Infinity
    # The cost to get from node 0 to 5 is 5.00
    # The cost to get from node 0 to 6 is 5.00
    # The cost to get from node 0 to 7 is 8.00
    # The cost to get from node 0 to 8 is Infinity
```


##### Adjacency Matrix - O(V^3)

- [Code Source](https://github.com/williamfiset/Algorithms/blob/master/src/main/java/com/williamfiset/algorithms/graphtheory/BellmanFordAdjacencyMatrix.java).

```Python
class BellmanFordAdjacencyMatrix:
    def __init__(self, start, matrix):
        """
        An implementation of the Bellman-Ford algorithm. The algorithm finds the shortest path between
        a starting node and all other nodes in the graph. The algorithm also detects negative cycles.
        If a node is part of a negative cycle then the minimum cost for that node is set to
        float('-inf').
        
        :param matrix: An adjacency matrix containing directed edges forming the graph
        :param start: The id of the starting node
        """
        self.n = len(matrix)
        self.start = start
        self.matrix = [row[:] for row in matrix]  # Copy input adjacency matrix
        self.solved = False
        self.dist = [float('inf')] * self.n
        self.prev = [None] * self.n

    def get_shortest_paths(self):
        if not self.solved:
            self.solve()
        return self.dist

    def reconstruct_shortest_path(self, end):
        if not self.solved:
            self.solve()
        path = []
        if self.dist[end] == float('inf'):
            return path
        at = end
        while self.prev[at] is not None:
            if self.prev[at] == -1:
                return None  # Infinite number of shortest paths
            path.append(at)
            at = self.prev[at]
        path.append(self.start)
        return path[::-1]

    def solve(self):
        if self.solved:
            return

        # Initialize the distance to all nodes to be infinity
        # except for the start node which is zero.
        self.dist[self.start] = 0

        # For each vertex, apply relaxation for all the edges
        for _ in range(self.n - 1):
            for i in range(self.n):
                for j in range(self.n):
                    if self.dist[i] + self.matrix[i][j] < self.dist[j]:
                        self.dist[j] = self.dist[i] + self.matrix[i][j]
                        self.prev[j] = i

        # Run algorithm a second time to detect which nodes are part
        # of a negative cycle. A negative cycle has occurred if we
        # can find a better path beyond the optimal solution.
        for _ in range(self.n - 1):
            for i in range(self.n):
                for j in range(self.n):
                    if self.dist[i] + self.matrix[i][j] < self.dist[j]:
                        self.dist[j] = float('-inf')
                        self.prev[j] = -1

        self.solved = True

def main():
    n = 9
    graph = [[float('inf')] * n for _ in range(n)]

    # Setup completely disconnected graph with the distance
    # from a node to itself to be zero.
    for i in range(n):
        graph[i][i] = 0

    graph[0][1] = 1
    graph[1][2] = 1
    graph[2][4] = 1
    graph[4][3] = -3
    graph[3][2] = 1
    graph[1][5] = 4
    graph[1][6] = 4
    graph[5][6] = 5
    graph[6][7] = 4
    graph[5][7] = 3

    start = 0
    solver = BellmanFordAdjacencyMatrix(start, graph)
    d = solver.get_shortest_paths()

    for i in range(n):
        print(f"The cost to get from node {start} to {i} is {d[i]:.2f}")

    # Output the shortest paths
    for i in range(n):
        path = solver.reconstruct_shortest_path(i)
        if path is None:
            str_path = "Infinite number of shortest paths."
        else:
            str_path = " -> ".join(map(str, path))
        print(f"The shortest path from {start} to {i} is: [{str_path}]")

if __name__ == "__main__":
    main()
```

## All-pair Shortest Path

### Floyd Warshall

- [Video](https://www.youtube.com/watch?v=4NQ3HnhyNfQ)
- This algorithm prefers adjacency matrix over adjaacency list.
- Time complexity: $O(V^3)$
- Run this algorithm a second time if a negative cycle check is required. If the shortest distance of a vertex is reduced, then the graph has a negative cycle.
- [Leetcode Practice Problem](https://leetcode.com/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/)

[Source code](https://github.com/williamfiset/Algorithms/blob/master/src/main/java/com/williamfiset/algorithms/graphtheory/FloydWarshallSolver.java).

```Python
import math

class FloydWarshallSolver:
    REACHES_NEGATIVE_CYCLE = -1

    def __init__(self, matrix):
        """
        As input, this class takes an adjacency matrix with edge weights between nodes, where
        math.inf is used to indicate that two nodes are not connected.

        NOTE: Usually the diagonal of the adjacency matrix is all zeros (i.e. matrix[i][i] = 0 for
        all i) since there is typically no cost to go from a node to itself, but this may depend on
        your graph and the problem you are trying to solve.
        """
        self.n = len(matrix)
        self.dp = [[matrix[i][j] for j in range(self.n)] for i in range(self.n)]
        self.next = [[j if matrix[i][j] != math.inf else None for j in range(self.n)] for i in range(self.n)]
        self.solved = False

    def get_apsp_matrix(self):
        self.solve()
        return self.dp

    def solve(self):
        if self.solved:
            return

        # Compute all pairs shortest paths.
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    if self.dp[i][k] + self.dp[k][j] < self.dp[i][j]:
                        self.dp[i][j] = self.dp[i][k] + self.dp[k][j]
                        self.next[i][j] = self.next[i][k]

        # Identify negative cycles by propagating the value 'math.inf'
        # to every edge that is part of or reaches into a negative cycle.
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    if self.dp[i][k] != math.inf and self.dp[k][j] != math.inf and self.dp[k][k] < 0:
                        self.dp[i][j] = -math.inf
                        self.next[i][j] = self.REACHES_NEGATIVE_CYCLE

        self.solved = True

    def reconstruct_shortest_path(self, start, end):
        self.solve()
        path = []
        if self.dp[start][end] == math.inf:
            return path
        at = start
        while at != end:
            if at == self.REACHES_NEGATIVE_CYCLE:
                return None  # Infinite number of shortest paths
            path.append(at)
            at = self.next[at][end]
        if self.next[at][end] == self.REACHES_NEGATIVE_CYCLE:
            return None  # Infinite number of shortest paths
        path.append(end)
        return path

    @staticmethod
    def create_graph(n):
        matrix = [[math.inf] * n for _ in range(n)]
        for i in range(n):
            matrix[i][i] = 0
        return matrix

def main():
    # Construct graph.
    n = 7
    m = FloydWarshallSolver.create_graph(n)

    # Add some edge values.
    m[0][1] = 2
    m[0][2] = 5
    m[0][6] = 10
    m[1][2] = 2
    m[1][4] = 11
    m[2][6] = 2
    m[6][5] = 11
    m[4][5] = 1
    m[5][4] = -2

    solver = FloydWarshallSolver(m)
    dist = solver.get_apsp_matrix()

    for i in range(n):
        for j in range(n):
            print(f"This shortest path from node {i} to node {j} is {dist[i][j]:.3f}")

    print()

    # Reconstructs the shortest paths from all nodes to every other nodes.
    for i in range(n):
        for j in range(n):
            path = solver.reconstruct_shortest_path(i, j)
            if path is None:
                str_path = "HAS AN ∞ NUMBER OF SOLUTIONS! (negative cycle case)"
            elif len(path) == 0:
                str_path = f"DOES NOT EXIST (node {i} doesn't reach node {j})"
            else:
                str_path = " -> ".join(map(str, path))
                str_path = f"is: [{str_path}]"

            print(f"The shortest path from node {i} to node {j} {str_path}")

if __name__ == "__main__":
    main()
```

## Strongly Connected Components

- [Strongly Connected Components](https://en.wikipedia.org/wiki/Strongly_connected_component)

### Overview

> [IMPORTANT]
> In order to find SCCs:
> - For an **undirected graph**, use DFS and visited array.
> - For a **directed graph**, use Kosaraju's or Tarjan's algorithm.
> - Kosaraju's algorithm is slower than Tarjan's, due to performing DFS twice rather than once.


### Tarjan's Algorithm (Time: $O(\text{V+E})$)

- [Video](https://www.youtube.com/watch?v=wUgWX0nc4NY)
- Used to find __SCC, articulation points, and bridges.__
- Time complexity: $O(\text{V+E})$
- [Bridges and Articulation Points [Video]](https://www.youtube.com/watch?v=aZXi1unBdJA)
- [Practice problem for bridges and articulation points](https://leetcode.com/problems/critical-connections-in-a-network)

```Python

def strongly_connected_components(graph):

    index_counter = [0]
    stack = []
    lowlinks = {}
    index = {}
    result = []
    
    def strongconnect(node):
        # set the depth index for this node to the smallest unused index
        index[node] = index_counter[0]
        lowlinks[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
    
        # Consider successors of `node`
        try:
            successors = graph[node]
        except:
            successors = []
        for successor in successors:
            if successor not in lowlinks:
                # Successor has not yet been visited; recurse on it
                strongconnect(successor)
                lowlinks[node] = min(lowlinks[node],lowlinks[successor])
            elif successor in stack:
                # the successor is in the stack and hence in the current strongly connected component (SCC)
                lowlinks[node] = min(lowlinks[node],index[successor])
        
        # If `node` is a root node, pop the stack and generate an SCC
        if lowlinks[node] == index[node]:
            connected_component = []
            
            while True:
                successor = stack.pop()
                connected_component.append(successor)
                if successor == node: break
            component = tuple(connected_component)
            # storing the result
            result.append(component)
    
    for node in graph:
        if node not in lowlinks:
            strongconnect(node)
    
    return result
```


#### Adjacency Matrix

[Source code](https://github.com/williamfiset/Algorithms/blob/master/src/main/java/com/williamfiset/algorithms/graphtheory/TarjanAdjacencyMatrix.java).

```Python
class TarjanAdjacencyMatrix:
    """
    Tarjan's algorithm finds the strongly connected components of a graph.
    Based on: http://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
    """

    def __init__(self, adj):
        self.n = len(adj)
        self.adj = adj
        self.marked = [False] * self.n
        self.id = [0] * self.n
        self.low = [0] * self.n
        self.stack = []
        self.pre = 0
        self.count = 0
        for u in range(self.n):
            if not self.marked[u]:
                self._dfs(u)

    def _dfs(self, u):
        self.marked[u] = True
        self.low[u] = self.pre
        min_low = self.low[u]
        self.pre += 1
        self.stack.append(u)

        for v in range(self.n):
            if self.adj[u][v]:
                if not self.marked[v]:
                    self._dfs(v)
                if self.low[v] < min_low:
                    min_low = self.low[v]

        if min_low < self.low[u]:
            self.low[u] = min_low
            return

        while True:
            v = self.stack.pop()
            self.id[v] = self.count
            self.low[v] = self.n
            if v == u:
                break
        self.count += 1

    def get_strongly_connected_components(self):
        return self.id[:]

    def count_strongly_connected_components(self):
        return self.count


if __name__ == "__main__":
    # Example usage:

    # As an example we create a graph with four strongly connected components
    NUM_NODES = 10

    adj_matrix = [[False] * NUM_NODES for _ in range(NUM_NODES)]

    # SCC 1 with nodes 0, 1, 2
    adj_matrix[0][1] = True
    adj_matrix[1][2] = True
    adj_matrix[2][0] = True

    # SCC 2 with nodes 3, 4, 5, 6
    adj_matrix[5][4] = True
    adj_matrix[5][6] = True
    adj_matrix[3][5] = True
    adj_matrix[4][3] = True
    adj_matrix[4][5] = True
    adj_matrix[6][4] = True

    # SCC 3 with nodes 7, 8
    adj_matrix[7][8] = True
    adj_matrix[8][7] = True

    # SCC 4 is node 9 all alone by itself
    # Add a few more edges to make things interesting
    adj_matrix[1][5] = True
    adj_matrix[1][7] = True
    adj_matrix[2][7] = True
    adj_matrix[6][8] = True
    adj_matrix[9][8] = True
    adj_matrix[9][4] = True

    sccs = TarjanAdjacencyMatrix(adj_matrix)

    print("Strong connected component count: ", sccs.count_strongly_connected_components())
    print("Strong connected components:\n", sccs.get_strongly_connected_components())

    # Output:
    # Strong connected component count: 4
    # Strong connected components:
    # [2, 2, 2, 1, 1, 1, 1, 0, 0, 3]

```

#### Adjacency List

```Python

class TarjanSccSolverAdjacencyList:
    UNVISITED = -1

    def __init__(self, graph):
        if graph is None:
            raise ValueError("Graph cannot be null.")
        self.graph = graph
        self.n = len(graph)
        self.solved = False
        self.sccCount = 0
        self.id = 0
        self.ids = [self.UNVISITED] * self.n
        self.low = [0] * self.n
        self.sccs = [0] * self.n
        self.visited = [False] * self.n
        self.stack = []

    def scc_count(self):
        if not self.solved:
            self.solve()
        return self.sccCount

    def get_sccs(self):
        if not self.solved:
            self.solve()
        return self.sccs

    def solve(self):
        if self.solved:
            return

        for i in range(self.n):
            if self.ids[i] == self.UNVISITED:
                self._dfs(i)

        self.solved = True

    def _dfs(self, at):
        self.ids[at] = self.low[at] = self.id
        self.id += 1
        self.stack.append(at)
        self.visited[at] = True

        for to in self.graph[at]:
            if self.ids[to] == self.UNVISITED:
                self._dfs(to)
            if self.visited[to]:
                self.low[at] = min(self.low[at], self.low[to])

        if self.ids[at] == self.low[at]:
            while True:
                node = self.stack.pop()
                self.visited[node] = False
                self.sccs[node] = self.sccCount
                if node == at:
                    break
            self.sccCount += 1

def create_graph(n):
    return [[] for _ in range(n)]

def add_edge(graph, from_node, to_node):
    graph[from_node].append(to_node)

# Example usage:
if __name__ == "__main__":
    n = 8
    graph = create_graph(n)

    add_edge(graph, 6, 0)
    add_edge(graph, 6, 2)
    add_edge(graph, 3, 4)
    add_edge(graph, 6, 4)
    add_edge(graph, 2, 0)
    add_edge(graph, 0, 1)
    add_edge(graph, 4, 5)
    add_edge(graph, 5, 6)
    add_edge(graph, 3, 7)
    add_edge(graph, 7, 5)
    add_edge(graph, 1, 2)
    add_edge(graph, 7, 3)
    add_edge(graph, 5, 0)

    solver = TarjanSccSolverAdjacencyList(graph)

    sccs = solver.get_sccs()
    multimap = {}
    for i in range(n):
        if sccs[i] not in multimap:
            multimap[sccs[i]] = []
        multimap[sccs[i]].append(i)

    print(f"Number of Strongly Connected Components: {solver.scc_count()}")
    for scc in multimap.values():
        print(f"Nodes: {scc} form a Strongly Connected Component.")
```

## Minimum Spanning Tree

### Prim's Algorithm


- [Video](https://www.youtube.com/watch?v=jsmMtJpPnhU)
- Start with any vertex. Use Priority Queue to process the smallest edge.
- Use visited array or distance array.
- Difference between Prims and Dijkstra is “Don’t add current vertex distance to calculate neighbour distance”.
- Example : u, v
- Dijkstra - dis[v] = dis[u] + graph[u][v];
- Prims - dis[v] = graph[u][v]
- Time Complexity is O(ElogV)
- https://www.youtube.com/watch?v=oP2-8ysT3QQ&t=430s&ab_channel=TusharRoy-CodingMadeSimple
- [Leetcode Practice Problem](https://leetcode.com/problems/min-cost-to-connect-all-points)

```Python
import heapq

class Edge:
    def __init__(self, from_node, to_node, cost):
        self.from_node = from_node
        self.to_node = to_node
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost


class LazyPrimsAdjacencyList:
    def __init__(self, graph):
        if graph is None or not graph:
            raise ValueError("Graph cannot be null or empty.")
        self.n = len(graph)
        self.graph = graph
        self.solved = False
        self.mstExists = False
        self.visited = [False] * self.n
        self.pq = []
        self.minCostSum = 0
        self.mstEdges = []

    def get_mst(self):
        self.solve()
        return self.mstEdges if self.mstExists else None

    def get_mst_cost(self):
        self.solve()
        return self.minCostSum if self.mstExists else None

    def add_edges(self, node_index):
        self.visited[node_index] = True
        for edge in self.graph[node_index]:
            if not self.visited[edge.to_node]:
                heapq.heappush(self.pq, edge)

    def solve(self):
        if self.solved:
            return
        self.solved = True

        m = self.n - 1
        edge_count = 0
        self.add_edges(0)

        while self.pq and edge_count != m:
            edge = heapq.heappop(self.pq)
            node_index = edge.to_node

            if self.visited[node_index]:
                continue

            self.mstEdges.append(edge)
            self.minCostSum += edge.cost
            edge_count += 1

            self.add_edges(node_index)

        self.mstExists = (edge_count == m)


def create_empty_graph(n):
    return [[] for _ in range(n)]


def add_directed_edge(graph, from_node, to_node, cost):
    graph[from_node].append(Edge(from_node, to_node, cost))


def add_undirected_edge(graph, from_node, to_node, cost):
    add_directed_edge(graph, from_node, to_node, cost)
    add_directed_edge(graph, to_node, from_node, cost)


if __name__ == "__main__":
    # Example usage:

    def example1():
        n = 10
        graph = create_empty_graph(n)

        add_undirected_edge(graph, 0, 1, 5)
        add_undirected_edge(graph, 1, 2, 4)
        add_undirected_edge(graph, 2, 9, 2)
        add_undirected_edge(graph, 0, 4, 1)
        add_undirected_edge(graph, 0, 3, 4)
        add_undirected_edge(graph, 1, 3, 2)
        add_undirected_edge(graph, 2, 7, 4)
        add_undirected_edge(graph, 2, 8, 1)
        add_undirected_edge(graph, 9, 8, 0)
        add_undirected_edge(graph, 4, 5, 1)
        add_undirected_edge(graph, 5, 6, 7)
        add_undirected_edge(graph, 6, 8, 4)
        add_undirected_edge(graph, 4, 3, 2)
        add_undirected_edge(graph, 5, 3, 5)
        add_undirected_edge(graph, 3, 6, 11)
        add_undirected_edge(graph, 6, 7, 1)
        add_undirected_edge(graph, 3, 7, 2)
        add_undirected_edge(graph, 7, 8, 6)

        solver = LazyPrimsAdjacencyList(graph)
        cost = solver.get_mst_cost()

        if cost is None:
            print("No MST exists")
        else:
            print("MST cost: ", cost)
            for edge in solver.get_mst():
                print(f"from: {edge.from_node}, to: {edge.to_node}, cost: {edge.cost}")

    example1()
```


### Kruskal

- [Video](https://www.youtube.com/watch?v=JZBQLXgSGfs)
- Sort all the edges by their weights and use union find to avoid cycle
- Time Complexity is O(ElogE)
- https://www.youtube.com/watch?v=fAuF0EuZVCk&t=261s&ab_channel=TusharRoy-CodingMadeSimple
- [Practice Leetcode Problem](https://leetcode.com/problems/min-cost-to-connect-all-points)

#### Edge List

```Python
class UnionFind:
    def __init__(self, n):
        self.id = list(range(n))
        self.sz = [1] * n

    def find(self, p):
        root = p
        while root != self.id[root]:
            root = self.id[root]
        while p != root:  # Path compression
            next_p = self.id[p]
            self.id[p] = root
            p = next_p
        return root

    def connected(self, p, q):
        return self.find(p) == self.find(q)

    def size(self, p):
        return self.sz[self.find(p)]

    def union(self, p, q):
        root1 = self.find(p)
        root2 = self.find(q)
        if root1 == root2:
            return
        if self.sz[root1] < self.sz[root2]:
            self.sz[root2] += self.sz[root1]
            self.id[root1] = root2
        else:
            self.sz[root1] += self.sz[root2]
            self.id[root2] = root1

class Edge:
    def __init__(self, from_node, to_node, cost):
        self.from_node = from_node
        self.to_node = to_node
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost

def kruskals(edges, n):
    if edges is None:
        return None

    edges.sort()  # Sort edges by cost
    uf = UnionFind(n)
    sum_cost = 0

    for edge in edges:
        if not uf.connected(edge.from_node, edge.to_node):
            uf.union(edge.from_node, edge.to_node)
            sum_cost += edge.cost
            if uf.size(0) == n:  # Optimization to stop early
                break

    # Ensure all nodes are included in the MST
    if uf.size(0) != n:
        return None

    return sum_cost

# Example usage:
edges = [
    Edge(0, 1, 10),
    Edge(0, 2, 6),
    Edge(0, 3, 5),
    Edge(1, 3, 15),
    Edge(2, 3, 4)
]
n = 4  # Number of nodes in the graph
print(kruskals(edges, n))  # Output: 19 (the total cost of the MST)
```

### Union Find

- Union Find / Disjoint Set data structure.
- Videos:
    - [Introduction](https://www.youtube.com/watch?v=ibjEGG7ylHk)
    - [Union and Find Operations](https://www.youtube.com/watch?v=0jNmHPfA_yE)
    - [Union Find Kruskal's Algorithm](https://www.youtube.com/watch?v=JZBQLXgSGfs)
    
- [Source Code](https://github.com/williamfiset/Algorithms/blob/da50861a53fc2f6642cfc7d82c285166f41d03e2/src/main/java/com/williamfiset/algorithms/datastructures/unionfind/UnionFind.java)

```Python
class UnionFind:
    def __init__(self, size):
        if size <= 0:
            raise ValueError("Size <= 0 is not allowed")
        
        # The number of elements in this union find
        self.size = size
        
        # Used to track the size of each component
        self.sz = [1] * size
        
        # id[i] points to the parent of i, if id[i] = i then i is a root node
        self.id = list(range(size))
        
        # Tracks the number of components in the union find
        self.num_components = size

    # Find which component/set 'p' belongs to, takes amortized constant time.
    def find(self, p):
        # Find the root of the component/set
        root = p
        while root != self.id[root]:
            root = self.id[root]
        
        # Compress the path leading back to the root.
        # Doing this operation is called "path compression"
        # and is what gives us amortized time complexity.
        while p != root:
            next_p = self.id[p]
            self.id[p] = root
            p = next_p
        return root

    # This is an alternative recursive formulation for the find method
    # def find(self, p):
    #     if p == self.id[p]:
    #         return p
    #     self.id[p] = self.find(self.id[p])
    #     return self.id[p]

    # Return whether or not the elements 'p' and 'q' are in the same components/set.
    def connected(self, p, q):
        return self.find(p) == self.find(q)

    # Return the size of the component/set 'p' belongs to
    def component_size(self, p):
        return self.sz[self.find(p)]

    # Return the number of elements in this UnionFind/Disjoint set
    def size(self):
        return self.size

    # Returns the number of remaining components/sets
    def components(self):
        return self.num_components

    # Unify the components/sets containing elements 'p' and 'q'
    def unify(self, p, q):
        # These elements are already in the same group!
        if self.connected(p, q):
            return

        root1 = self.find(p)
        root2 = self.find(q)

        # Merge smaller component/set into the larger one.
        if self.sz[root1] < self.sz[root2]:
            self.sz[root2] += self.sz[root1]
            self.id[root1] = root2
            self.sz[root1] = 0
        else:
            self.sz[root1] += self.sz[root2]
            self.id[root2] = root1
            self.sz[root2] = 0

        # Since the roots found are different we know that the
        # number of components/sets has decreased by one
        self.num_components -= 1

# Example usage:
uf = UnionFind(10)
uf.unify(1, 2)
uf.unify(2, 3)
print(uf.connected(1, 3))  # Output: True
print(uf.component_size(1))  # Output: 3
print(uf.components())  # Output: 8
```

## Travelling Salesman

### Brute Force - O(N^3)

```Python
import itertools
import math

def tsp(matrix):
    n = len(matrix)
    permutation = list(range(n))
    best_tour = permutation[:]
    best_tour_cost = float('inf')

    # Try all n! tours
    for perm in itertools.permutations(permutation):
        tour_cost = compute_tour_cost(perm, matrix)

        if tour_cost < best_tour_cost:
            best_tour_cost = tour_cost
            best_tour = list(perm)

    return best_tour

def compute_tour_cost(tour, matrix):
    cost = 0

    # Compute the cost of going to each city
    for i in range(1, len(matrix)):
        from_city = tour[i - 1]
        to_city = tour[i]
        cost += matrix[from_city][to_city]

    # Compute the cost to return to the starting city
    last_city = tour[-1]
    first_city = tour[0]
    return cost + matrix[last_city][first_city]

def main():
    n = 10
    matrix = [[100] * n for _ in range(n)]

    # Construct an optimal tour
    edge_cost = 5
    optimal_tour = [2, 7, 6, 1, 9, 8, 5, 3, 4, 0, 2]
    for i in range(1, len(optimal_tour)):
        matrix[optimal_tour[i - 1]][optimal_tour[i]] = edge_cost

    best_tour = tsp(matrix)
    print("Best tour:", best_tour)

    tour_cost = compute_tour_cost(best_tour, matrix)
    print("Tour cost:", tour_cost)

if __name__ == "__main__":
    main()
```

### Dynamic Programming

<details>

<summary>Show iterative</summary>

```Python
import itertools
import collections

class TspDynamicProgrammingIterative:
    def __init__(self, start, distance):
        self.N = len(distance)
        self.start = start
        self.distance = distance
        self.minTourCost = float('inf')
        self.tour = []
        self.ranSolver = False

        if self.N <= 2:
            raise ValueError("N <= 2 not yet supported.")
        if self.N != len(distance[0]):
            raise ValueError("Matrix must be square (n x n)")
        if start < 0 or start >= self.N:
            raise ValueError("Invalid start node.")
        if self.N > 32:
            raise ValueError(
                "Matrix too large! A matrix that size for the DP TSP problem with a time complexity of"
                + " O(n^2*2^n) requires way too much computation for any modern home computer to handle")

    def get_tour(self):
        if not self.ranSolver:
            self.solve()
        return self.tour

    def get_tour_cost(self):
        if not self.ranSolver:
            self.solve()
        return self.minTourCost

    def solve(self):
        if self.ranSolver:
            return

        END_STATE = (1 << self.N) - 1
        memo = [[None] * (1 << self.N) for _ in range(self.N)]

        for end in range(self.N):
            if end == self.start:
                continue
            memo[end][(1 << self.start) | (1 << end)] = self.distance[self.start][end]

        for r in range(3, self.N + 1):
            for subset in self.combinations(r, self.N):
                if not self.not_in(self.start, subset):
                    for next in range(self.N):
                        if next == self.start or self.not_in(next, subset):
                            continue
                        subset_without_next = subset ^ (1 << next)
                        min_dist = float('inf')
                        for end in range(self.N):
                            if end == self.start or end == next or self.not_in(end, subset):
                                continue
                            new_distance = memo[end][subset_without_next] + self.distance[end][next]
                            if new_distance < min_dist:
                                min_dist = new_distance
                        memo[next][subset] = min_dist

        for i in range(self.N):
            if i == self.start:
                continue
            tour_cost = memo[i][END_STATE] + self.distance[i][self.start]
            if tour_cost < self.minTourCost:
                self.minTourCost = tour_cost

        last_index = self.start
        state = END_STATE
        self.tour.append(self.start)

        for i in range(1, self.N):
            best_index = -1
            best_dist = float('inf')
            for j in range(self.N):
                if j == self.start or self.not_in(j, state):
                    continue
                new_dist = memo[j][state] + self.distance[j][last_index]
                if new_dist < best_dist:
                    best_index = j
                    best_dist = new_dist

            self.tour.append(best_index)
            state = state ^ (1 << best_index)
            last_index = best_index

        self.tour.append(self.start)
        self.tour.reverse()

        self.ranSolver = True

    def not_in(self, elem, subset):
        return ((1 << elem) & subset) == 0

    def combinations(self, r, n):
        subsets = []
        self._combinations(0, 0, r, n, subsets)
        return subsets

    def _combinations(self, set, at, r, n, subsets):
        elements_left_to_pick = n - at
        if elements_left_to_pick < r:
            return

        if r == 0:
            subsets.append(set)
        else:
            for i in range(at, n):
                set ^= (1 << i)
                self._combinations(set, i + 1, r - 1, n, subsets)
                set ^= (1 << i)


def main():
    # Create adjacency matrix
    n = 6
    distance_matrix = [[10000] * n for _ in range(n)]
    distance_matrix[5][0] = 10
    distance_matrix[1][5] = 12
    distance_matrix[4][1] = 2
    distance_matrix[2][4] = 4
    distance_matrix[3][2] = 6
    distance_matrix[0][3] = 8

    start_node = 0
    solver = TspDynamicProgrammingIterative(start_node, distance_matrix)

    # Prints: [0, 3, 2, 4, 1, 5, 0]
    print("Tour: ", solver.get_tour())

    # Print: 42.0
    print("Tour cost: ", solver.get_tour_cost())


if __name__ == "__main__":
    main()
```

</details>

<details>

<summary>Show recursive</summary>

```Python
"""
Recursive implementation of the TSP problem using dynamic programming. The
main idea is that since we need to do all n! permutations of nodes to find the optimal solution
that caching the results of sub paths can improve performance.

For example, if one permutation is: '... D A B C' then later when we need to compute the value
of the permutation '... E B A C' we should already have cached the answer for the subgraph
containing the nodes {A, B, C}.

Time Complexity: O(n^2 * 2^n) Space Complexity: O(n * 2^n)
"""
import sys

class TspDynamicProgrammingRecursive:
    def __init__(self, start_node, distance):
        self.distance = distance
        self.N = len(distance)
        self.START_NODE = start_node
        self.FINISHED_STATE = (1 << self.N) - 1
        self.minTourCost = float('inf')
        self.tour = []
        self.ranSolver = False

        # Validate inputs
        if self.N <= 2:
            raise ValueError("TSP on 0, 1 or 2 nodes doesn't make sense.")
        if self.N != len(distance[0]):
            raise ValueError("Matrix must be square (N x N)")
        if self.START_NODE < 0 or self.START_NODE >= self.N:
            raise ValueError("Starting node must be: 0 <= startNode < N")
        if self.N > 32:
            raise ValueError(
                "Matrix too large! A matrix that size for the DP TSP problem with a time complexity of"
                + " O(n^2*2^n) requires way too much computation for any modern home computer to handle")

    def get_tour(self):
        if not self.ranSolver:
            self.solve()
        return self.tour

    def get_tour_cost(self):
        if not self.ranSolver:
            self.solve()
        return self.minTourCost

    def solve(self):
        state = 1 << self.START_NODE
        memo = [[None] * (1 << self.N) for _ in range(self.N)]
        prev = [[None] * (1 << self.N) for _ in range(self.N)]
        self.minTourCost = self.tsp(self.START_NODE, state, memo, prev)

        # Regenerate path
        index = self.START_NODE
        while True:
            self.tour.append(index)
            next_index = prev[index][state]
            if next_index is None:
                break
            state |= 1 << next_index
            index = next_index
        self.tour.append(self.START_NODE)
        self.ranSolver = True

    def tsp(self, i, state, memo, prev):
        if state == self.FINISHED_STATE:
            return self.distance[i][self.START_NODE]

        if memo[i][state] is not None:
            return memo[i][state]

        min_cost = float('inf')
        index = -1
        for next in range(self.N):
            if state & (1 << next):
                continue
            next_state = state | (1 << next)
            new_cost = self.distance[i][next] + self.tsp(next, next_state, memo, prev)
            if new_cost < min_cost:
                min_cost = new_cost
                index = next

        prev[i][state] = index
        memo[i][state] = min_cost
        return min_cost

def main():
    n = 6
    distance_matrix = [[10000] * n for _ in range(n)]
    distance_matrix[1][4] = distance_matrix[4][1] = 2
    distance_matrix[4][2] = distance_matrix[2][4] = 4
    distance_matrix[2][3] = distance_matrix[3][2] = 6
    distance_matrix[3][0] = distance_matrix[0][3] = 8
    distance_matrix[0][5] = distance_matrix[5][0] = 10
    distance_matrix[5][1] = distance_matrix[1][5] = 12

    start_node = 0
    solver = TspDynamicProgrammingRecursive(start_node, distance_matrix)

    print("Tour: ", solver.get_tour())
    print("Tour cost: ", solver.get_tour_cost())

if __name__ == "__main__":
    main()

```

</details>

## Find Articulation Points

<details>

<summary>Click to view code</summary>


```Python
from typing import List

class ArticulationPointsAdjacencyList:
    def __init__(self, graph: List[List[int]], n: int):
        if graph is None or n <= 0 or len(graph) != n:
            raise ValueError("Invalid graph or number of nodes.")
        self.graph = graph
        self.n = n
        self.solved = False
        self.id = 0
        self.rootNodeOutcomingEdgeCount = 0
        self.low = [0] * n
        self.ids = [0] * n
        self.visited = [False] * n
        self.isArticulationPoint = [False] * n

    def find_articulation_points(self) -> List[bool]:
        if self.solved:
            return self.isArticulationPoint

        for i in range(self.n):
            if not self.visited[i]:
                self.rootNodeOutcomingEdgeCount = 0
                self.dfs(i, i, -1)
                self.isArticulationPoint[i] = (self.rootNodeOutcomingEdgeCount > 1)

        self.solved = True
        return self.isArticulationPoint

    def dfs(self, root: int, at: int, parent: int):
        if parent == root:
            self.rootNodeOutcomingEdgeCount += 1

        self.visited[at] = True
        self.low[at] = self.ids[at] = self.id
        self.id += 1

        for to in self.graph[at]:
            if to == parent:
                continue
            if not self.visited[to]:
                self.dfs(root, to, at)
                self.low[at] = min(self.low[at], self.low[to])
                if self.ids[at] <= self.low[to]:
                    self.isArticulationPoint[at] = True
            else:
                self.low[at] = min(self.low[at], self.ids[to])

def create_graph(n: int) -> List[List[int]]:
    return [[] for _ in range(n)]

def add_edge(graph: List[List[int]], frm: int, to: int):
    graph[frm].append(to)
    graph[to].append(frm)

def main():
    test_example2()

def test_example1():
    n = 9
    graph = create_graph(n)

    add_edge(graph, 0, 1)
    add_edge(graph, 0, 2)
    add_edge(graph, 1, 2)
    add_edge(graph, 2, 3)
    add_edge(graph, 3, 4)
    add_edge(graph, 2, 5)
    add_edge(graph, 5, 6)
    add_edge(graph, 6, 7)
    add_edge(graph, 7, 8)
    add_edge(graph, 8, 5)

    solver = ArticulationPointsAdjacencyList(graph, n)
    is_articulation_point = solver.find_articulation_points()

    for i in range(n):
        if is_articulation_point[i]:
            print(f"Node {i} is an articulation")

def test_example2():
    n = 3
    graph = create_graph(n)

    add_edge(graph, 0, 1)
    add_edge(graph, 1, 2)

    solver = ArticulationPointsAdjacencyList(graph, n)
    is_articulation_point = solver.find_articulation_points()

    for i in range(n):
        if is_articulation_point[i]:
            print(f"Node {i} is an articulation")

if __name__ == "__main__":
    main()
```

</details>

- Checking for a cycle in a graph:
    - __Directed__: Use color array with values [0,1,2].
    - __Undirected__:
        1. Use Union find.
        2. Use parent pointer and visited array (DFS or BFS).
