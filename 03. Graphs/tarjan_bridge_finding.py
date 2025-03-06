"""
Articulation Points
===================

Let's say there is a node 𝑉 in some graph 𝐺 that can be reached by a node 𝑈 through some intermediate
nodes (maybe non intermediate nodes) following some DFS traversal.

If 𝑉 can also be reached by 𝐴 = "ancestor of 𝑈" without passing through 𝑈, then
𝑈 is NOT an articulation point because it means that if we remove 𝑈 from 𝐺 we
can still reach 𝑉 from 𝐴, hence, the number of connected components will remain the same.

So, we can conclude that the only 2 conditions for 𝑈 to be an articulation point are:

    1. If all paths from 𝐴 to 𝑉 require 𝑈 to be in the graph.
    2. If 𝑈 is the root of the DFS traversal with at least 2 children subgraphs disconnected from each other.

    We can break condition 1 into 2 subconditions:
        1.1. 𝑈 is an articulation point if it does not have an adjacent node 𝑉 that can reach 𝐴 without requiring 𝑈 to be in 𝐺.
        1.2. 𝑈 is an articulation point if it is the root of some cycle in the DFS traversal.


Articulation Points and Cycle Examples
======================================

Example 1: Simple Chain with an Articulation Point
--------------------------------------------------
    A
    |
    B
    |
    C

Description:
- B is an articulation point because all paths from A (ancestor) to C require passing through B.
- Removing B disconnects the graph into two components.


Example 2: Cycle with No Articulation Point
-------------------------------------------
      A
      |
      B
     / \
    C---D

Description:
- B is not an articulation point.
- There is an alternative path (C to D) that does not go through B.


Example 3: Simple Cycle with an Articulation Point
--------------------------------------------------
    A
    |
    B
   / \
  C---D

Description:
- B is an articulation point.
- Removing B disconnects the graph, isolating C and D from A.


Example 4: Complex Cycle with Multiple Articulation Points
----------------------------------------------------------
       A
      / \
     B   C
    / \ / \
   D   E   F
    \ / \ /
     G---H

Description:
- B and C are articulation points.
- Removing B disconnects D and E from the rest of the graph.
- Removing C disconnects F and E from the rest of the graph.
- G and H are not articulation points because there are multiple paths between vertices.


Example 5: Linear Chain with Multiple Articulation Points
---------------------------------------------------------
    A — B — C — D — E

Description:
- Each vertex in the middle of the chain (B, C, and D) is an articulation point.
- Removing any of these vertices disconnects the graph into two components.


Example 6: Fully Connected Graph (No Articulation Points)
---------------------------------------------------------

    A — B — C
     \  |  /
        D

Description:
- No articulation points because removing any vertex still leaves all others connected.

Resources
=========

https://codeforces.com/blog/entry/71146
https://takeuforward.org/graph/bridges-in-graph-using-tarjans-algorithm-of-time-in-and-low-time-g-55/
"""
def bridges(n: int, edges: List[List[int]]) -> List[List[int]]:
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    in_time = [-1] * n   # Discovery times
    low = [-1] * n       # Low-link values
    bridges = []         # Store bridges (u,v)
    timer = 0

    def dfs(u, parent):
        nonlocal timer
        in_time[u] = low[u] = timer
        timer += 1

        for v in adj[u]:
            if v == parent:  # Skip the edge to parent
                continue

            if in_time[v] == -1:  # If v is unvisited
                dfs(v, u)         # Recur for the child node

                # Update low-link value after returning from DFS call
                low[u] = min(low[u], low[v])

                # Check if the edge (u, v) is a bridge
                if low[v] > in_time[u]:
                    bridges.append((u, v))
            else:
                # Update low-link value based on back edge
                low[u] = min(low[u], in_time[v])

    for u in range(n):
        dfs(0, -1)

    return bridges



