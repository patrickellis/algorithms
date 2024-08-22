import random
from collections import deque

N = random.randint(4, 8)
m = [
    [int(random.randint(0, 1)) for _ in range(N)] for _ in range(N)
]  # Input matrix of size R x C
R, C = len(m), len(m[0])
directions = ((0, 1), (0, -1), (1, 0), (-1, 0))
queue = deque()
nodes_left_in_layer = 0
nodes_in_edge_layer = 1
move_count = 0
visited = set()


def explore_neighbours(v: tuple[int, int]):
    global nodes_in_edge_layer, queue
    r, c = v
    for d in directions:
        rr = r + d[0]
        cc = c + d[1]

        if rr < 0 or cc < 0:
            continue
        if rr >= R or cc >= C:
            continue

        if (rr, cc) in visited:
            continue
        if m[rr][cc] == "#":
            continue  # Problem specific definition of 'Blocked' cell

        nodes_in_edge_layer += 1
        queue.append((rr, cc))


def solve(s: tuple[int, int], e: tuple[int, int]) -> int:
    """Returns the move count if a path to the end was found, else -1."""
    global nodes_left_in_layer, nodes_in_edge_layer, move_count, visited, queue
    queue.append(s)
    nodes_left_in_layer = 1
    while queue:
        v = queue.popleft()
        visited.add(v)

        if v == e:
            return move_count

        explore_neighbours(v)
        nodes_left_in_layer -= 1

        if nodes_left_in_layer == 0:
            nodes_left_in_layer = nodes_in_edge_layer
            nodes_in_edge_layer = 0
            move_count += 1

    return -1


def bfs(s: tuple[int, int], e: tuple[int, int]) -> int:
    return solve(s, e)


res = bfs((0, 0), (2, 0))
for i in range(N):
    print(" ".join(map(str, m[i])))

print("")
print(f"Moves: {res}")
