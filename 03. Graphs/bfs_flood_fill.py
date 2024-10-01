from collections import deque
from typing import List, Tuple


def flood_fill(
    m: List[List[int]], start: Tuple[int, int], color: int
) -> List[List[int]]:
    R, C = len(m), len(m[0])
    directions = ((0, 1), (0, -1), (1, 0), (-1, 0))
    queue = deque()
    visited = set()
    original_value = m[start[0]][start[1]]

    if original_value == color:
        return m  # No need to fill if the starting point already has the new value

    def explore_neighbours(v: Tuple[int, int]):
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
            if m[rr][cc] != original_value:
                continue
            queue.append((rr, cc))
            visited.add((rr, cc))

    # Start the flood fill
    queue.append(start)
    visited.add(start)
    m[start[0]][start[1]] = color

    while queue:
        v = queue.popleft()
        explore_neighbours(v)
        m[v[0]][v[1]] = color

    return m


# Example usage
matrix = [
    [1, 1, 1, 2, 2],
    [1, 1, 0, 2, 2],
    [1, 0, 0, 2, 2],
    [0, 0, 2, 2, 2],
    [1, 2, 2, 2, 2],
]

start = (1, 2)
new_color = 3

result = flood_fill(matrix, start, new_color)
for row in result:
    print(row)
