from typing import List, Tuple

def solve(N: int, M: int, R: List[Tuple[int,int]], C: List[Tuple[int,int]]) -> None:
    sim = [0] * 100
    rd = cd = 0
    rs = cs = -1
    ri = ci = 0
    for segment in range(100):
        if ri < len(R) and segment >= rd:
            dist, rs = R[ri]
            rd += dist
            ri += 1
        if ci < len(C) and segment >= cd:
            dist, cs = C[ci]
            cd += dist
            ci += 1
        sim[segment] = max(0, cs-rs)
    result = max(sim)

    with open("speeding.out","w") as f:
        f.write(str(result))

with open("speeding.in", "r") as f:
    N, M = map(int, f.readline().split())
    rs = [tuple(map(int,f.readline().split())) for _ in range(N)]
    bs = [tuple(map(int,f.readline().split())) for _ in range(M)]
solve(N, M, rs, bs)
