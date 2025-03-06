from typing import List

def solve(M: int, N: int, K: int, s: List[str]) -> None:
    ns = [[""]*(N*K) for _ in range(M*K)]
    for i in range(M):
        for j in range(N):
            for k in range(K):
                ns[i+k][j*K:j*K+K] = [s[i][j]]*K

    with open("cowsignal.out", "w") as f:
        for i in range(M*K):
            f.write("".join(ns[i]))


M, N, K = map(int, input().split())
s = [input() for _ in range(M)]
solve(M, N, K, s)


