"""Levenstein distance."""

def solve(s1: str, s2: str):
    m, n = len(s1), len(s2)
    dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for i in range(n + 1):
        dp[0][i] = i

    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1

    return dp[m][n]

s1 = "zxyz"
s2 = "zyxz"

print(f"{s1} -> {s2} requires {solve(s1,s2)} operations.")

