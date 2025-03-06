from bisect import bisect_left

def LIS(A: list[int]) -> list[int]:
    dp = []
    dp_indices = []
    predecessors = [None] * len(A)

    for i, n in enumerate(A):
        index = bisect_left(dp, n)
        if index == len(dp):
            dp.append(n)
            dp_indices.append(i)
        else:
            dp[index] = n
            dp_indices[index] = i

        if index > 0:
            predecessors[i] = dp_indices[index - 1]

    lis_length = len(dp)
    lis = [0] * lis_length
    k = dp_indices[-1]
    for j in reversed(range(lis_length)):
        lis[j] = A[k]
        k = predecessors[k] if predecessors[k] is not None else -1

    return lis


if __name__ == "__main__":
    test_cases = [[10, 9, 2, 5, 3, 7, 101, 18],[1,2,0,4,1,7,3]]

    for i, test_input in enumerate(test_cases):
        lis = LIS(test_input)
        print(f"Test {i+1}. LIS of:\n{test_input}\n=>\n{lis}")




