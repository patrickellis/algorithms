def generate_subsets(A):
    result = []
    for i in range(1 << len(A)):  # Loop over 0 to 2^n - 1
        subset = []
        for j in range(len(A)):
            if i & (1 << j):  # Check if jth bit is set
                subset.append(A[j])
        result.append(subset)
    return result


A = [1, 2, 3]
subsets = generate_subsets(A)
print(subsets)
