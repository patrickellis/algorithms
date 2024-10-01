"""Generate Permutations using Heap's algorithm.

https://www.geeksforgeeks.org/heaps-algorithm-for-generating-permutations/
"""
permutations = []


def heapPermutation(A, size):
    if size == 1:
        permutations.append(A[:])
        return

    for i in range(size):
        heapPermutation(A, size - 1)

        # if size is odd, swap 0th i.e (first)
        # and (size-1)th i.e (last) element
        # else If size is even, swap ith
        # and (size-1)th i.e (last) element
        if size & 1:
            A[0], A[size - 1] = A[size - 1], A[0]
        else:
            A[i], A[size - 1] = A[size - 1], A[i]


A = [i + 1 for i in range(4)]
heapPermutation(A, len(A))
print(f"Found {len(permutations)} Permutations of {A}:")
for result in permutations:
    print(result)
