import random
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def mergeSort(A):
    if len(A) <= 1:
        return A
    mp = len(A) // 2
    L = A[:mp]
    R = A[mp:]

    mergeSort(L)
    mergeSort(R)
    i = j = k = 0

    while i < len(L) and j < len(R):
        if L[i] < R[j]:
            A[k] = L[i]
            i += 1
        else:
            A[k] = R[j]
            j += 1
        k += 1

    print("L: ", L[i:])
    print("R: ", R[j:])
    A[k:] = L[i:] + R[j:]
    print("A: ", A)


if __name__ == "__main__":
    UPPER_LIM = 100
    for _ in range(5):
        A = [random.randint(1, UPPER_LIM) for _ in range(15)]
        copy = A.copy()
        logger.info("Original Array: %s", A)
        mergeSort(A)
        assert A == sorted(copy)
        logger.info("Sorted Array: %s", A)
