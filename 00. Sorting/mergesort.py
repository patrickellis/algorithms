
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

    while i < len(L):
        A[k] = L[i]
        i += 1
        k += 1
    while j < len(R):
        A[k] = R[j]
        j += 1
        k += 1

if __name__ == "__main__":
    UPPER_LIM = 100
    A = [random.randint(1, UPPER_LIM) for _ in range(15)]
    logger.info("Original Array: %s", A)
    mergeSort(A)
    logger.info("Sorted Array: %s", A)
