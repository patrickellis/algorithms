import random
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def bubble_sort(A: list[int]):
    n = len(A)
    while n > 1:
        next_n = 0
        for i in range(1,n):
            if A[i-1] > A[i]:
                A[i-1], A[i] = A[i], A[i-1]
                next_n = i
        n = next_n


if __name__ == "__main__":
    UPPER_LIM = 100
    A = [random.randint(1, UPPER_LIM) for _ in range(15)]
    logger.info("Original Array: %s", A)
    bubble_sort(A)
    logger.info("Sorted Array: %s", A)
