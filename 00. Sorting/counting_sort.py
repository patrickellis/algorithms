"""Sorts collections of objects with keys that are small positive integers.

https://en.wikipedia.org/wiki/Counting_sort

- Counting sort is NOT a comparison sort.
- Counting sort is a LINEAR TIME sort.

Worst-case time complexity: O(N+K)
Worst-case space complexity: O(N+K)
   -K is the range of the non-negative key values.
   -N is the number of elements in the input array.

Two Versions
============

    1) In-place, NOT stable.

        Places the items into sorted order within the same array that was given to it
        as the input, using only the count array as auxiliary storage.
        The modified in-place version of counting sort is not stable.

    2) Stable.

        Places the items into sorted order in a separate output array.

"""
import random
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def counting_sort(A: list[int], k: int) -> list[int]:
    """Sorts an array using Counting Sort and returns the sorted array.

    1. Initialize the count array with zeros.
    2. Count occurrences of each value in the input array.
    3. Update the count array to accumulate the count of elements up to each index.
    4. Place the elements in the output array in sorted order.
    """
    n = len(A)
    count = [0]*(k+1)
    output = [0]*n

    for i in range(n):
        count[A[i]] += 1

    for i in range(1, k+1):
        count[i] += count[i-1]

    for i in range(n-1,-1,-1):
        count[A[i]] -= 1
        output[count[A[i]]] = A[i]

    return output


if __name__ == "__main__":
    UPPER_LIM = 100
    A = [random.randint(1, UPPER_LIM) for _ in range(15)]
    logger.info("Original Array: %s", A)
    A = counting_sort(A, UPPER_LIM)
    logger.info("Sorted Array: %s", A)
