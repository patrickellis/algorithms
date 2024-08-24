"""In-place sorting algorithm that produces the MINIMAL number of writes to the original array.

https://en.wikipedia.org/wiki/Cycle_sort


  - Cycle Sort offers the advantage of lIttle or no additional storage.
  - It is an in-place sorting Algorithm.
  - It is optimal in terms of number of memory writes.
      It makes minimum number of writes to the memory and hence efficient when
      the array is stored in e.g. Flash memory, where every write reduces the
      lifespan of the memory.
  - Unlike nearly every other sort (Quick, insertion , merge sort), items
      are never written elsewhere in the array simply to push them out of the
      way.
  - Each value is either written:
        - Zero (0) times, if it's already in it's correct position.
        - One (1) time TO its correct position.
"""
import random
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def cycle_sort(A: list[int]) -> None:
    """Sorts an array in place.

    For each item `x` in the array:

        1. Count the number of items that are less than it. This is the
           final index of the item in the sorted array.
        2. If the element is already in the correct position, continue.
        3. Otherwise, write it to its correct position.
           That position is inhabited by another element, which we then have to
           move to _its_ correct position.
        4. We keep moving elements to their correct positions until an element is moved to the original position of `x`.
           This completes a cycle.

    The item has been placed in the correct position when `pos` is equal to `i`.
    Until then, repeat the process of counting the number of items less than
    the current item, and placing it right after duplicates.

    N.B. Items are placed right after duplicates, if any.
    """
    n = len(A)

    for i in range(n-1):
        item = A[i]
        pos = i

        for j in range(i+1, n):
            if A[j] < item:
                pos += 1

        if pos == i:
            continue

        while item == A[pos]:
            pos += 1

        A[pos], item = item, A[pos]

        while pos != i:
            pos = i
            for j in range(i+1, n):
                if A[j] < item:
                    pos += 1
            while item == A[pos]:
                pos += 1
            A[pos], item = item, A[pos]


if __name__ == "__main__":
    A = [random.randint(1, 100) for _ in range(15)]
    logger.info("Original Array: %s", A)
    cycle_sort(A)
    logger.info("Sorted Array: %s", A)
