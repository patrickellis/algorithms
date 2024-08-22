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

    1. Loop through each item in the array to find cycles to rotate.
    2. In each iteration, select an item from the array and count the number
       of items that are less than it.
    3. Place the item right after duplicates, if any.
    4. The item has been placed in the correct position when `pos` is equal to `i`.
        Until then, repeat the process of counting the number of items less than
        the current item, and placing it right after duplicates.
    """
    n = len(A)

    for i in range(0, n-1):
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
