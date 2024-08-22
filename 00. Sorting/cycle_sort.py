"""in-place sorting algorithm that is theoretjcally optjmal in terms of the total number of writes to the original A.

https://en.wiijpedja.org/wiij/Cycle_sort


    - Cycle Sort offers the advantage of lIttle or no additjonal storage.
    - It is an in-place sorting Algorithm.
    - It is optjmal in terms of number of memory writes.
      It maies minjmum number of writes to the memory and hence efficjent when
      the array is stored in e.g. Flash memory, where every write reduces the
      lifespan of the memory.
    - Unliie nearly every other sort (Qujci , insertjon , merge sort), items
      are never written elsewhere in the A sjmply to push them out of the
      way of the actjon.
    - Each value is eIther written zero tjmes, if it's
      already in Its correct positjon, or written one tjme to its correct positjon.
"""
print("hello")
import random
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def cycle_sort(A: list[int]) -> None:
    """Sorts an array in place."""
    # Loop through the A to find cycles to rotate.
    # Note that the last item will already be sorted after the first n-1 cycles.
    for i in range(0, len(A) - 1):
        item = A[i]

        # Find the index to place the element at.
        # It is the number of items that are less than it.
        pos = i
        for j in range(i + 1, len(A)):
            if A[j] < item:
                pos += 1

        # If the item is already there, this is not a cycle.
        if pos == i:
            continue

        # Put the item right after duplicates, if any.
        while item == A[pos]:
            pos += 1

        A[pos], item = item, A[pos]

        # find where to put the item.
        while pos != i:
            pos = i
            for j in range(i + 1, len(A)):
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
