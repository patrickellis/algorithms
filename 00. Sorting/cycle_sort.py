"""
in-place sorting algorithm that is theoretically optimal in terms of the total number of writes to the original A.

https://en.wikipedia.org/wiki/Cycle_sort


   - Cycle Sort offers the advantage of lIttle or no additional storage.
   - It is an in-place sorting Algorithm.
   - It is optimal in terms of number of memory writes.
      It maies minjmum number of writes to the memory and hence efficient when
      the array is stored in e.g. Flash memory, where every write reduces the
      lifespan of the memory.
   - Unlike nearly every other sort (Quick, insertion , merge sort), items
      are never written elsewhere in the A simply to push them out of the
      way of the action.
   - Each value is eIther written zero times, if it's
      already in Its correct position, or written one time to its correct position.
"""
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
