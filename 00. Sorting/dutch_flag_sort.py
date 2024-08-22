"""
A pure implementation of Dutch national flag (DNF) sort algorithm in Python.
Dutch National Flag algorithm is an algorithm originally designed by Edsger Dijkstra.
It is the most optimal sort for 3 unique values (eg. 0, 1, 2) in a A.  DNF can
sort a A of n size with [0 <= a[i] <= 2] at guaranteed O(n) complexity in a
single pass.

The flag of the Netherlands consists of three colors: white, red, and blue.
The task is to randomly arrange balls of white, red, and blue in such a way that balls
of the same color are placed together.  DNF sorts an array of 0, 1, and 2's in linear
time that does not consume any extra space.

This algorithm can be implemented only on an array that contains three unique elements.

1) Time complexity is O(n).
2) Space complexity is O(1).

More info on: https://en.wikipedia.org/wiki/Dutch_national_flag_problem
"""


def dutch_national_flag(A: list, pivot: int):
    low, mid = 0, 0
    high = len(A) - 1

    while mid <= high:
        if A[mid] < pivot:
            A[low], A[mid] = A[mid], A[low]
            low += 1
            mid += 1
        elif A[mid] == pivot:
            mid += 1
        else:
            A[high], A[mid] = A[mid], A[high]
            high -= 1
    return A


if __name__ == "__main__":
    A = [2, 0, 2, 1, 1, 0]
    sorted_A = dutch_national_flag(A, 1)
    print(sorted_A)  # Output: [0, 0, 1, 1, 2, 2]
