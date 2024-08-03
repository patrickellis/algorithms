def quicksort(A: list, left: int, right: int) -> None:
    if left >= right:
        return

    pivot = A[left]
    low = left
    mid = left
    high = right

    while mid <= high:
        if A[mid] < pivot:
            A[low], A[mid] = A[mid], A[low]
            low += 1
            mid += 1
        elif A[mid] > pivot:
            A[high], A[mid] = A[mid], A[high]
            high -= 1
        else:
            mid += 1

    quicksort(A, left, low - 1)
    quicksort(A, high + 1, right)


A = [3, 5, 8, 4, 1, 9, -2]
print(f"Original: {A}")
quicksort(A, 0, len(A) - 1)
print(f"Sorted: {A}")
