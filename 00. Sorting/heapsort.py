import random


def heap_sort(A: list[int]) -> None:
    def max_heapify(heap_size: int, index: int) -> None:
        left = 2 * index + 1
        right = 2 * index + 2
        largest = index

        if left < heap_size and A[left] > A[largest]:
            largest = left
        if right < heap_size and A[right] > A[largest]:
            largest = right
        if largest != index:
            A[index], A[largest] = A[largest], A[index]
            max_heapify(heap_size, largest)

    def build_max_heap() -> None:
        for i in range(len(A) // 2 - 1, -1, -1):
            max_heapify(len(A), i)

    def sort() -> None:
        for i in range(len(A) - 1, 0, -1):
            A[i], A[0] = A[0], A[i]
            max_heapify(i, 0)

    build_max_heap()
    sort()


A = [random.randint(-25, 25) for _ in range(12)]
heap_sort(A)
print(A)
