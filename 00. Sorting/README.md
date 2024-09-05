# Sorting

## Heap Sort

### Pros

- Faster than Bubble Sort, Insertion Sort, and Selection Sort.

### Cons

- Not stable.
- Performs worse than other $\text{O}(N\log{N})$ sorting algorithms
  due to poor cache locality.
- Swaps elements based on their location in heap. This can cause many cache
  misses.

```Python
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

    def sort_heap() -> None:
        for i in range(len(A) - 1, 0, -1):
            A[i], A[0] = A[0], A[i]
            max_heapify(i, 0)

    build_max_heap()
    sort_heap()
```

## Bubble Sort

### Optimizations

- After every pass, all elements after the last swap are sorted, and do not
  need to be checked again. This results in about a worst case 50% improvement
  in comparison count.

```Python
def bubble_sort(A: list[int]):
    n = len(A)
    while n > 1:
        next_n = 0
        for j in range(n-1):
            if A[j] > A[j+1]:
                A[j], A[j+1] = A[j+1], A[j]
                next_n = j
        n = next_n
```

## Insertion Sort

```Python
def insertion_sort(A: list[int]):
    n = len(A)
    for i in range(1,n):
        j = i
        while j > 0 and A[j] < A[j-1]:
            A[j], A[j-1] = A[j-1], A[j]
            j -= 1
```

## Selection Sort

```Python
def selection_sort(A: list[int]):
    n = len(A)
    for i in range(n-1):
        jMin = i
        for j in range(i+1,n):
            if A[j] < A[jMin]:
                jMin = j
        A[i], A[jMin] = A[jMin], A[i]
```
