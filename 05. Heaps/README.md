# Trees

## Glossary

- [Min-Heap](#min-heap)

## Min-Heap

```Python
class MinHeap:
    def __init__(self, initial_capacity=10):
        # Create a complete binary tree using an array
        self.heap = [0] * (initial_capacity + 1)
        self.size = 0

    def add(self, element):
        self.size += 1
        if self.size >= len(self.heap):
            self.heap.extend([0] * len(self.heap))  # Double the size of the array if needed
        self.heap[self.size] = element
        self._sift_up(self.size)

    def peek(self):
        if self.size == 0:
            print("Heap is empty")
            return None
        return self.heap[1]

    def pop(self):
        if self.size == 0:
            print("Heap is empty")
            return None
        root = self.heap[1]
        self.heap[1] = self.heap[self.size]
        self.size -= 1
        self._sift_down(1)
        return root

    def _sift_up(self, index):
        parent_index = index // 2
        while parent_index > 0 and self.heap[index] < self.heap[parent_index]:
            self.heap[index], self.heap[parent_index] = self.heap[parent_index], self.heap[index]
            index = parent_index
            parent_index = index // 2

    def _sift_down(self, index):
        while index * 2 <= self.size:
            smallest_child_index = self._get_smallest_child_index(index)
            if self.heap[index] > self.heap[smallest_child_index]:
                self.heap[index], self.heap[smallest_child_index] = self.heap[smallest_child_index], self.heap[index]
            index = smallest_child_index

    def _get_smallest_child_index(self, index):
        left_child_index = index * 2
        right_child_index = index * 2 + 1
        if right_child_index > self.size or self.heap[left_child_index] < self.heap[right_child_index]:
            return left_child_index
        return right_child_index

    def heapify(self, elements):
        # Build a heap from a list of elements
        self.size = len(elements)
        self.heap = [0] + elements[:]
        for i in range(self.size // 2, 0, -1):
            self._sift_down(i)

    def __len__(self):
        return self.size

    def __str__(self):
        return str(self.heap[1:self.size + 1])

if __name__ == "__main__":
    # Test cases
    minHeap = MinHeap()
    minHeap.add(3)
    minHeap.add(1)
    minHeap.add(2)
    print(minHeap)  # [1, 3, 2]
    print(minHeap.peek())  # 1
    print(minHeap.pop())  # 1
    print(minHeap.pop())  # 2
    print(minHeap.pop())  # 3
    minHeap.add(4)
    minHeap.add(5)
    print(minHeap)  # [4, 5]

    # Building a heap from a list of elements
    elements = [4, 10, 3, 5, 1]
    minHeap.heapify(elements)
    print(minHeap)  # [1, 4, 3, 5, 10]
```
