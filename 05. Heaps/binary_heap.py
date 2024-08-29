from random import randint
import math
from io import StringIO


class BinaryHeap:
    def __init__(self):
        self.heap = []

    def max_heapify(self, index: int) -> None:
        left = 2 * index + 1
        right = 2 * index + 2
        largest = index
        heap_size = len(self.heap)

        if left < heap_size and self.heap[left] > self.heap[largest]:
            largest = left
        if right < heap_size and self.heap[right] > self.heap[largest]:
            largest = right
        if largest != index:
            self.heap[index], self.heap[largest] = self.heap[largest], self.heap[index]
            self.max_heapify(largest)

    def build_max_heap(self) -> None:
        for i in range(len(self.heap) // 2 - 1, -1, -1):
            self.max_heapify(i)

    def insert(self, val: int) -> None:
        index = len(self.heap)
        parent = (index-1)//2
        self.heap.append(val)

        while index > 0 and self.heap[index] > self.heap[parent]:
            self.heap[parent], self.heap[index] = self.heap[index], self.heap[parent]
            index = parent
            parent = (index-1)//2

    def extract_max(self) -> int:
        if len(self.heap) < 1:
            raise IndexError("Heap is empty")

        max_val = self.heap[0]
        last_val = self.heap.pop()

        if len(self.heap) > 0:
            self.heap[0] = last_val
            self.max_heapify(0)

        return max_val

    def get_max(self) -> int:
        if len(self.heap) < 1:
            raise IndexError("Heap is empty")
        return self.heap[0]

    def __str__(self):
        def show_tree(tree, total_width=80, fill=" "):
            """Pretty-print a tree.
            total_width depends on your input size"""
            output = StringIO()
            last_row = -1
            res = ""
            for i, n in enumerate(tree):
                if i:
                    row = int(math.floor(math.log(i + 1, 2)))
                else:
                    row = 0
                if row != last_row:
                    output.write("\n")
                columns = 2**row
                col_width = int(math.floor((total_width * 1.0) / columns))
                output.write(str(n).center(col_width, fill))
                last_row = row
            res = output.getvalue()
            res += "\n" + "-" * total_width
            return res

        return show_tree(self.heap)


heap = BinaryHeap()
for _ in range(14):
    n = randint(0, 100)
    heap.insert(n)

# heap.insert(10)
# heap.insert(20)
# heap.insert(5)
# heap.insert(30)

print("Heap:", heap)  # Output: Heap: [30, 20, 5, 10]
print("Max val:", heap.get_max())  # Output: Max val: 30

max_val = heap.extract_max()
print("Extracted max:", max_val)  # Output: Extracted max: 30
print("Heap after extraction:", heap)  # Output: Heap after extraction: [20, 10, 5]
