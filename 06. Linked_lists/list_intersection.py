"""Finds the intersection node of two linked lists.

READ AHEAD!

This implementation is identical to the algorithm that finds the LCA of two
nodes in a binary tree.
"""


class Node:
    def __init__(self, val):
        self.val = val
        self.next = None


class Solution:
    def getIntersectionNode_intuitive(self, p: "Node", q: "Node") -> "Node":
        def length(node: "Node") -> int:
            length = 0
            while node:
                node = node.next
                length += 1
            return length

        ph = length(p)
        qh = length(q)

        while ph > qh:
            p = p.next
            ph -= 1
        while qh > ph:
            q = q.next
            qh -= 1

        while p != q:
            p = p.next
            q = q.next

        return p

    def getIntersectionNode_concise(self, p: "Node", q: "Node") -> "Node":
        p_head, q_head = p, q

        while p != q:
            p = p.next if p else q_head
            q = q.next if q else p_head

        return p


if __name__ == "__main__":
    # Test case 1
    # Intersection at 8
    p = Node(4)
    p.next = Node(1)
    p.next.next = Node(8)
    p.next.next.next = Node(4)
    p.next.next.next.next = Node(5)
    q = Node(5)
    q.next = Node(6)
    q.next.next = Node(1)
    q.next.next.next = p.next.next
    q.next.next.next.next = Node(9)
    if Solution().getIntersectionNode_concise(p, q).val == 8:
        print("Test case 1 passed.")
    else:
        print("Test case 1 failed.")
