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
