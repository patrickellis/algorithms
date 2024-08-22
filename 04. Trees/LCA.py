"""finds the lowest common ancestor of two nodes in a binary tree.

READ AHEAD!

This implementation is identical to the algorithm that finds the intersection
of two nodes in separate Linked Lists.
"""


class Node:
    def __init__(self, val=0, left=None, right=None, parent=None):
        self.val = val
        self.left = left
        self.right = right
        self.parent = parent


def lowestCommonAncestor(self, p: "Node", q: "Node") -> "Node":
    p_head, q_head = p, q

    while p != q:
        p = p.parent if p else q_head
        q = q.parent if q else p_head

    return p


def lowestCommonAncestor_simple(self, p: "Node", q: "Node") -> "Node":
    def height(node: "Node") -> int:
        height = 0
        while node:
            node = node.parent
            height += 1
        return height

    ph = height(p)
    qh = height(q)

    while ph > qh:
        p = p.parent
        ph -= 1
    while qh > ph:
        q = q.parent
        qh -= 1

    while p != q:
        p = p.parent
        q = q.parent

    return p


def lowestCommonAncestor_sets(self, p: "Node", q: "Node") -> "Node":
    ancestors = set()

    while p:
        ancestors.add(p)
        p = p.parent

    while q:
        if q in ancestors:
            return q
        q = q.parent

    return None
