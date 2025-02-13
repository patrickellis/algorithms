"""finds the lowest common ancestor of two nodes in a binary tree.

READ AHEAD!

"""

"""
BINARY TREE
===========

1. Find LCA of p and q given the root. NO parent pointers.
   Solution: DFS

2. Find LCA of p and q. PARENT POINTERS.
   Solution: Intersection of two linked lists.

3. Find LCA of p and q

BINARY SEARCH TREE
==================

1. Find LCA of p and q given the root. NO parent pointers.
   Solution: Iterative search.

"""

def LCA_two_nodes(root: "TreeNode", p: "TreeNode", q: "TreeNode"):
    """
    THE STANDARD ALGORITHM. Uses DFS.

    Finds the lowest common ancestor (LCA) of two nodes p and q in a binary tree.

    Parameters:
    - root (TreeNode): The root of the binary tree.
    - p (TreeNode): The first node.
    - q (TreeNode): The second node.

    Returns:
    - TreeNode: The lowest common ancestor of nodes p and q.
    """
    if root is None:
        return None

    if root == p or root == q:
        return root

    left = LCA(root.left, p, q)
    right = LCA(root.right, p, q)

    if left and right:
        return root

    return left or right

def LCA_list_of_nodes(root: 'TreeNode', nodes: 'List[TreeNode]') -> 'TreeNode':
    """
    THE STANDARD ALGORITHM. Uses DFS.

    Finds the lowest common ancestor (LCA) of a LIST OF NODES in a binary tree.

    Parameters:
    - root (TreeNode): The root of the binary tree.
    - nodes (list[TreeNode]): List of nodes for which to find LCA.

    Returns:
    - TreeNode: The lowest common ancestor of all nodes.
    """
        if root is None:
            return None

        if root in nodes:
            return root

        left = LCA_list_of_nodes(root.left, nodes)
        right = LCA_list_of_nodes(root.right, nodes)

        if left and right:
            return root

        return left or right


def LCA_parent_pointer_1(p: "DoublyLinkedNode", q: "DoublyLinkedNode") -> "DoublyLinkedNode":
    """
    This implementation is identical to the algorithm that finds the intersection
    of two nodes in separate Linked Lists.
    """
    p_head, q_head = p, q

    while p != q:
        p = p.parent if p else q_head
        q = q.parent if q else p_head

    return p


def LCA_parent_pointer_2(p: "DoublyLinkedNode", q: "DoublyLinkedNode") -> "DoublyLinkedNode":
    def height(node: "DoublyLinkedNode") -> int:
        height = 0
        while node:
            node = node.parent
            height += 1
        return height

    ph = height(p)
    qh = height(q)

        p = p.parent
        ph -= 1
    while qh > ph:
        q = q.parent
        qh -= 1

    while p != q:
        p = p.parent
        q = q.parent

    return p


def LCA_parent_pointer_3(p: "DoublyLinkedNode", q: "DoublyLinkedNode") -> "DoublyLinkedNode":
    ancestors = set()

    while p:
        ancestors.add(p)
        p = p.parent

    while q:
        if q in ancestors:
            return q
        q = q.parent

    return None


