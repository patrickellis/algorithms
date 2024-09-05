from itertools import zip_longest

class TreeNode:
    def __init__(self, val: int, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right

"""
This is just a Depth-First traversal function, where an expand function
provides the order we would like to visit the children and parent, which are
then reversed() because stacks are LIFO (so nodes will be popped in reverse).

We don't use a visited set, since trees are acyclic, but we do mark the nodes
as expanded or not so that we can re-visit each parent without expanding it twice.
"""
def tree_iterator(root, expand):
    frontier = [(False, root)]
    while frontier:
        expanded, curr = frontier.pop()
        if not curr:
            continue
        if expanded:
            yield curr
        else:
            frontier.extend((x is curr, x) for x in reversed(expand(curr)))


def preorder(root):
    return tree_iterator(root, lambda node: (node, node.left, node.right))
def inorder(root):
    return tree_iterator(root, lambda node: (node.left, node, node.right))
def postorder(root):
    return tree_iterator(root, lambda node: (node.left, node.right, node))

def reverse_preorder(root):
    return tree_iterator(root, lambda node: (node, node.right, node.left))
def reverse_inorder(root):
    return tree_iterator(root, lambda node: (node.right, node, node.left))
def reverse_postorder(root):
    return tree_iterator(root, lambda node: (node.right, node.left, node))

#####################
# Example use cases #
#####################

def count_nodes(root):
    return sum(1 for _ in inorder(root))

def count_leaves(root):
    return sum(1 for x in inorder(root) if not x.left and not x.right)

def flatten(root):
    return [x.val for x in inorder(root)]

def kth_smallest(root, k):
    return next(x.value for i,x in enumerate(inorder(root)) if i == k-1)

def kth_largest(root, k):
    return next(x.value for i,x in enumerate(inorder(root)) if i == k-1)


def is_symmetric(tree_a, tree_b):
    return all(a.value == b.value for a,b in zip_longest(inorder(tree_a), reverse_inorder(tree_b)))

