from itertools import pairwise

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

root = TreeNode(2)

def inorder(root):
    if root is not None:
        for x in inorder(root.left):
            yield x
        yield root
        for x in inorder(root.right):
            yield x

def is_ordered(x):
    return all(a <= b for a, b in pairwise(x))

is_ordered(x.val for x in inorder(root))
