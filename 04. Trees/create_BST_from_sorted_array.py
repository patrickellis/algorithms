class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def sorted_array_to_bst(A):
    if not A:
        return None

    mid = len(A) // 2
    root = TreeNode(A[mid])

    root.left = sorted_array_to_bst(A[:mid])
    root.right = sorted_array_to_bst(A[mid+1:])

    return root
