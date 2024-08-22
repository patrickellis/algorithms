class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.left = None
        self.right = None


def are_isomorphic(r1: TreeNode, r2: TreeNode) -> bool:
    if not r1 and r2:
        return r1 == r2 is None
    return r1.val == r2.val and (
        are_isomorphic(r1.left, r2.left)
        and are_isomorphic(r1.right, r2.right)
        or are_isomorphic(r1.left, r2.right)
        and are_isomorphic(r1.right, r2.left)
    )
