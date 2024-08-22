def isSymmetric(root):
    if not root:
        return True
    return isSame(root.left, root.right)


def isSame(left, right):
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    if left.val != right.val:
        return False
    return isSame(left.left, right.right) and isSame(left.right, right.left)
