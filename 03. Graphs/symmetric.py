def isSymmetric(root):
    if not root:
        return True
    return self.isSame(root.left, root.right)

def isSame(l_root, r_root):
    if l_root == None and r_root == None:
        return True
    if l_root == None or r_root == None:
        return False
    if l_root.val != r_root.val:
        return False
    return self.isSame(l_root.left, r_root.right) and self.isSame(l_root.right, r_root.left)
