from typing import Optional

"""Diameter of Binary Tree.

https://leetcode.com/problems/diameter-of-binary-tree/


"""

# Definition for a binary tree node
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    diameter = 0

    def diameterOfBinaryTree(self, root: Optional['TreeNode']) -> int:
        def dfs(root: Optional['TreeNode']) -> int:
            if not root:
                return 0
            left = dfs(root.left)
            right = dfs(root.right)
            self.diameter = max(self.diameter, left+right)
            return 1+max(left,right)
        dfs(root)
        return self.diameter


