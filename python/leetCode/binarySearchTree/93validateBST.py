# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def helper(self, root, lower=float('-inf'), upper=float('inf')):
        if not root: return True
        left = self.helper(root.left, lower, root.val)
        right = self.helper(root.right, root.val, upper)
        relroot = lower < root.val < upper
        return relroot and left and right

    def isValidBST(self, root: TreeNode) -> bool:
        return self.helper(root)
