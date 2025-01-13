class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None


import sys


class Solution:
    """
    @param root: the root of binary tree
    @return: the root of the minimum subtree
    """

    def min_subtree_Sum(self, root):
        if root is None:
            return 0
        left_sum = self.min_subtree_Sum(root.left)
        right_sum = self.min_subtree_Sum(root.right)
        root_sum = left_sum + root.val + right_sum

        if root_sum < self.min_sum:  # subtree always contains left + root + right
            self.min_sum = root_sum
            self.min_subtree = root
        return root_sum

    def findSubtree(self, root):
        self.min_sum = sys.maxsize
        self.min_subtree = None
        root_sum = self.min_subtree_Sum(root)

        return self.min_subtree
