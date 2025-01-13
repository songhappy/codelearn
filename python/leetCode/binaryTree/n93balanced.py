"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""


class Solution:
    """
    @param root: The root of binary tree.
    @return: True if this Binary tree is Balanced, or false.
    """

    def max_depth(self, root):
        if not root: return 0
        left = self.max_depth(root.left)
        right = self.max_depth(root.right)
        return max(left, right) + 1

    def isBalanced(self, root):
        # write your code here
        if not root: return True
        left = self.isBalanced(root.left)
        right = self.isBalanced(root.right)
        relroot = True if abs(
            self.max_depth(root.left) - self.max_depth(root.right)) <= 1 else False
        return left and right and relroot


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution2:
    def isBalanced(self, root) -> bool:
        return self.isBalancedHelper(root)[0]

    def isBalancedHelper(self, root):
        if not root:
            return [True, 0]
        left = self.isBalancedHelper(root.left)
        right = self.isBalancedHelper(root.right)
        height = max(left[1], right[1]) + 1
        balance = left[0] and right[0] and abs(left[1] - right[1]) <= 1
        return [balance, height]
