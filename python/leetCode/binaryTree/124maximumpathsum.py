# 可以不从根节点出发
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


import sys


class Solution1:
    """
    @param root: The root of binary tree.
    @return: An integer
    """

    def maxPathSum(self, root):
        # write your code here
        result = self.dfs(root)
        return result[0]

    def dfs(self, root):
        # exit
        if root is None:
            return -sys.maxsize, -sys.maxsize

        # devide
        left = self.dfs(root.left)
        right = self.dfs(root.right)

        # conquer
        acrossRoot = max(left[1], 0) + root.val + max(right[1],
                                                      0)  # 等于上一层 left root2any + node.val + right root2any
        root2Any = max(left[1], right[1], 0) + root.val
        any2Any = max(left[0], right[0], acrossRoot)

        return any2Any, root2Any


class Solution:
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        # self.maxSum = -sys.maxsize - 1

        def max_gain(node):
            nonlocal max_sum
            if not node:
                return 0

            # max sum on the left and right sub-trees of node
            left_gain = max(max_gain(node.left), 0)
            right_gain = max(max_gain(node.right), 0)

            # the price to start a new path where `node` is a highest node
            price_newpath = node.val + left_gain + right_gain

            # update max_sum if it's better to start a new path
            max_sum = max(max_sum, price_newpath)

            # for recursion :
            # return the max gain if continue the same path # 从当前这个root 出发的
            return node.val + max(left_gain, right_gain)

        max_sum = float('-inf')
        max_gain(root)
        return max_sum


# 要求包括根节点
def maxPath2(root):
    if root is None: return -sys.maxint
    left = maxPath2(root.left)
    right = maxPath2(root.right)
    return max(0, left, right) + root.val
