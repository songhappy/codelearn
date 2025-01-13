# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def flatten(self, root) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if not root:
            return None
        node = root
        while node:
            if node.left:
                rightmost = node.left
                while rightmost.right:
                    rightmost = rightmost.right
                rightmost.right = node.right
                node.right = node.left
                node.left = None
            node = node.right

        return None


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution2:  # explicitly write out conditions
    def flatten(self, root) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        node = root
        stack = []
        while node:
            if node.left:
                if node.right:
                    stack.append(node.right)
                node.right = node.left
                node.left = None
            if not node.left and not node.right and stack:
                node.right = stack.pop()
            node = node.right


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution3:
    def flatten(self, root) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        # 递归出口一定要到叶子节点
        if not root: return None
        if not root.left and not root.right: return root

        lefttail = self.flatten(root.left)
        righttail = self.flatten(root.right)
        if lefttail:
            lefttail.right = root.right
            root.right = root.left
            root.left = None
        return righttail if righttail else None
