"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""


class Solution:
    """
    @param root: A Tree
    @return: Preorder in ArrayList which contains node values.
    """

    def preorderTraversal1(self, root):
        # write your code here
        self.preorder = []
        self.traverse(root)
        return self.preorder

    def traverse(self, root):
        if not root:
            return
        self.preorder.append(root.val)
        self.traverse(root.left)
        self.traverse(root.right)

    def preorderTraversal2(self, root):
        if not root:
            return []
        preorder = []
        preorder.append(root.val)
        left = self.preorderTraversal2(root.left)  # divide
        right = self.preorderTraversal2(root.right)
        return preorder + left + right  # conquer

    def preorderTraversal(self, root):
        if not root:
            return []
        stack = [root]
        preorder = []
        while stack:
            node = stack.pop()
            preorder.append(node.val)
            if node.right: stack.append(node.right)
            if node.left: stack.append(node.left)
        return preorder
