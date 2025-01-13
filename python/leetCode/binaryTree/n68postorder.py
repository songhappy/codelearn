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
    @return: Postorder in ArrayList which contains node values.
    """

    def traverse(self, root):
        if not root: return
        self.traverse(root.left)
        self.traverse(root.right)
        self.result.append(root.val)

    def postorderTraversal1(self, root):
        # write your code here
        if not root:
            return []
        self.result = []
        self.traverse(root)
        return self.result

    def postorderTraversal(self, root):
        if not root: return []
        left = self.postorderTraversal(root.left)
        right = self.postorderTraversal(root.right)
        return left + right + [root.val]
