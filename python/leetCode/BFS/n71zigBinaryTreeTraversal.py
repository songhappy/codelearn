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
    @return: A list of lists of integer include the zigzag level order traversal of its nodes' values.
    """

    def zigzagLevelOrder(self, root):
        # write your code here
        if root is None: return []
        queue = [root]
        res = []
        n = 0
        while queue:
            queueAtLevel = []
            atLevel = []
            for node in queue:
                atLevel.append(node.val)
                if node.left: queueAtLevel.append(node.left)
                if node.right: queueAtLevel.append(node.right)
            if n % 2 == 1: atLevel.reverse()
            queue = queueAtLevel
            res.append(atLevel)
            n += 1
        return res
