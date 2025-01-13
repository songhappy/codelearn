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
    @return: Level order a list of lists of integer
    """

    # time O(n), space O(n)
    # two loops
    def levelOrder(self, root):
        # write your code here
        if root is None: return []

        queue = [root]
        res = []
        while queue:  # for each level
            queueAtLevel = []
            atLevel = []
            for node in queue:  # for each node in that level
                atLevel.append(node.val)
                if node.left: queueAtLevel.append(node.left)
                if node.right: queueAtLevel.append(node.right)
            queue = queueAtLevel
            res.append(atLevel)
        return res
