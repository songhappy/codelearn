# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# complexity: nGn, where Gn = (4^n)/n^(3/2), Catalan number

class Solution:
    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:

        return self.search(1, n)

    def search(self, fromm, to):
        if fromm > to: return [None]

        res = []
        for i in range(fromm, to + 1):
            left = self.search(fromm, i - 1)
            right = self.search(i + 1, to)
            for l in left:
                for r in right:
                    root = TreeNode(i)
                    root.left = l
                    root.right = r
                    res.append(root)
        return res
