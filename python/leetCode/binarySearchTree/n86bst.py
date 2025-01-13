"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

Example of iterate a tree:
iterator = BSTIterator(root)
while iterator.hasNext():
    node = iterator.next()
    do something for node
"""


class BSTIterator:
    """
    @param: root: The root of binary tree.
    """

    def __init__(self, root):
        # do intialization if necessary
        self.stack = []
        curr = root
        while curr:
            self.stack.append(curr)
            curr = curr.left

    """
    @return: True if there has next node, or false
    """

    def hasNext(self):
        # write your code here
        return len(self.stack) > 0

    """
    @return: return next node
    """

    def next(self):
        # write your code here
        node = self.stack[-1]
        if node.right:
            curr = node.right
            while curr:
                self.stack.append(curr)
                curr = curr.left
        else:
            curr = self.stack.pop()
            while len(self.stack) > 0 and self.stack[-1].right == curr:
                curr = self.stack.pop()

        return node


class BSTIterator:

    def __init__(self, root):
        # Array containing all the nodes in the sorted order
        self.nodes_sorted = []

        # Pointer to the next smallest element in the BST
        self.index = -1

        # Call to flatten the input binary search tree
        self._inorder(root)

    def _inorder(self, root):
        if not root:
            return
        self._inorder(root.left)
        self.nodes_sorted.append(root.val)
        self._inorder(root.right)

    def next(self) -> int:
        """
        @return the next smallest number
        """
        self.index += 1
        return self.nodes_sorted[self.index]

    def hasNext(self) -> bool:
        """
        @return whether we have a next smallest number
        """
        return self.index + 1 < len(self.nodes_sorted)
