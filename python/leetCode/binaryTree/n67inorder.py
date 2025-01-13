class Node:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None


class Solution:
    """
    @param root: A Tree
    @return: Inorder in ArrayList which contains node values.
    """

    def inorderTraversal1(self, root):
        # write your code here
        if not root:
            return []
        current = [root.val]
        left = self.inorderTraversal1(root.left)
        right = self.inorderTraversal1(root.right)
        return left + current + right

    def inorderTraversal2(self, root):
        self.inorder = []
        self.traverse(root)
        return self.inorder

    def traverse(self, root):
        if not root:
            return
        self.traverse(root.left)
        self.inorder.append(root.val)
        self.traverse(root.right)
        return

    # still problematicssss
    def inorderTraversal(self, root):
        stack = []
        result = []
        curr = root
        while (curr or stack):
            while curr:
                stack.append(curr)
                curr = curr.left
            curr = stack.pop()
            result.append(curr.val)
            curr = curr.right
        return result


if __name__ == '__main__':
    s = Solution()
    root = Node(1)
    root.left = Node(2)
    root.right = Node(3)
    root.left.left = Node(4)
    root.left.right = Node(5)
    # res = s.inorderTraversal(root)
    # print(res)
    res1 = s.inorderTraversal1(root)
    res2 = s.inorderTraversal2(root)
    print(res1)
    print(res2)
    print("hello")
