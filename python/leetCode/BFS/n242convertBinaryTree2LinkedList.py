class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    # @param {TreeNode} root the root of binary tree
    # @return {ListNode[]} a lists of linked list
    def binaryTreeToLists1(self, root):
        # Write your code here
        if root is None: return []

        res = []
        queue = [root]
        dummy = ListNode(0)
        lastNode = None
        while queue:
            queueAtLevel = []
            atLevel = []
            dummy.next = None
            lastNode = dummy
            for node in queue:
                lnode = ListNode(node.val)
                atLevel.append(lnode)
                if node.left:
                    queueAtLevel.append(node.left)
                if node.right:
                    queueAtLevel.append(node.right)
            for l in atLevel[:]:
                lastNode.next = l
                lastNode = lastNode.next
            queue = queueAtLevel
            res.append(dummy.next)
        return res

    def binaryTreeToLists(self, root):
        result = []
        if root is None: return result

        import collections
        queue = collections.deque()
        queue.append(root)

        dummy = ListNode(0)

        while queue:
            p = dummy
            size = len(queue)
            for i in range(size):
                head = queue.popleft()
                p.next = ListNode(head.val)
                p = p.next

                if head.left is not None:
                    queue.append(head.left)
                if head.right is not None:
                    queue.append(head.right)

            result.append(dummy.next)

        return result
