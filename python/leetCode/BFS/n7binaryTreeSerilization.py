# i is index of nodes in queue, you always want to have a queue

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None


class Solution:
    """
    @param root: An object of TreeNode, denote the root of the binary tree.
    This method will be invoked first, you should design your own algorithm
    to serialize a binary tree which denote by a root node to a string which
    can be easily deserialized by your own "deserialize" method later.
    """

    def serialize(self, root):
        # write your code here
        if root is None: return "{}"
        queue = [root]
        i = 0
        while i < len(queue):
            if queue[i]:
                queue.append(queue[i].left)
                queue.append(queue[i].right)
            i = i + 1

        res = [str(node.val) if node else "#" for node in queue]
        return '{%s}' % ','.join(res)

    """
    @param data: A string serialized by your serialize method.
    This method will be invoked second, the argument data is what exactly
    you serialized at method "serialize", that means the data is not given by
    system, it's given by your own serialize method. So the format of data is
    designed by yourself, and deserialize it here as you serialize it in 
    "serialize" method.
    """

    def deserialize(self, data):
        # write your code here
        data = data.strip()
        if data == '{}': return None
        values = data[1:-1].split(',')
        root = TreeNode(int(values[0]))
        queue = [root]
        forLeft = True
        i = 0
        for val in values[1:]:
            if val is not "#":
                node = TreeNode(int(val))
                if forLeft:
                    queue[i].left = node
                else:
                    queue[i].right = node
                queue.append(node)
            if not forLeft:  # done with two children, go next node
                i = i + 1
            forLeft = not forLeft  # done with left, then go right
        return root
