import collections


# no level can be detected
class UndirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []


class Solution:
    """
    @param node: A undirected graph node
    @return: A undirected graph node
    """

    def cloneGraph(self, node):
        root = node
        # write your code here
        if not node: return None
        # get all modes
        nodes = self.get_nodes(root)
        # copy nodes
        mapping = {}
        for node in nodes:
            mapping[node] = UndirectedGraphNode(node.label)
        # copy neighbors, build edges
        for node in nodes:
            new_node = mapping[node]
            for neighbor in node.neighbors:
                new_node.neighbors.append(mapping[neighbor])

        return mapping[root]

    def get_nodes(self, root):
        nodes = set([root])
        queue = collections.deque([root])
        while queue:
            node = queue.popleft()
            for neighbor in node.neighbors:
                if neighbor not in nodes:
                    queue.append(neighbor)
                    nodes.add(neighbor)
        return nodes
