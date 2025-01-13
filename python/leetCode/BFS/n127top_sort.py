"""
Definition for a Directed graph node
class DirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []
"""
import collections


class Solution:
    """
    @param: graph: A list of Directed graph node
    @return: Any topological order for the given graph.
    """

    def topSort(self, graph):
        # write your code here
        # 1. get get_indegrees
        # 2. nodes of zero indgrees into queue
        # 3. neighbors indgrees - 1
        # 4. always add into result when popout
        # 5. if visited, always into visited when into queue
        if not graph: return []
        node_indgrees = self.get_indegrees(graph)
        start_nodes = [node for node in graph if node_indgrees[node] == 0]
        queue = collections.deque(start_nodes)
        result = []  # always give results at when popout
        while queue:
            node = queue.popleft()
            result.append(node)
            for neigbor in node.neighbors:
                node_indgrees[neigbor] -= 1
                if node_indgrees[neigbor] == 0:
                    queue.append(neigbor)
        return result

    def get_indegrees(self, graph):
        node_indgrees = {}
        for node in graph:
            node_indgrees[node] = 0
        for node in graph:
            for neighbor in node.neighbors:
                node_indgrees[neighbor] += 1
        return node_indgrees
