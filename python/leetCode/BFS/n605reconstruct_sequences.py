import collections


class Solution:
    """
    @param org: a permutation of the integers from 1 to n
    @param seqs: a list of sequences
    @return: true if it can be reconstructed only one or false
    """

    def sequenceReconstruction(self, org, seqs):
        # build graph from seqs not from org
        # top_sort when len(queue)> 1 means more than one top_order
        graph = {}
        for seq in seqs:
            for node in seq:
                if node not in graph:
                    graph[node] = []

        for seq in seqs:
            for i in range(0, len(seq) - 1):
                graph[seq[i]].append(seq[i + 1])

        order = self.top_sort(graph)
        return order == org

    def get_indegrees(self, graph):
        indegrees = {node: 0 for node in graph}
        for node in graph:
            for neighbor in graph[node]:
                indegrees[neighbor] += 1
        return indegrees

    def top_sort(self, graph):
        indegrees = self.get_indegrees(graph)
        order = []
        start_nodes = [node for node in indegrees if indegrees[node] == 0]
        queue = collections.deque(start_nodes)
        while queue:
            if len(queue) > 1: return None  # len(queue)> 1 means more than one top_order
            node = queue.popleft()
            order.append(node)
            for neighbor in graph[node]:
                indegrees[neighbor] -= 1
                if indegrees[neighbor] == 0:
                    queue.append(neighbor)
        return order
