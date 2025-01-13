# topo sort 找last level leaves, double directed graph
import collections


class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if n <= 0: return [0]
        if len(edges) == 0: return [0]

        graph = {i: [] for i in range(n)}
        indegrees = {i: 0 for i in range(n)}

        for node1, node2 in edges:
            graph[node1].append(node2)
            graph[node2].append(node1)
            indegrees[node1] += 1  # double direction
            indegrees[node2] += 1

        start_nodes = [node for node in indegrees if indegrees[node] == 1]
        queue = collections.deque(start_nodes)

        while queue:
            res = []  # nodes of last level of toplogical sort order
            size = len(queue)
            for i in range(size):  # level by level, explicitly loop level
                node = queue.popleft()
                res.append(node)
                indegrees[node] -= 1
                for neighbor in graph[node]:
                    indegrees[neighbor] -= 1
                    if indegrees[neighbor] == 1:
                        queue.append(neighbor)
        return res
