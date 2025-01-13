# dependencies=[[a,b],[b,c],[d,e],[c,e]], n=5
# output [e,d,c,b,a]
import collections


class Solution:
    def get_indegrees(self, dependencies):
        indegrees = {}
        graph = {}
        for [node, neighbor] in dependencies:
            indegrees[node] = 0
            indegrees[neighbor] = 0
            graph[node] = []
            graph[neighbor] = []
        for [neighbor, node] in dependencies:
            indegrees[neighbor] += 1
            graph[node].append(neighbor)
        print(indegrees)
        print(graph)
        return indegrees, graph

    def top_sort(self, dependencies):
        indegrees, graph = self.get_indegrees(dependencies)
        start_nodes = [k for k, v in indegrees.items() if v == 0]
        queue = collections.deque(start_nodes)
        top_order = []
        while queue:
            node = queue.popleft()
            top_order.append(node)

            for neighbor in graph[node]:
                indegrees[neighbor] = indegrees[neighbor] - 1
                if indegrees[neighbor] == 0:
                    queue.append(neighbor)
        print(top_order)
        print(graph)
        if len(top_order) != len(graph):
            return None
        return top_order


if __name__ == '__main__':
    s = Solution()
    dependencies = [['a', 'b'], ['b', 'c'], ['d', 'e'], ['c', 'e']]
    n = 5
    result = s.top_sort(dependencies)
    print(result)
