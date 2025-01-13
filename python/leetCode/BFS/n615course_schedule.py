import collections


class Solution:
    """
    @param: numCourses: a total of n courses
    @param: prerequisites: a list of prerequisite pairs
    @return: true if can finish all courses or false
    """

    def canFinish(self, numCourses, prerequisites):
        # write your code here
        # how to build graph, always a dictionary, other given graphs could be rootnode or matrix
        # order = topsort graph
        # length of order == n
        # always add into result when popout

        if numCourses <= 0: return False
        if len(prerequisites) == 0: return True

        graph = {i: [] for i in range(numCourses)}
        indegrees = {i: 0 for i in range(numCourses)}
        for [node, neighbor] in prerequisites:
            graph[node].append(neighbor)
            indegrees[neighbor] += 1
        order = self.top_sort(graph, indegrees)

        if numCourses == len(order): return True
        return False

    def top_sort(self, graph, indegrees):
        start_nodes = [node for node in indegrees if indegrees[node] == 0]
        queue = collections.deque(start_nodes)
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in graph[node]:
                indegrees[neighbor] -= 1
                if indegrees[neighbor] == 0:
                    queue.append(neighbor)

        return order


class Solution2:
    """
    @param: numCourses: a total of n courses
    @param: prerequisites: a list of prerequisite pairs
    @return: the course order
    """

    def findOrder(self, numCourses, prerequisites):
        # write your code here
        if len(prerequisites) == 0: return [i for i in range(numCourses)]
        graph = {i: [] for i in range(numCourses)}
        indegrees = {i: 0 for i in range(numCourses)}
        for [neighbor, node] in prerequisites:
            graph[node].append(neighbor)
            indegrees[neighbor] += 1  # 从node 指向了neighbor，所以统计neighber 的indegrees

        order = self.top_sort(graph, indegrees)
        if len(order) == numCourses: return order

        return []

    def top_sort(self, graph, indegrees):
        order = []
        import collections
        start_nodes = [node for node in indegrees if indegrees[node] == 0]
        queue = collections.deque(start_nodes)
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in graph[node]:
                indegrees[neighbor] -= 1
                if indegrees[neighbor] == 0:
                    queue.append(neighbor)
        return order
