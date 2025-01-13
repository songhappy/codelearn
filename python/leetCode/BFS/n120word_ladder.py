import collections


class Solution:
    """
    @param: start: a string
    @param: end: a string
    @param: dict: a set of string
    @return: An integer
    """

    def ladderLength(self, start, end, dict):
        dict.add(end)
        queue = collections.deque([start])
        length = {start: 1}
        while queue:
            node = queue.popleft()
            if node == end:
                return length[node]
            neighbors = self.get_neighbors(node)
            for neighbor in neighbors:
                if neighbor in dict and neighbor not in length:
                    queue.append(neighbor)
                    length[neighbor] = length[node] + 1
        return -1

    def get_neighbors(self, word):
        neighbors = []
        for i in range(len(word)):
            pre = word[:i]
            post = word[i + 1:]
            for new_c in "abcdefghijklmnopqrstuvwxyz":
                new_word = pre + new_c + post
                neighbors.append(new_word)
        return neighbors
