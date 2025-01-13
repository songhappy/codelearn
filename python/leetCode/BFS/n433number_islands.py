class Solution:
    """
    @param grid: a boolean 2D matrix
    @return: an integer
    """

    # 5. if visited, always into visited when into queue

    def numIslands(self, grid):
        # write your code here
        matrix = grid
        if not matrix: return 0

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.visited = set()
        count = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] and (i, j) not in self.visited:
                    self.bfs(matrix, i, j)  # update visited
                    count = count + 1
        return count

    def bfs(self, matrix, i, j):
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        from collections import deque
        self.visited.add((i, j))
        queue = deque([(i, j)])
        while queue:
            (i, j) = queue.popleft()
            for (di, dj) in neighbors:
                (new_i, new_j) = (i + di, j + dj)
                if self.is_valid(new_i, new_j, matrix):
                    queue.append((new_i, new_j))
                    self.visited.add((new_i, new_j))
        return

    def is_valid(self, i, j, matrix):
        if i >= len(matrix) or i < 0: return False
        if j >= len(matrix[0]) or j < 0: return False
        if (i, j) in self.visited: return False
        if not matrix[i][j]: return False
        return True


class Solution1:
    def numIslands(self, grid) -> int:
        if not grid or len(grid) == 0: return 0

        num = 0
        m = len(grid)
        n = len(grid[0])
        seen = set()

        def is_valid(i, j):
            if (i, j) not in seen and i>=0 and i < len(grid) and j >= 0 and j <len(grid[0]) and grid[i][j] == "1":
                return True
            return False

        def dfs(i, j):
            if not is_valid(i, j):  #careful about 递归的出口
                return

            seen.add((i, j))

            dfs(i - 1, j)
            dfs(i + 1, j)
            dfs(i, j - 1)
            dfs(i, j + 1)

        for i in range(m):
            for j in range(n):
                if is_valid(i, j):
                    num += 1
                    dfs(i, j)
        return num


class Solution2:
    def numIslands(self, grid) -> int:
        if not grid:
            return 0

        def dfs(grid, i, j):
            if i < 0 or i >= len(grid): return
            if j < 0 or j >= len(grid[0]): return
            if grid[i][j] == "0": return

            grid[i][j] = "0"
            dfs(grid, i - 1, j)
            dfs(grid, i + 1, j)
            dfs(grid, i, j - 1)
            dfs(grid, i, j + 1)

        num = 0
        m = len(grid)
        n = len(grid[0])
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1":
                    num += 1
                    dfs(grid, i, j)
        return num




if __name__ == '__main__':
    grid1 = [["1", "1", "1", "1", "0"], ["1", "1", "0", "1", "0"], ["1", "1", "0", "0", "0"],
     ["0", "0", "0", "0", "0"]]
    grid2 = [["1", "1", "1", "1", "0"], ["1", "1", "0", "1", "0"], ["1", "1", "0", "0", "0"],
     ["0", "0", "0", "0", "0"]]
    solution = Solution1()
    print(solution.numIslands(grid1))

    solution2 = Solution2()
    print(solution2.numIslands(grid2))