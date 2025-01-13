class Solution(object):
    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        n, m = len(board), len(board[0])
        seen = set()

        def is_valid(i, j):
            return 0 <= i < n and 0 <= j < m and board[i][j] == "O" and (i, j) not in seen

        def is_border(i, j):
            return i == 0 or i == n - 1 or j == 0 or j == m - 1

        def dfs(i, j):#去改变grid 的值
            if not is_valid(i, j): #出口
                return

            board[i][j] = "y"  #
            seen.add((i, j))  # two ways of 标记 i,j

            dfs(i - 1, j)  #拆分
            dfs(i + 1, j)
            dfs(i, j - 1)
            dfs(i, j + 1)

        # 只针对连到border上的连通块，搞一遍把"O"标记成"Y"，以后变成"O"，其余的都应该是"X"
        for i in range(n):
            for j in range(m):
                if is_border(i, j) and board[i][j] == "O":
                    dfs(i, j)

        # 针对连到border上的连通块，搞一遍都变成"y"了，所以变成"O"，其余的都应该是"X"了
        for i in range(n):
            for j in range(m):
                if board[i][j] == "y":
                    board[i][j] = "O"
                else:
                    board[i][j] = "X"
        return board


if __name__ == '__main__':
    solustion = Solution()
    board = [["X", "X", "X", "X"], ["X", "O", "O", "X"], ["X", "X", "O", "X"], ["X", "O", "X", "X"]]
    board = solustion.solve(board)
    print(board)