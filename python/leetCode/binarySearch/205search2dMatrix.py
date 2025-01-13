class Solution:
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if matrix is None or len(matrix) == 0 or len(matrix[0]) == 0: return False
        m = len(matrix)
        n = len(matrix[0])
        i, j = 0, n - 1
        while (i <= m - 1 and j >= 0):
            if (target == matrix[i][j]):
                return True
            elif (target > matrix[i][j]):
                i = i + 1
            else:
                j = j - 1
        return False
