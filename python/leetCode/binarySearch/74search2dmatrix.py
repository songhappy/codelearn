class Solution:
    def k2ij(self, k, n):
        i = k // n
        j = k % n
        return i, j

    def searchMatrix(self, matrix, target: int) -> bool:
        if matrix is None or len(matrix) == 0 or len(matrix[0]) == 0:
            return False

        m = len(matrix)
        n = len(matrix[0])

        left = 0;
        right = m * n - 1
        while (left + 1 < right):
            mid = (left + right) // 2
            i, j = self.k2ij(mid, n)
            if (target == matrix[i][j]):
                return True
            elif (target > matrix[i][j]):
                left = mid
            else:
                right = mid

        i, j = self.k2ij(left, n)
        if (target == matrix[i][j]): return True
        i, j = self.k2ij(right, n)
        if (target == matrix[i][j]): return True

        return False
