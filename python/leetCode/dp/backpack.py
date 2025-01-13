class Backpack1:
    # Given n items with size Ai, and an integer m denotes the size of a backpack.
    # How full you can fill this backpack?
    # array = [3, 4, 8, 5], size = 10
    # output = 9

    def backpack(self, m, A):
        n = len(A)
        dp = [[0] * (m + 1) for _ in range(n + 1)]  # ith item, size j, maximum
        for i in range(1, n):
            for j in range(1, m):
                dp[i][j] = max(dp[i - 1][j],
                               dp[i - 1][j - A[i]] + A[i])  # either have or not have A[i]

        return dp[n - 1][m - 1]


if __name__ == '__main__':
    backpack = Backpack1()
    res = backpack.backpack(10, [3, 4, 8, 5])
    print(res)
