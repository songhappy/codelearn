# 55, can jump

class Solution:
    """
    @param A: A list of integers
    @return: A boolean
    """

    def canJump(self, A):
        if not A:
            return False

        n = len(A)
        # state: dp[i] 代表能否跳到坐标 i
        dp = [False] * n

        # initialization: 一开始站在0这个位置
        dp[0] = True

        # function
        for i in range(1, n):
            for j in range(i):
                # 高效的写法:
                if dp[j] and A[j] >= i - j:
                    dp[i] = True
                    break
                # 偷懒的写法
                # dp[i] = dp[i] or dp[j] and (j + A[j] >= i)

        # answer
        return dp[n - 1]

    # 45 minimal number of steps
    def jump(self, nums):
        n = len(nums)
        dp = [0] * n
        for i in range(1, n):
            dp[i] = float('inf')
            for j in range(i):
                # 如果从j可以到跳到i
                if nums[j] >= i - j:
                    dp[i] = min(dp[i], dp[j] + 1)
        return dp[n - 1]

    # 1306 if canreach O(N)
    def canReach(self, arr: List[int], start: int) -> bool:
        if 0 <= start < len(arr) and arr[start] >= 0:
            if (arr[start] == 0):
                return True

            arr[start] = -arr[start]
            return self.canReach(arr, start + arr[start]) or self.canReach(arr, start - arr[start])

        return False

    # these two jump games need better solution since submittion on leetcode  raises "time limit exceeded"

    # 1871
    def canReach(self, s: str, minJump: int, maxJump: int):
        """
        s="001010101111000"
        if i + minJump <= j <= min(i + maxJump, s.length - 1), and
        s[j] == '0', then can reach. ask if can reach last element of the string
        """
        if s[-1] == '1': return False
        if minJump <= len(s) - 1 <= maxJump: return True

        dp = [False] * len(s)
        dp[0] = True

        for i in range(minJump, len(s)):
            for j in range(max(0, i - maxJump), i - minJump + 1):
                if dp[j] and s[i] == '0':
                    dp[i] = True
                    break
        return dp[-1]
