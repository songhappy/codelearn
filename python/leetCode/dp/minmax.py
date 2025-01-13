import sys


class CoinChange322:
    def coinChange(self, coins: List[int], amount: int) -> int:

        f = [sys.maxsize for _ in range(amount + 1)]  # fewest number of coins to make up i
        f[0] = 0

        for i in range(1, amount + 1):
            for j in coins:
                if i - j >= 0:
                    f[i] = min(f[i - j] + 1, f[i])
        if f[amount] < sys.maxsize:
            return f[amount]
        else:
            return -1


class TicketsCosts983:
    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        days.sort()
        days_set = set(days)
        durations = [1, 7, 30]
        dp = [sys.maxsize for _ in range(days[-1] + 1)]
        dp[0] = 0

        for i in range(1, days[-1] + 1):
            if i not in days_set:  ### make sure this
                dp[i] = dp[i - 1]
            for d, c in zip(durations, costs):
                previous_cost = 0 if i - d <= 0 else dp[i - d]  ###
                dp[i] = min(dp[i], previous_cost + c)
        return dp[days[-1]]


class Minstesp746:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        dp = [sys.maxsize for _ in range(len(cost) + 1)]

        # dp[i] = min(dp[i-1] + cost[i-1], dp[i-2] + cost[i-2])
        # dp[len(cost)]  # top means beyound the last step
        dp[0], dp[1] = 0, 0

        for i in range(2, len(cost) + 1):
            dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])

        return dp[len(cost)]
