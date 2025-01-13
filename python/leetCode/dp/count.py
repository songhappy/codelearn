# examples:
# N=1, return 1
# N=2, return 2: (1,1), (2)
# N=3, return 3
# N=4, return 5

memo = {0: 0, 1: 1, 2: 2}


def num_ways_to_top(N):
    if N in memo:
        return memo[N]
    memo[N] = num_ways_to_top(N - 1) + num_ways_to_top(N - 2)
    return memo[N]


class UniqueBST96:
    # time complexity = O(n) because any number only be calculated once

    def numTrees(self, n: int) -> int:
        memo = {0: 1, 1: 1, 2: 2}

        return self.search(n, memo)

    def search(self, n, memo):
        if n in memo: return memo[n]

        # for each root node, there are
        # (numTrees(left_subtree)) * (numTrees(right_subtree)) unique BST's
        res = 0
        for i in range(1, n + 1):
            res += self.search(i - 1, memo) * self.search(n - i, memo)
        memo[n] = res

        return memo[n]


if __name__ == '__main__':
    for i in range(10):
        res = num_ways_to_top(i)
        print(res)
