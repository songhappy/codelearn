class Solution:
    """
    @param L: Given n pieces of wood with length L[i]
    @param k: An integer
    @return: The maximum length of the small pieces
    """

    def woodCut(self, L, k):
        # write your code here
        # the last length that >= k
        if L is None or len(L) == 0: return 0

        def getNpieces(i, L):  # get the number of pieces given a lenth of i
            sum = 0
            for l in L:
                sum = sum + l // i
            return sum

        left, right = 1, max(L)
        while (left + 1 < right):
            mid = left + (right - left) // 2
            if getNpieces(mid, L) >= k:
                left = mid
            else:
                right = mid
        if getNpieces(right, L) >= k: return right
        if getNpieces(left, L) >= k: return left

        return 0
