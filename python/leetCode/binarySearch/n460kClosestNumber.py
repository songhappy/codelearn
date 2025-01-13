class Solution:
    """
    @param A: an integer array
    @param target: An integer
    @param k: An integer
    @return: an integer array
    """

    def kClosestNumbers(self, A, target, k):
        # write your code here
        if A is None or len(A) == 0: return -1

        # find position of target, if not target if target 3, Input: A = [1, 4, 6, 8], then the line
        l = 0;
        r = len(A) - 1
        while (l + 1 < r):
            mid = l + (r - l) // 2
            if A[mid] < target:
                l = mid
            else:
                r = mid
        print(l, r)

        res = []
        left = l;
        right = r;
        i = 0
        for _ in range(k):
            if left >= 0 and right <= len(A) - 1:
                if target - A[left] <= A[right] - target:
                    res.append(A[left])
                    left = left - 1
                else:
                    res.append(A[right])
                    right = right + 1

            elif right > len(A) - 1 and left >= 0:
                res.append(A[left])
                left = left - 1

            elif left < 0 and right <= len(A) - 1:
                res.append(A[right])
                right = right + 1
        return res
