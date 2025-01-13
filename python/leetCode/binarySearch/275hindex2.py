class Solution:
    # find the fist place that citations[idx] >= h where h = n - idx
    def hIndex(self, citations) -> int:
        if citations is None or len(citations) == 0:
            return 0
        n = len(citations)
        left = 0;
        right = n - 1
        while (left + 1 < right):
            mid = left + (right - left) // 2
            if citations[mid] < n - mid:
                left = mid
            else:
                right = mid

        if citations[left] >= n - left: return n - left
        if citations[right] >= n - right: return n - right

        return 0
