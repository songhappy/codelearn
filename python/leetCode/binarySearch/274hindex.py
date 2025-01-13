class Solution:
    def hIndex(self, citations) -> int:
        if citations is None or len(citations) == 0: return 0

        sorted_citations = sorted(citations)
        # fint the fist place that c[idx] >= h, where h = n - idx
        n = len(citations)
        left = 0;
        right = n - 1
        while (left + 1 < right):
            mid = left + (right - left) // 2
            if sorted_citations[mid] < n - mid:
                left = mid
            else:
                right = mid

        if sorted_citations[left] >= n - left: return n - left
        if sorted_citations[right] >= n - right: return n - right

        return 0
