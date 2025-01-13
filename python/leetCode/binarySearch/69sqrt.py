class Solution:
    # find the last position that i * i <= x
    def mySqrt(self, x: int) -> int:
        left, right = 1, x
        while (left + 1 < right):
            mid = left + (right - left) // 2
            if mid * mid > x:
                right = mid
            if mid * mid <= x:
                left = mid
        if right * right <= x: return right
        if left * left <= x: return left

        return 0
