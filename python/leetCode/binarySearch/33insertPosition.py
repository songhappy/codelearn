class Solution:
    def searchInsert(self, nums, target: int) -> int:
        left = 0
        right = len(nums) - 1
        while (left + 1 < right):
            mid = int((left + right) / 2)
            if (target <= nums[mid]):
                right = mid
            else:
                left = mid

        if (target <= nums[left]): return left
        if (target > nums[left] and target <= nums[right]): return right
        if (target > nums[right]): return right + 1
