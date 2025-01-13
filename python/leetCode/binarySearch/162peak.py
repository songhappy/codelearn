class Solution:
    def findPeakElement(self, nums) -> int:
        # do not have to compare with num[i-1] since num[i] and num[i+1] are enough to make conclusion.
        if (not nums or len(nums) == 0): return -1

        left = 0;
        right = len(nums) - 1
        while (left + 1 < right):
            mid = (left + right) // 2
            if nums[mid] <= nums[mid + 1]:
                left = mid
            else:
                right = mid

        return right if nums[left] < nums[right] else left

        return -1
