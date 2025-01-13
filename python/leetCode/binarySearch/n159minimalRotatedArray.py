class Solution:
    """
    @param nums: a rotated sorted array
    @return: the minimum number in the array
    """

    def findMin(self, nums):
        # write your code here
        if nums is None or len(nums) == 0: return -1
        left = 0;
        right = len(nums) - 1
        target = nums[-1]
        while left + 1 < right:
            mid = left + (right - left) // 2
            if nums[mid] > target:
                left = mid
            else:
                right = mid

        return min(nums[left], nums[right])
