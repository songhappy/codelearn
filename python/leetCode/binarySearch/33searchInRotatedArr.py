class Solution:
    def search(self, nums, target: int) -> int:
        if nums is None or len(nums) == 0: return -1
        left = 0;
        right = len(nums) - 1
        while (left + 1 < right):
            mid = left + (right - left) // 2
            if nums[mid] > nums[left]:
                if nums[mid] >= target and nums[left] <= target:
                    right = mid
                else:
                    left = mid
            else:
                if nums[mid] <= target and nums[right] >= target:
                    left = mid
                else:
                    right = mid
        if nums[left] == target: return left
        if nums[right] == target: return right

        return -1
