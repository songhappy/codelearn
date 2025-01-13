import sys


class Solution:
    """
    @param nums: an integer array
    @param target: An integer
    @return: the difference between the sum and the target
    """

    # always remenber move pointer after finish the work

    def twoSumClosest(self, nums, target):
        nums.sort()
        left, right = 0, len(nums) - 1
        diff = sys.maxsize
        while left < right:
            if nums[left] + nums[right] < target:
                diff = min(diff, target - nums[left] - nums[right])
                left += 1
            else:
                diff = min(diff, nums[left] + nums[right] - target)
                right -= 1

        return diff


class Solution1:
    """
    @param numbers: Give an array numbers of n integer
    @param target: An integer
    @return: return the sum of the three integers, the sum closest target.
    """

    def threeSumClosest(self, numbers, target):
        numbers.sort()
        res = None
        for i in range(len(numbers) - 1):
            left, right = i + 1, len(numbers) - 1
            while left < right:
                the_sum = numbers[i] + numbers[left] + numbers[right]
                if res is None or abs(the_sum - target) < abs(res - target):
                    res = the_sum
                if the_sum > target:
                    right -= 1
                else:
                    left += 1
        return res
