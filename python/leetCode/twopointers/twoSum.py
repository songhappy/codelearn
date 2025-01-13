class Solution(object):
    def twoSum(self, nums, target):
        dict = {}
        for i in range(len(nums)):
            diff = target - nums[i]
            if diff in dict.keys():
                return [i, dict[diff]]
            dict[nums[i]] = i
        return []
