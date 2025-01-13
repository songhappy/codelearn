class Solution0(object):
    def twoSum(self, nums, target):
        dict = {}
        for i in range(len(nums)):
            diff = target - nums[i]
            if diff in dict.keys():
                return [i, dict[diff]]
            dict[nums[i]] = i
        return []


# 170
class TwoSum:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.counter = {}

    def add(self, number: int) -> None:
        """
        Add the number to an internal data structure..
        """
        self.counter[number] = self.counter.get(number, 0) + 1

    def find(self, value: int) -> bool:
        """
        Find if there exists any pair of numbers which sum is equal to the value.
        """
        flag = False
        for ele in self.counter:
            if ele + ele == value:
                if self.counter[ele] >= 2:
                    return True
            elif value - ele in self.counter:
                return True

        return False


# 608 sorted, does not care about duplicates
class Solution1:
    """
    @param nums: an array of Integer
    @param target: target = nums[index1] + nums[index2]
    @return: [index1 + 1, index2 + 1] (index1 < index2)
    """

    def twoSum(self, nums, target):
        # write your code here
        left, right = 0, len(nums) - 1
        while left < right:
            sumtwo = nums[left] + nums[right]
            if sumtwo == target:
                return [left + 1, right + 1]
            elif sumtwo > target:
                right -= 1
            else:
                left += 1
        return []


# 587, only two pointers
class Solution:
    """
    @param nums: an array of integer
    @param target: An integer
    @return: An integer
    """

    def twoSum6(self, nums, target):
        pairs = []
        nums.sort()

        left, right = 0, len(nums) - 1
        while left < right:
            the_sum = nums[left] + nums[right]
            if the_sum == target:
                pairs.append((nums[left], nums[right]))
                left += 1
                right -= 1
                while left < right and nums[left] == nums[left-1]:
                    left = left + 1
                while left < right and nums[right] == nums[right + 1]:
                    right = right - 1
            elif the_sum > target:
                right -= 1
            else:
                left += 1
        return len(list(set(pairs)))  # remove duplicate
