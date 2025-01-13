class Solution:
    """
    @param numbers: Give an array numbers of n integer
    @return: Find all unique triplets in the array which gives the sum of zero.
    """

    # remove duplidate at the end of a loop, after left += 1 or right -= 1, do not do other stuff, otherwise, index out of range
    def threeSum(self, numbers):
        self.result = []
        numbers.sort()
        right = len(numbers) - 1
        for i in range(len(numbers) - 2):
            if i > 0 and numbers[i] == numbers[i - 1]:
                continue
            left = i + 1
            diff = -numbers[i]
            self.two_sum(numbers, diff, left, right)
        return self.result

    def two_sum(self, numbers, target, l, r):
        while l < r:
            the_sum = numbers[l] + numbers[r]
            if the_sum == target:
                self.result.append([-target, numbers[l], numbers[r]])
                l += 1
                r -= 1
                while l < r and numbers[l] == numbers[l - 1]:
                    l += 1
                while l < r and numbers[r] == numbers[r + 1]:
                    r -= 1
            elif the_sum > target:
                r -= 1
            else:
                l += 1


class Solution1:
    """
    @param numbers: Give an array
    @param target: An integer
    @return: Find all unique quadruplets in the array which gives the sum of zero
    """

    def fourSum(self, numbers, target):
        # write your code here
        numbers.sort()
        length = len(numbers)
        res = []
        for i in range(length - 4):
            if i and numbers[i] == numbers[i - 1]:
                continue
            for j in range(i + 1, length - 3):
                if j > i + 1 and numbers[j] == numbers[j - 1]:
                    continue
                left, right = j + 1, length - 1
                while left < right:
                    the_sum = numbers[i] + numbers[j] + numbers[left] + numbers[right]
                    if the_sum == target:
                        res.append([numbers[i], numbers[j], numbers[left], numbers[right]])
                        left += 1
                        right -= 1
                        while left < right and numbers[left] == numbers[left - 1]:
                            left += 1
                        while left < right and numbers[right] == numbers[right + 1]:
                            right -= 1
                    elif the_sum > target:
                        right -= 1
                    else:
                        left += 1
        return res

    def fourSum1(self, numbers, target):
        # write your code here
        numbers.sort()
        length = len(numbers)
        res = []
        for i in range(length - 4):
            if i and numbers[i] == numbers[i - 1]:
                continue
            for j in range(i + 1, length - 3):
                if j > i + 1 and numbers[j] == numbers[j - 1]:
                    continue
                left, right = j + 1, length - 1
                diff = target - numbers[i] - numbers[j]

                def twoSum(nums, target, l, r):
                    pairs = []
                    while l < r:
                        the_sum = nums[l] + nums[r]
                        if the_sum == target:
                            pairs.append((nums[l], nums[r]))
                            l += 1
                            r -= 1
                            while l < r and nums[l] == nums[l - 1]:
                                l = l + 1 # remove duplicate
                            while l < r and nums[r] == nums[r + 1]:
                                r = r - 1 # remove duplicate
                        elif the_sum > target:
                            r -= 1
                        else:
                            l += 1
                    return pairs

                two_res = twoSum(numbers, diff, left, right)
                for pair in two_res:
                    res.append([[numbers[i], numbers[j]] + pair])

        return res
