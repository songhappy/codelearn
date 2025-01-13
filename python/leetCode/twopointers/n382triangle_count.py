class Solution:
    """
    @param S: A list of integers
    @return: An integer
    """

    def triangleCount(self, S):
        # write your code here
        S.sort()
        counter = 0
        for i in range(1, len(S) - 1):
            longest = S[-i]
            left, right = 0, len(S) - i - 1

            while left < right:
                if longest < S[left] + S[right]:
                    counter = counter + (right - left)
                    right -= 1
                else:
                    left += 1
        return counter
