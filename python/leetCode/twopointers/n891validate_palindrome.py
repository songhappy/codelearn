class Solution:
    """
    @param s: a string
    @return bool: whether you can make s a palindrome by deleting at most one character
    """

    # specifically break a loop, otherwise endless
    # memorizse two pointers loop

    def validPalindrome(self, s):
        # Write your code here
        left, right = 0, len(s) - 1
        left, right = self.two_pointers(s, 0, len(s) - 1)  # find the place not equal
        if left >= right:
            return True

        return self.is_palindrome(s, left + 1, right) or self.is_palindrome(s, left, right - 1)

    def two_pointers(self, s, left, right):
        while left < right:
            if s[left] != s[right]:
                return left, right
            left += 1
            right -= 1
        return left, right

    def is_palindrome(self, s, left, right):
        left, right = self.two_pointers(s, left, right)
        return left >= right


if __name__ == '__main__':
    solution = Solution()
    solution.validPalindrome('abcfdcba')
