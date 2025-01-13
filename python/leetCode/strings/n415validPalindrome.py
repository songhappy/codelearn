class Solution:
    """
    @param s: A string
    @return: Whether the string is a valid palindrome
    """

    def isPalindrome(self, s):
        # write your code here
        left = 0;
        right = len(s) - 1
        while left < right:
            while left < right and not s[left].isalpha() and not s[left].isdigit():
                left = left + 1
            while left < right and not s[right].isalpha() and not s[right].isdigit():
                right = right - 1
            if s[left].lower() == s[right].lower():
                left = left + 1
                right = right - 1
            else:
                return False
        return True
