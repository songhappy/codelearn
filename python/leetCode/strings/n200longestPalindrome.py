class Solution:
    """
    @param s: input string
    @return: the longest palindromic substring
    """

    def get_palindrome(self, s, left, right):
        while left >= 0 and right < len(s):
            if s[left] == s[right]:
                left -= 1
                right += 1
            else:
                break
        return s[left + 1:right]

    def longestPalindrome(self, s):
        # write your code here
        if not s:
            return ""

        longest = ""
        for mid in range(len(s)):
            sub = self.get_palindrome(s, mid, mid)
            if len(sub) > len(longest):
                longest = sub
            sub = self.get_palindrome(s, mid, mid + 1)
            if len(sub) > len(longest):
                longest = sub
        return longest
