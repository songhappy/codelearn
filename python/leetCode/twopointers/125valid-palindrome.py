class Solution:
    def isPalindrome(self, s: str) -> bool:
        left, right = 0, len(s) - 1
        while left < right:
            while left < right and not s[left].isalpha() and not s[left].isdigit():
                left = left + 1
            while left < right and not s[right].isalpha() and not s[right].isdigit():
                right = right - 1
            if left < right and s[left].lower() != s[right].lower():
                return False
            left = left + 1
            right = right - 1
        return True
