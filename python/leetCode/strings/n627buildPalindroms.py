class Solution:
    """
    @param s: a string which consists of lowercase or uppercase letters
    @return: the length of the longest palindromes that can be built
    """

    # hash to save characters which could not be used to build palindromes and want to be removed, odd frequencies
    def longestPalindrome(self, s):
        # write your code here
        hash = {}
        for c in s:
            if c in hash:
                del hash[c]
            else:
                hash[c] = True

        remove = len(hash)
        if remove > 0:
            remove -= 1

        return len(s) - remove
