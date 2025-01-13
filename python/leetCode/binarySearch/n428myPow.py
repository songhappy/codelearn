class Solution:
    """
    @param x {float}: the base number
    @param n {int}: the power number
    @return {float}: the result
    """

    def myPow1(self, x, n):
        # write your code here

        if x is None or x == 0: return 0
        if n == 0: return 1
        if n < 0:
            x = 1 / x
            n = -n

        left = 0;
        right = n
        res = 1
        tmp = x
        while n != 0:
            if n % 2 == 1: res = res * tmp
            tmp = tmp * tmp
            n = n // 2
        return res

    def myPow(self, x, n):
        # write your code here
        if x is None or x == 0: return 0
        if n == 0: return 1
        if n < 0:
            x = 1 / x
            n = -n

        if n % 2 == 1:
            tmp = self.myPow(x, n // 2)
            res = tmp * tmp * x
        else:
            tmp = self.myPow(x, n // 2)
            res = tmp * tmp

        return res
