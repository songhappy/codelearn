class Solution:
    def search(self, reader, target):
        """
        :type reader: ArrayReader
        :type target: int
        :rtype: int
        """
        # define boundary
        left = 0;
        right = 1
        while (target > reader.get(right)):
            left = right
            right = 2 * right

        # binary search
        while (left + 1 < right):
            mid = (left + right) // 2
            if (reader.get(mid) < target):
                left = mid
            else:
                right = mid
        if (target == reader.get(left)): return left
        if (target == reader.get(right)): return right
        return -1
