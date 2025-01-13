class Solution:
    """
    @param image: a binary matrix with '0' and '1'
    @param x: the location of one of the black pixels
    @param y: the location of one of the black pixels
    @return: an integer
    """

    def minArea(self, image, x, y):
        # write your code here
        if image is None or len(image) == 0: return -1
        if len(image[0]) == 0: return -1
        m = len(image);
        n = len(image[0])

        def checkColumn(image, j):
            for i in range(m):
                if image[i][j]: return True
            return False

        def checkRow(image, i):
            for j in range(n):
                if image[i][j]: return True
            return False

        # search min x
        left = 0;
        right = x
        while (left + 1 < right):
            mid = left + (right - left) // 2
            if checkRow(image, mid):
                right = mid
            else:
                left = mid
        minx = left if checkRow(image, left) else right

        # search max x
        left = x
        right = m - 1
        while (left + 1 < right):
            mid = left + (right - left) // 2
            if checkRow(image, mid):
                left = mid
            else:
                right = mid
        maxx = right if checkRow(image, right) else left
        # search top
        left = 0;
        right = y
        while (left + 1 < right):
            mid = left + (right - left) // 2
            if checkColumn(image, mid):
                right = mid
            else:
                left = mid
        miny = left if checkColumn(image, left) else right

        # search bottum
        left = y;
        right = n - 1
        while (left + 1 < right):
            mid = left + (right - left) // 2
            if checkColumn(image, mid):
                left = mid
            else:
                right = mid
        maxy = right if checkColumn(image, right) else left
        print(minx, maxx, miny, maxy)
        return (maxx - minx + 1) * (maxy - miny + 1)


if __name__ == '__main__':
    s = Solution()
    image = [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    image2 = [[0, 0, 1, 0],
              [0, 1, 1, 0],
              [0, 1, 0, 0]]
    x = 0;
    y = 2
    res = s.minArea(image, x, y)
    print(res)
