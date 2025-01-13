1. always compare nums[i] <> target, put nums[i] at first in comparison
2. comparison in the loop
   [mid] > target, right = mid
   [mid] < target, left = mid if first position
   [mid] = target, right = mid
   [left] = target, return left
   [right] = target, return right if last position
   [mid] = target, left = mid
   [right] = target, return right
   [left] = target, return left

3. 2d matrix, do not think about x, y, and coordinate, think i in row, j in column n600 smallest
   rectangel
4. 
