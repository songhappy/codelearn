1. when to use 滑动窗口(90%)
   时间复杂度要求O(n) (80%)
   要求原地操作，只可以使用交换，不能使用额外空间(80%)
   有子数组subarray/子字符串 substring 的关键词(50%)
   有回文palindrome 关键词(50%)
2. complexity time complexity O(n), 每一个干几次。space complexity O(1) 只需要两个指针的额外内存
3. template

```python
# 双向指针（parition in quicksort)
def partition(self, A, start, end):
    if start > end:
        return 
    left, right = start, end
    # key1: pivot is the value, not the index
    pivot = A[(start + end) // 2]
    # key 2, every time compare left & right, it should be left <= right, not left < right
    while left < right:
        while left <= right and A[left] < pivot:
            left += 1
        while left <= right and A[right] >= pivot:
            right += 1
        if left <= right:
            A[left], A[right] = A[right], A[left]
# 背向指针
left = position 
right = position + 1
while left >= 0 and right < len(s):
    if left 和 right 可以停下来了：
        break
    left -= 1
    right += 1

# 同向指针
j = 0
for i in range(n):
    # 不满足则循环到满足搭配为止
    while j < n and i 到 j 之间不满足条件：
        j += 1
    if i 到 j 之间满足条件：
        处理i 到 j 这段区间

# 合并双指针
def merge(list1, list2):
    new_list = []
    i, j = 0, 0
    # 合并的过程只能操作i,j 的移动，不能pop 等改变list1, list2 
    while i < len(list1) and j < len(list2):
        if list[i] < list1[j]:
            new_list.append(list1[i])
            i += 1
        else:
            new_list.append(list2[j])
            j += 1
    # 合并剩下的到new_list 里
    # 不要用new_list.extend(list[i:] 之类的方法，因为list1[i:] 会产生额外空间消耗
    while i < len(list1):
        new_list.append(list1[i])
        i += 1
    while j < len(list2):
        new_list.append(list2[j])
        j += 1
    return  new_list
```

4. exmaples

5.tips 1. always remember to move pointer after finish the work

# online algorithm

数据只遍历一次 需要得到一些中间结果 不停改变数据集

# offline algorithm

数据集一开始是给定 随便遍历几次都可以 最后得到一个结果就可以

