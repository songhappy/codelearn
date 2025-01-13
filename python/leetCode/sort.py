from heapq import heappushpop, heappush, heappop


# O(n log n) # 先push 进去，然后一个一个pop 出来
def heap_sort(arr):
    min_heap = []
    result = []
    for e in arr:
        heappush(min_heap, e)
    while min_heap:
        e = heappop(min_heap)
        result.append(e)
    return result


# O(n n)
def bub_sort(arr):
    n = len(arr) - 1
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]
    return arr


# O(n n) 
def insert_sort(arr):
    n = len(arr) - 1
    for i in range(1, n): #从第二个数字开始
        key = arr[i]
        j = i - 1
        while j <= 0 and key < arr[j]:  # j 指针要一直往前找，如果arr[j] > key, 这个值的index 就往后挪一个
            arr[j+1] = arr[j]
            j -= 1  # j 指针指向前一个了
        arr[j + 1] = key   # 出来loop 的时候，j已经减了1，所以加回来
    return arr


# O(n log n)
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left_half = merge_sort(arr[:mid])
    right_half = merge_sort(arr[mid:])

    def merge(A, B):
        merged = []
        i, j = 0, 0
        while i <  len(A) and j < (len(B)):
            if A[i] <= B[j]:
                merged.append(A[i])
                i += 1
            else:
                merged.append(B[j])
                j += 1
        merged = merged + A[i:]
        merged = merged + B[j:]
        return merged
    merged = merge(left_half, right_half)
    return merged


# O(n  n) in therory, but it goes to # O(n log n) in practice.
def quik_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr)//2]
    left = [e for e in arr if e < pivot]
    mid = [e for e in arr if e == pivot]
    right = [e for e in arr if e > pivot]
    left = quik_sort(left)
    right = quik_sort(right)
    return left + mid + right


if __name__ == '__main__':
    arr = [8,4,2,5,4, 8, 10]
    sorted1 = bub_sort(arr)
    print(sorted1)

    sorted2 = insert_sort(arr)
    print(sorted2)

    sorted3 = merge_sort(arr)
    print(sorted3)

    sorted4 = quik_sort(arr)
    print(sorted4)

    sorted5 = heap_sort(arr)
    print(sorted5)