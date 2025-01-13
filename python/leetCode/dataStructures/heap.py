# since the heappush heappop matains a min heap
# min_heapify
# build_min_heap
# insert
# extract_min
# heapsort
# 1. when to use 找最大值最小值（60%） 要求O(nlogk) 找第K大(50%)
#   要求logn 对数据操作(40%)

import heapq

class MinHeap1:  # not clear now

    def __init__(self):
        self.heap = []

    def parent(self,i):
        return (i-1) //2

    def left(self, i):
        return 2 * i + 1

    def right(self, i):
        return 2 * i + 2

    def _swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def _sift_up(self, i):  # ith value sift_up
        while i > 0 and self.heap[i] < self.heap[self.parent(i)]:
            self._swap(i, self.parent(i))
            i = self.parent(i)

    def _sift_down(self, i):  # ith value sift_down, maintain min property
        n = len(self.heap)
        while i < n: # 一直下，一直要下到底
            min_index = i
            left = self.left(i)
            right = self.right(i)

            if left < n and self.heap[left] < self.heap[min_index]:
                min_index = left

            if right < n and self.heap[right] < self.heap[min_index]:
                min_index = right

            if min_index != i:
                self._swap(i, min_index)
                i = min_index
            else:
                break

    def insert(self, num):
        self.heap.append(num)
        self._sift_up(len(self.heap) - 1)

    def delete(self, i):
        self.heap[i] = self.heap[len(self.heap)-1]
        self.heap.pop()
        self._sift_down(i)

    # def delete(self, value):
    #     if value in self.heap:
    #         index = self.heap.index(value)
    #         self._swap(index, len(self.heap) - 1)
    #         self.heap.pop()
    #         self._sift_down(index)

    def heapify(self, arr):
        self.heap = arr
        n = len(self.heap)
        for i in range(n // 2 - 1, -1, -1):
            self._sift_down(i)


def heap_sort1(arr):
    heap = []
    for element in arr:
        heapq.heappush(heap, element)

    sorted_arr = []
    while heap:
        sorted_arr.append(heapq.heappop(heap)) #heappop pop the smallest value
    return sorted_arr


class MinHeap2:
    def __init__(self):
        self.heap = []

    def insert(self, value):
        heapq.heappush(self.heap, value)

    def delete(self, value):
        self.heap.remove(value)
        heapq.heapify(self.heap)

    def heapify(self, arr):
        self.heap = arr[:]
        heapq.heapify(self.heap)

    def heap_sort(self, arr):
        sorted = []
        self.heapify(arr)
        while self.heap:
            self.heap[0], self.heap[len(self.heap) -1] = self.heap[len(self.heap) -1], self.heap[0]
            min = self.heap.pop()
            sorted.append(min)
            self.heapify(self.heap)
        return sorted


if __name__ == '__main__':
    arr = [1,3,2,5,7,4]
    print(arr)
    heap = MinHeap2()
    sorted = heap.heap_sort(arr)
    print(sorted)
    print(arr)
    sorted1 = heap_sort1(arr)
    print(sorted1)