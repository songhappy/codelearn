import heapq

class MedianFinder:
    def __init__(self):
        self.min_heap = []  # Stores larger half of elements  # put one more if odd number
        self.max_heap = []  # Stores smaller half of elements

    def addNum(self, num):
        if len(self.max_heap) == len(self.min_heap):
            # Add the new element to the max_heap, then pop to min_heap,
            # so eventually max_heap has one more element than min_heap
            poped = -heapq.heappushpop(self.max_heap, -num)
            heapq.heappush(self.min_heap, poped)
        else: # means min_heap has one more element than max_heap
            # Add the new element to the max_heap eventually
            poped = heapq.heappushpop(self.min_heap, num)
            heapq.heappush(self.max_heap, -poped)

    def findMedian(self):
        if len(self.max_heap) == len(self.min_heap):
            # If the number of elements is even, the median is the average of the top elements from both heaps
            return (self.min_heap[0] - self.max_heap[0]) / 2.0
        else:
            # If the number of elements is odd, the median is the top element of the min_heap
            return float(self.min_heap[0])


from sortedcontainers import SortedList
class MedianFinder2:
    def __init__(self):
        self.arr = SortedList([])

    def addNum(self, num: int) -> None:
        self.arr.add(num)

    def findMedian(self) -> float:
        n = len(self.arr)
        if (n % 2 == 0):
            return (self.arr[n // 2] + self.arr[n // 2 - 1]) / 2
        else:
            return (self.arr[n // 2])