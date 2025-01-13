Heap

1. when to use 找最大值最小值（60%） 要求O(nlogk) 找第K大(50%)
   要求logn 对数据操作(40%)

```python
class Heap:  #not clear now
    # python heapq, 
    import heapq
    def __init__(self):
        self.minheap = []
        self.deleted_set = set()
    
    def push(self, index, val):
        heapq.heappush(self.minheap, (val, index))
    
    def _lazy_deletion(self):
        while self.minheap and self.minheap[0][1] in self.deleted_set:
            heapq.heappop(self.minheap)
    
    def top(self):
        self._lazy_deletion()
        return self.minheap[0]

    def pop(self):
        self._lazy_deletion()
        heapq.heappop(self.minheap)
    
    def delete(self, index):
        self.deleted_set.add(index)
    
    def is_empty(self):
        return not bool(self.minheap)
        
```

Union Find

1. when to use # not understand now 需要查询图的联通状况问题 需要支持快速合并两个集合
2. time complexity union O(1), find O(1), space complexity O(n)
3. template

```python
class UnionFind:
    def __init__(self):
        # 初始化父指针，集合大小，集合数量
        self.father = {}
        self.size_of_set = {}
        self.num_of_set = 0
    
    def add(self, other):
        # 如果点已经出现，操作无效
        if other in self.father:
            return 
        # 初始化点的父亲为空对象None
        # 初始化该点所在集合大小为1
        # 集合数量 + 1
        self.father[other] = None 
        self.num_of_set += 1
        self.size_of_set[other] = 1
    
    def merge(self, x, y):
        # 找到两个节点的根
        root_x, root_y = self.find(x), self.find(y)
        # 如果不是同一个则链接
        if root_x != root_y:
            # 将一个点的根变成新的根
            # 集合数量减少 1
            # 计算新的根所在集合大小
            self.father[root_x] = root_y
            self.num_of_set -= 1
            self.size_of_set[root_y] += self.size_of_set[root_x]
        
    def find(self, x):
        # 指针root 指向北查找的点x, 不断找到root 的父亲，直到root 指向x 的根节点
        root = x
        while self.father[root] != None:
            root = self.father[root]
        #将路径上所有点指向根节点 root
        while x != root:
            original_father = self.father[x]
            self.father[x] = root 
            x = original_father
        return  root 
    
    def is_connected(self, x, y):
        return self.find(x) == self.find(y)
    
    def get_sum_set(self):
        return  self.num_of_set
    
    def get_size_of_set(self, x):
        # 获得某个点所在集合大小
        return self.size_of_set[self.find(x)]

```

Trie
