1. when to use BFS graph traversal level order traversal connected component, 找到连通分量的个数啊,
   给一个点，一圈一圈的遍历 topological sorting shortest path in simple graph iteration solution for all
   possible results 给定一个变化规则，从初始状态到终止状态需要几步
2. time complexity time O(n + m)
   space o(n)
3. queue.Queue is intended for allowing different threads to communicate using queued messages/data,
   whereas collections.deque is simply intended as a datastructure. That's why Queue.Queue has
   methods like put_nowait(), get_nowait(), and join(), whereas collections.deque doesn't.
   Queue.Queue isn't intended to be used as a collection, which is why it lacks the likes of the in
   operator. It boils down to this: if you have multiple threads and you want them to be able to
   communicate without the need for locks, you're looking for Queue.Queue; if you just want a queue
   or a double-ended queue as a data structure, use collections.deque. deque.append(), deque.pop(),
   deque.popleft()
   queue.get(), queue.put()

template

```python

# graph, if not specified, build a graph
# in_degrees = {}
# visited/distance = {}

def bfs(start_node):
    # BFS 必须用queue s
    # distance(dict) 有两个作用，一个记录是否被丢进贵队列，避免重复访问，另外是记录start_node到所有几点的最短距离
    # 如果只要连通性，可以换成set 
    import collections
    queue = collections.deque([start_node])
    distance = {start_node: 0}
    # while 队列不空，不停从队列里拿出一个点，拓展邻居节点放到队列里
    while queue:
        node = queue.popleft()
        # 如果有明确的终点判断，这里加
        if node is last:
            break or return something 
        for neighbor in node.get_neighbors():
            if neighbor is in distance:
                continue 
            queue.append(neighbor)
            distance[neighbor] = distance[node] + 1
    # 如果需要返回所有点离起点的距离，return hashmap
    return distance
    # 如果需要返回所有连同的节点，return HashMap 里所有点
    return distance.keys()
    # 如果需要返回离终点的最短距离
    return distance[end_node]
```

```python
def get_indegrees(nodes):
    counter = {node: 0 for node in nodes}
    for node in nodes:
        for neighbor in node.get_neighbors():
            counter = counter[neighbor] + 1
    return counter

def top_sort(nodes):
    indegrees = get_indegrees(nodes)
    import collections
    queue = collections.deque([node for node in nodes if indegrees[node] == 0])
    top_order = []
    while queue:
        node = queue.popleft()
        top_order.append(node)
        for neightbor in node.get_neighbors():
            indegrees[neightbor] = indegrees[node] - 1
            if indegrees[neightbor] == 0:
                queue.append(neightbor)
    if len(top_order) != len(nodes):
        return 有循环依赖，没有top_order 
    return top_order
```

tips

4. by level VS no level 由点到面，每一次一个点，可以省一层循环
5. graph bfs different from tree bfs cause in graph you need check if it is visited or not, and
   {}.add(node) and queue.append(node) at the same time n433
6. careful when constructing a graph, could be a root node, could be matrix or dictionaries

7. toplogical sorted related, need build graph and calculate indegrees, and return an order, eg.
   n127, n615, n605 ask : any top_order n 127 is there any top_order n615, 611 the only top order,
   len(queue) > 1, then None n605

8. bfs related ask: bfs order, return the order n137, n242 shortest path, need update visited in bfs
   or return order. length in n120, distance in n611, visited in n433
