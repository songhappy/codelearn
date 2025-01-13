# dependencies=[[a,b],[b,c],[d,e],[c,e]], n=5
# output [e,d,c,b,a]

import collections

class Solution:
    def top_sort(dependencies, n):


      char_to_int = {chr(i + 97): i for i in range(n)}
      int_to_char = {i: chr(i + 97) for i in range(n)}


      lnks = [[] for _ in range(n)]
      indegree = [0] * n

      for x in dependencies:
          u = char_to_int[x[0]]
          v = char_to_int[x[1]]
          lnks[u].append(v)
          indegree[v] += 1

      q = queue.Queue()
      for i in range(n):
          if indegree[i] == 0:
              q.put(i)

      ret = []

      while not q.empty():
          node = q.get()
          ret.append(node)
          for x in lnks[node]:
              indegree[x] -= 1
              if indegree[x] == 0:
                  q.put(x)

      ret.reverse()

      return [int_to_char[node] for node in ret]
                
     
if __name__ == '__main__':
    s = Solution()
    dependencies = [['a', 'b'], ['b', 'c'], ['d', 'e'], ['c', 'e']]
    n = 5
    result = s.top_sort(dependencies)
    print(result)