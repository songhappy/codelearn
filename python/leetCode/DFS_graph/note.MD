1. when to use 找满足某个条件的所有方案（99%） 二叉树问题（90%） 组合问题（95%） 求所有满足条件的组合 判断条件：组合中的元素是顺序无关的 排练问题（95）
   求所有满足条件的排列 判断条件：组合中的元素是顺序相关的
2. when not to use 用BFS的
3. complexity O(# of solutions * time to get each solution)
   树的遍历： O(N)
   排列问题：O(n!*n)
   组合问题： （2^n * n)
4. template def dfs(参数列表)： if 递归出口： 记录答案 return for 所有的拆解可能性： 修改所有的参数： dfs(参数列表)
   还原所有被修改过的参数 return something 如果需要，很多时候不需要，除了divide and conquer

5. examples n67 n652
   
