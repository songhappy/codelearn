#
# 算法：并查集
#
# 并查集是一种可以动态维护若干个不重叠的集合，并支持合并与查询两种操作的一种数据结构
#
# 一般我们建立一个数组fa 或者用map表示一个并查集，fai表示i的父节点。
#
# 初始化：每一个点都是一个集合，因此自己的父节点就是自己fa[i]=i
# 查询：每一个节点不断寻找自己的父节点，若此时自己的父节点就是自己，那么该点为集合的根结点，返回该点。
# 修改：合并两个集合只需要合并两个集合的根结点，即fa[RootA]=RootB，其中RootA,RootB是两个元素的根结点。
# 路径压缩：实际上，我们在查询过程中只关心根结点是什么，并不关心这棵树的形态(有一些题除外)。因此我们可以在查询操作的时候将访问过的每个点都指向树根，这样的方法叫做路径压缩，单次操作复杂度为O(logn),经过路径压缩后可将查询时间优化到O(1)
# 凡是这种动态的加入，查询某个集合的问题都可以考虑并查集
#
# 这题需要通过map来作为father并用owner记录拥有者，
# find数组中，当father[x]！=[x]时，不断循环查询，找到最小x,同时更新路径上的所有点的最小指向（路径压缩）来达到最快的查询速度
# connet中，若a,b未连接，通过father数组将a,b指向同一个块
# 我们通过并查集找到每个人拥有的邮箱，并将它们通过一个map存下来，用set去重，最后遍历这个map将答案输出
# 复杂度分析
#
# 时间复杂度O(n)
# 用了路径压缩，查询效率为O(1)
# 空间复杂度O(n)
# n为邮箱的数目
#

import collections


class Solution:
    """
    @param accounts: List[List[str]]
    @return: return a List[List[str]]
    """

    def __init__(self):
        self.father = {}

    def accountsMerge(self, accounts):
        email2set = collections.defaultdict(
            set)  # defaultdict, root email: email set, default 的value 是个empty set
        email2acct = {}  # email: account name

        # write your code here
        if not accounts:
            return []

        for account in accounts:
            if not account or len(account) < 2:
                continue
            for email in account[1:]:
                self.father[email] = email

        for account in accounts:
            if not account or len(account) <= 2:
                continue
            for i in range(2, len(account)):
                self.union(account[i - 1], account[i])  # 将每一个account 里面的emails union 起来

        for account in accounts:
            if not account:
                continue
            name = account[0]
            for email in account[1:]:
                root_email = self.find(email)  # 找到每一个email 的根节点
                email2set[root_email].add(email)  # 加到根节点为key：子节点们为value 的set 里面去
                if root_email not in email2acct:
                    email2acct[root_email] = name
        results = []
        for root_email in email2set:  # 通过root email 把名字和所有的子email set 串起来
            temp = [email2acct[root_email]] + sorted(list(email2set[root_email]))
            results.append(temp)
        return results

    def find(self, node):
        root = node
        while self.father[root] != root:  # 一直往上找，直到找到node 的root
            root = self.father[root]
        while node != root:  # 如果node != root，将father[node] 断开，然后 father[node]=root, 将node的father 指向root
            temp = self.father[node]  # 同时更新了这个father[node], 使其指向了最根root，让后查询比较快点
            self.father[node] = root
            node = temp
        return root

    def union(self, node1, node2):
        root1 = self.find(node1)
        root2 = self.find(node2)
        if root1 != root2:
            self.father[root2] = root1


if __name__ == '__main__':
    s = Solution()
    accounts = [["John", "johnsmith@mail.com", "john_newyork@mail.com"],
                ["John", "johnsmith@mail.com", "john00@mail.com"],
                ["Mary", "mary@mail.com"],
                ["John", "johnnybravo@mail.com"]]
    out = s.accountsMerge(accounts)
    print(out)
