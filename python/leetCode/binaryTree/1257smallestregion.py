class Solution:
    def findSmallestRegion(self, regions, region1, region2):
        # put everything and relationships into ancestorRef map
        ancestorsRef = {}
        for line in regions:
            ancestorsRef[line[0]] = None
        for line in regions:
            for ele in line[1:]:
                ancestorsRef[ele] = line[0]
        print(ancestorsRef)

        # put ancestors or region1 into a stack of ancestors
        ancestors1 = [];
        ancestors2 = []
        while region1:
            ancestors1.append(region1)
            ancestor = ancestorsRef[region1]
            region1 = ancestor

        while region2:
            ancestors2.append(region2)
            ancestor = ancestorsRef[region2]
            region2 = ancestor

        print(ancestors1)
        print(ancestors2)

        while ancestors1 and ancestors2:
            ancestor1 = ancestors1[-1]
            ancestor2 = ancestors2[-1]
            if ancestor1 == ancestor2:
                ancestor = ancestors1.pop()
                ancestor = ancestors2.pop()
            else:
                break
        return ancestor


if __name__ == '__main__':
    a = Solution()
    regions = [["Earth", "North America", "South America"],
               ["North America", "United States", "Canada"],
               ["United States", "New York", "Boston"],
               ["Canada", "Ontario", "Quebec"],
               ["South America", "Brazil"]]
    region1 = "Quebec"
    region2 = "New York"
    out = a.findSmallestRegion(regions, region1, region2)
    print(out)
