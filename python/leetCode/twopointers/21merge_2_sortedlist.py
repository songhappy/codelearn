# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[
        ListNode]:
        # 三个指针，一个dummy node, 两个指针指向两个list, curr 是用来merge 的，关键看cur.next 指向谁，谁小就指向谁。最后把剩下的连起来

        node1 = list1
        node2 = list2
        dummy = cur = ListNode()
        while node1 and node2:
            if node1.val <= node2.val:
                cur.next = node1
                node1 = node1.next
                cur = cur.next
            else:
                cur.next = node2
                node2 = node2.next
                cur = cur.next
        cur.next = node1 if node1 else node2
        return dummy.next


