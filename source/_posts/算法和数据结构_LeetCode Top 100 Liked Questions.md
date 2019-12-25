---
title: LeetCode Top 100 Liked Questions
date: 2019-04-02 10:17:39
tags: 
- LeetCode
categories: 算法和数据结构
---
[[toc]]
# LeetCode 198. House Robber
原题：https://leetcode.com/problems/house-robber/
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.

Example 1:

Input: [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
             Total amount you can rob = 1 + 3 = 4.
Example 2:

Input: [2,7,9,3,1]
Output: 12
Explanation: Rob house 1 (money = 2), rob house 3 (money = 9) and rob house 5 (money = 1).
             Total amount you can rob = 2 + 9 + 1 = 12.

分析：动态规划
```python
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        N = len(nums)
        dp = [0]*N
        dp[0] = nums[0]
        for i in range(1,N):
            dp[i] = max(dp[i-1],dp[i-2]+nums[i])
        return dp[N-1]
```
Runtime: 24 ms, faster than 35.84% of Python

# LeetCode 62. Unique Paths
原题：https://leetcode.com/problems/unique-paths/
A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?
![](/images/lc62.png)

分析：动态规划
需要注意的地方：
- 第0行和第0列的dp值都是1
- python初始化二维数组的方法：
`matrix = [[0 for j in range(cols)] for i in range(rows)]`

```python
class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        if m==0 or n==0:
            return 0
        if m==1 or n==1:
            return 1
        dp = [[0 for i in range(n)] for i in range(m) ]
        dp[0][0] = 0
        dp[0][1] = 1
        dp[1][0] = 1
        for i in range(1,m):
            dp[i][0] = 1
        for i in range(1,n):
            dp[0][i] = 1
        for i in range(1, m):
            for j in range(1, n):
                x = 0
                y = 0
                if i-1>=0 and i-1<m:
                    x = dp[i-1][j]
                if j-1>=0 and j-1<n:
                    y = dp[i][j-1]
                dp[i][j] = x+y
        return dp[m-1][n-1]
        
```
Runtime: 24 ms, faster than 34.59% 

# LeetCode 64. Minimum Path Sum
原题：https://leetcode.com/problems/minimum-path-sum/
Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which minimizes the sum of all numbers along its path.

Note: You can only move either down or right at any point in time.

Example:

Input:
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
Output: 7
Explanation: Because the path 1→3→1→1→1 minimizes the sum.
分析：动态规划，和LeetCode 62.是一样的，先判断0和1，再把第0行和第0列的dp设置初始值（这里和LeetCode 62.稍有不同），然后给从[1][1]开始的dp依次赋值

```python
class Solution(object):
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        if not grid:
            return -1
        M = len(grid)
        N = len(grid[0])
        if M==1 and N==1:
            return grid[0][0]
        dp = [[0 for j in range(N)] for i in range(M)]
        dp[0][0] = grid[0][0]
        for j in range(1,N):
                dp[0][j] = dp[0][j-1]+grid[0][j]
        if M==1:
            return dp[0][N-1]
        for i in range(1,M):
                dp[i][0] = dp[i-1][0]+grid[i][0]
        if N==1:
            return dp[M-1][0]
        for i in range(1, M):
            for j in range(1, N):
                dp[i][j] = min(dp[i-1][j], dp[i][j-1])+grid[i][j]
        return dp[M-1][N-1]
```
Runtime: 164 ms, faster than 13.63%

# LeetCode 96. Unique Binary Search Trees
原题：https://leetcode.com/problems/unique-binary-search-trees/
Given n, how many structurally unique BST's (binary search trees) that store values 1 ... n?

Example:

Input: 3
Output: 5
Explanation:
Given n = 3, there are a total of 5 unique BST's:
```

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3

分析：这里有规律，
以1为节点，则left subtree只能有0个节点，而right subtree有2, 3两个节点。所以left/right subtree一共的combination数量为：f(0) * f(2) = 2
dp[2] =  dp[0] * dp[1]　　　(1为根的情况)

　　　　+ dp[1] * dp[0]　　  (2为根的情况)

同理可写出 n = 3 的计算方法：

dp[3] =  dp[0] * dp[2]　　　(1为根的情况)

　　　　+ dp[1] * dp[1]　　  (2为根的情况)

 　　　  + dp[2] * dp[0]　　  (3为根的情况)
```
```python
class Solution(object):
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n==0 or n==1:
            return 1
        dp = [0]*(n+1)
        dp[0] = 1
        dp[1] = 1
        for i in range(2, n+1):
            for j in range(i):
                dp[i]+=dp[j]*dp[i-1-j] #这里是dp[i]，注意
        return dp[n]

```
# LeetCode 494. Target Sum
原题：https://leetcode.com/problems/target-sum/
You are given a list of non-negative integers, a1, a2, ..., an, and a target, S. Now you have 2 symbols + and -. For each integer, you should choose one from + and - as its new symbol.

Find out how many ways to assign symbols to make sum of integers equal to target S.

Example 1:
Input: nums is [1, 1, 1, 1, 1], S is 3. 
Output: 5
Explanation: 

-1+1+1+1+1 = 3
+1-1+1+1+1 = 3
+1+1-1+1+1 = 3
+1+1+1-1+1 = 3
+1+1+1+1-1 = 3

There are 5 ways to assign symbols to make the sum of nums be target 3.
Note:
The length of the given array is positive and will not exceed 20.
The sum of elements in the given array will not exceed 1000.
Your output answer is guaranteed to be fitted in a 32-bit integer.

分析：动态规划+0-1背包
首先进行数学分析，赋予标号后，集合中包含负数这正数，则有 sum(P)-sum(N) = S。因为sum(P)+sum(N)=sum(nums)，则有2*sum(P)=sum(nums)+S，故sum(P)=(sum(nums)+S)/2，由于target = (sum(nums)+S)/2是固定的整数，所以只需要找到和为它的组合数即可，题目就变成了：从nums中（nums中的元素为非负数）找到和为target的子集。
然后呢，转换的问题里用到了0-1背包问题，也就是，对于nums中的每个元素，都有选择或者不选择它两种情况，状态转移方程 F(i,C) = max{F(i-1, C), v(i) + F(i-1, C-w(i))}
```python
class Solution(object):
    def findTargetSumWays(self, nums, S):
        if sum(nums)<S:
            return 0
        if (S+sum(nums))%2==1:
            return 0
        target = (S+sum(nums))//2
        dp = [0]*(target+1)
        dp[0] = 1
        for num in nums:
            for i in range(target, num-1, -1):
                dp[i] += dp[i-num]
        return dp[-1]
```
Runtime: 80 ms, faster than 95.04% 

# LeetCode 416. Partition Equal Subset Sum
原题：https://leetcode.com/problems/partition-equal-subset-sum/
Given a non-empty array containing only positive integers, find if the array can be partitioned into two subsets such that the sum of elements in both subsets is equal.

Note:

Each of the array element will not exceed 100.
The array size will not exceed 200.
 

Example 1:

Input: [1, 5, 11, 5]

Output: true

Explanation: The array can be partitioned as [1, 5, 5] and [11].
 

Example 2:

Input: [1, 2, 3, 5]

Output: false

Explanation: The array cannot be partitioned into equal sum subsets.

分析：如果可以将nums分成两个和相等的部分，那么sum(nums)一定是个偶数，作为判断条件可以排除为技术的情况。
那么问题就变成了：从nums中选择若干元素，是否能组成和为target = sum(nums)//2。这是一个经典的0-1背包问题。

可以设置dp[i][j]表示：从nums中选择i个元素，其和为j。首先，dp[0][0]=True，由于当背包为0时肯定能组成，所以dp[][0]=True，而当背包不为0，元素个数为0时必为False，所以dp[0][]=False，然后i和j从1开始给dp赋值。

```python
class Solution(object):
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if not nums:
            return False
        if sum(nums)%2!=0:
            return False
        target = sum(nums)//2
        N = len(nums)
        dp = [[False for j in range(target+1)] for i in range(N)]
        dp[0][0] = True
        for i in range(1, N):
            dp[i][0] = True
        for i in range(1, target+1):
            dp[0][i] = False
        for i in range(1, N):
            for j in range(1, target+1):
                if (j-nums[i])<=target: #这里你自己想想就知道了
                # ，因为你需要考虑添加这个元素时，那么背包就变成了j-nums[i]，但它是作为dp中的下标的，不能越界
                    dp[i][j] = dp[i-1][j] or dp[i-1][j-nums[i]]
                else:
                    dp[i][j] = dp[i-1][j]
        return dp[N-1][target]
```
Runtime: 2092 ms, faster than 14.93% 

另一种简化的写法：只使用一维dp数组，dp[j]表示从数组中任意取数字的和能不能构成j。状态转移方程就是忽略掉二维数组的第一个维度即可，即
`dp[j] = dp[j] || dp[j - nums[i]]`

还要说一下，为什么需要从后向前更新dp，这是因为每个位置依赖与前面的一个位置加上nums[i]，如果我们从前向后更新的话，那么dp[i - 2]会影响dp[i - 1]，然后dp[i - 1]接着影响dp[i]，即同样的一个nums[i]被反复使用了多次，结果肯定是不正确的。但是从后向前更新就没有问题了。

```python
class Solution(object):
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if not nums:
            return False
        if sum(nums)%2!=0:
            return False
        target = sum(nums)//2
        N = len(nums)
        dp = [False for i in range(target+1)]
        dp[0] = True
        for num in nums:
            for j in range(target, num-1, -1):
                dp[j] = dp[j] or dp[j-num]
        return dp[target]
```
Runtime: 712 ms, faster than 52.18%

# LeetCode 438. Find All Anagrams in a String
原题：https://leetcode.com/problems/find-all-anagrams-in-a-string/
Given a string s and a non-empty string p, find all the start indices of p's anagrams in s.

Strings consists of lowercase English letters only and the length of both strings s and p will not be larger than 20,100.

The order of output does not matter.

Example 1:
```
Input:
s: "cbaebabacd" p: "abc"

Output:
[0, 6]

Explanation:
The substring with start index = 0 is "cba", which is an anagram of "abc".
The substring with start index = 6 is "bac", which is an anagram of "abc".
```
Example 2:
```
Input:
s: "abab" p: "ab"

Output:
[0, 1, 2]

Explanation:
The substring with start index = 0 is "ab", which is an anagram of "ab".
The substring with start index = 1 is "ba", which is an anagram of "ab".
The substring with start index = 2 is "ab", which is an anagram of "ab".
```
分析：利用hash+slide window
先把字符串p中各元素存到hash表里，然后移动滑动窗口，这里由于字符串的匹配要求所以slide window的大小是固定的，每匹配一次，右边界右移一位，同时左边界也右移。这里用到python中一个hash匹配的技巧：Counter类，它可以获得Iterable对象中各元素的个数。

需要注意的是，不要每次移动窗口都新建一个Counter，会超时哒

Runtime: 188 ms, faster than 46.14%
```python
from collections import Counter
class Solution(object):
    def findAnagrams(self, s, p):
        N = len(s)
        P = len(p)
        res = []
        p_counter = Counter(p)
        s_counter = Counter(s[:P-1])
        for i in range(P-1, N):
            s_counter[s[i]] += 1
            idx = i+1-P
            if s_counter == p_counter:
                res.append(idx)
            s_counter[s[idx]] -= 1
            if s_counter[s[idx]] == 0:
                del s_counter[s[idx]]
        return res
```

# LeetCode 141. Linked List Cycle
原题：https://leetcode.com/problems/linked-list-cycle/
Given a linked list, determine if it has a cycle in it.

To represent a cycle in the given linked list, we use an integer pos which represents the position (0-indexed) in the linked list where tail connects to. If pos is -1, then there is no cycle in the linked list. 

Example 1:
```
Input: head = [3,2,0,-4], pos = 1
Output: true
Explanation: There is a cycle in the linked list, where tail connects to the second node.
```
![](/images/lc1411.png)
Example 2:

```
Input: head = [1,2], pos = 0
Output: true
Explanation: There is a cycle in the linked list, where tail connects to the first node.
```
![](/images/lc141_2.png)
Example 3:
```
Input: head = [1], pos = -1
Output: false
Explanation: There is no cycle in the linked list.
```
![](/images/lc141_3.png)

分析：判断链表是否有环
思路1，遍历链表并记录node，如果存在环那么肯定会有重复的node出现

Runtime: 1172 ms, faster than 5.13%
```python
    def hasCycle(self, head):
        visit = []
        cur = head
        while cur:
            if cur in visit:
                return True
            visit.append(cur)
            cur = cur.next
        return False
```

思路2，设置两个快慢指针，初始fast = head.next, slow=head，每次fast走两步，slow走一步，如果存在环那么fast一定会和slow相遇。如果没环则fast指向None节点停止。

注意：fast走到倒数第二个节点
![](/images/lc141.png)

Runtime: 48 ms, faster than 32.29%
```python
    def hasCycle(self, head):
        if not head.next:
            return False
        fast = head.next
        slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast==slow:
                return True
        return False
```
# LeetCode 155. Min Stack
原题：https://leetcode.com/problems/min-stack/
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

push(x) -- Push element x onto stack.
pop() -- Removes the element on top of the stack.
top() -- Get the top element.
getMin() -- Retrieve the minimum element in the stack.
Example:
```
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> Returns -3.
minStack.pop();
minStack.top();      --> Returns 0.
minStack.getMin();   --> Returns -2.
```
分析：用两个栈，一个栈存元素，一个栈存从栈底到此元素中每个元素的最小值
![](/images/lc155.png)
Runtime: 56 ms, faster than 55.69%
```python
    def __init__(self):
        self.stack = []
        self.minstack = []

    def push(self, x):
        self.stack.append(x)
        if not self.minstack:
            self.minstack.append(x)
        else:
            top = self.minstack[-1]
            if x<top:
                self.minstack.append(x)
            else:
                self.minstack.append(top)

    def pop(self):
        if self.stack:
            self.stack.pop(-1)
            self.minstack.pop(-1)

    def top(self):
        if self.stack:
            return self.stack[-1]

    def getMin(self):
        if self.minstack:
            return self.minstack[-1]
```
# LeetCode 234. Palindrome Linked List
原题：https://leetcode.com/problems/palindrome-linked-list/
Given a singly linked list, determine if it is a palindrome.

Example 1:
```
Input: 1->2
Output: false
```
Example 2:
```
Input: 1->2->2->1
Output: true
```
Follow up:
**Could you do it in O(n) time and O(1) space?**

分析：判断一个链表是不是回文链表
思路1，设置快慢两个指针遍历链表，当fast走到头时，slow是链表的中点。(画个图明确：如果是1->9->3则slow停在9，如果是1->2->3->4则slow停在2)
slow走时把元素存到栈里，到中点后就和栈中元素一一比对。
- 这里需要利用fast是不是None判断节点的奇偶个数，画个图看栈中情况

Runtime: 92 ms, faster than 21.82%
```python
    def isPalindrome(self, head):
        if not head or not head.next:
            return True
        fast = head.next
        slow = head
        stack = []
        while fast and fast.next:
            stack.append(slow.val)
            fast = fast.next.next
            slow = slow.next
        cur = slow.next #
        if fast: #如果是偶数个节点，需要把slow.val加到stack中去
            stack.append(slow.val)
        while cur:
            if cur.val!=stack[-1]:
                return False
            stack.pop(-1)
            cur = cur.next
        return True
```
思路2，如果在O(1)空间内如何实现？同样利用快慢两个指针找到中点slow，然后，将slow后面的[链表反转](http://note.youdao.com/noteshare?id=e4b44d43ee9bdffdc7b609bb403f8857)，反转到最后得到后半部分的头节点，然后就可以从两半部分各自的头节点比较元素啦

链表反转
![](/images/reverselist.png)

- 将slow.next是后半部分链表的第一个结点，可以把slow看成是pre，slow.next看成当前此node。
- 反转之前，需要把**slow.next=None**，否则会出现死循环，画图就看出来了。

Runtime: 112 ms, faster than 10.67%
```python
    def isPalindrome(self, head):
        if not head:
            return True
        if not head.next:
            return True
        fast = head.next
        slow = head
        cur = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        # slow.next是需要反转链表的第一个节点
        pre = slow
        node = slow.next
        slow.next = None # 注意这里
        # 链表翻转
        while node:
            nxt = node.next
            node.next = pre
            pre = node
            node = nxt
        # 两半部分链表的元素一一比对判断回文
        while cur:
            print(cur.val)
            if cur.val!= pre.val:
                return False
            cur = cur.next
            pre = pre.next
        return True
```
# LeetCode 160. Intersection of Two Linked Lists
原题：https://leetcode.com/problems/intersection-of-two-linked-lists/
Write a program to find the node at which the intersection of two singly linked lists begins.
For example, the following two linked lists:
![](/images/lc160_0.png)
begin to intersect at node c1.
Example 1:
![](/images/lc160_1.png)
```
Input: intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, skipB = 3
Output: Reference of the node with value = 8
Input Explanation: The intersected node's value is 8 (note that this must not be 0 if the two lists intersect). From the head of A, it reads as [4,1,8,4,5]. From the head of B, it reads as [5,0,1,8,4,5]. There are 2 nodes before the intersected node in A; There are 3 nodes before the intersected node in B.
```
Example 2:
![](/images/lc160_2.png)
```
Input: intersectVal = 2, listA = [0,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
Output: Reference of the node with value = 2
Input Explanation: The intersected node's value is 2 (note that this must not be 0 if the two lists intersect). From the head of A, it reads as [0,9,1,2,4]. From the head of B, it reads as [3,2,4]. There are 3 nodes before the intersected node in A; There are 1 node before the intersected node in B.
```
Example 3:
![](/images/lc160_3.png)
```
Input: intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
Output: null
Input Explanation: From the head of A, it reads as [2,6,4]. From the head of B, it reads as [1,5]. Since the two lists do not intersect, intersectVal must be 0, while skipA and skipB can be arbitrary values.
Explanation: The two lists do not intersect, so return null.
```
分析：找到两个链表的公共节点的第一个
[剑指Offer 36](https://yugutouoo.github.io/2019/04/01/%E5%89%91%E6%8C%87offer%E9%A2%98%E7%9B%AE%E7%B2%BE%E8%A7%A3/)原题
两个链表从第一个公共节点开始之后的节点都是相同的，因为节点的next指向应该也相同
![](/images/jz36.png)

思路1， 先各自遍历一遍，确定两个链表的各自长度M、N。然后长链表的指针先走|M-N|步，然后长短链表的指针再一起走，碰到相同节点就return

Runtime: 248 ms, faster than 15.54%
```python
    def getIntersectionNode(self, headA, headB):
        M = 0
        N = 0
        dif = 0
        curA = headA
        curB = headB
        while curA:
            M += 1
            curA = curA.next
        while curB:
            N += 1
            curB = curB.next
        curA = headA
        curB = headB
        if M>N:
            dif = M-N
            while dif>0:
                curA = curA.next
                dif -= 1
        elif N>M:
            dif = N-M
            while dif>0:
                curB = curB.next
                dif -= 1
        while curA and curB:
            if curA==curB:
                return curA
            curA = curA.next
            curB = curB.next
        return None  
```

思路2， 
<u>1 2 3 6 7 4 5</u> 6 7
<u>4 5 6 7 1 2 3</u> 6 7
两个指针一起走，当p1走到终点时，就让他指向第二个链表的头；当p2走到终点时，就让p2指向第一个链表的头。

**边界控制：如果没有公共节点的话，最后两个指针都会指向None，相同的None节点会退出～**

Runtime: 212 ms, faster than 51.17% 
```python
    def getIntersectionNode(self, headA, headB):
        curA = headA
        curB = headB
        while curA != curB:
            curA = curA.next if curA else headB
            curB = curB.next if curB else headA
        return curA
```

# LeetCode 102. Binary Tree Level Order Traversal
原题：https://leetcode.com/problems/binary-tree-level-order-traversal/
Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).

For example:
Given binary tree [3,9,20,null,null,15,7],
```
    3
   / \
  9  20
    /  \
   15   7
```
return its level order traversal as:
```
[
  [3],
  [9,20],
  [15,7]
]
```
分析：二叉树的层次遍历，不过要把每一层分开输出
level traverse ——> 队列， 同样是用队列，不过你要知道每次要出队几个元素。比如初始队列里只有1个元素，那么你出队1个，并且要记录此次出队过程中，入队了多少个元素，这个过程中入队的元素个数，就是下一次出队的个数。可以设out_cnt=1表示此次出队个数，in_cnt记录此次出队过程中又入队了多少个元素，作为下次出队的个数。

Runtime: 24 ms, faster than 99.98%
```python
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        queue = []
        queue.append(root)
        out_cnt = 1
        in_cnt = 0
        res = []
        while queue:
            tmp = []
            while out_cnt>0:
                top = queue[0]
                tmp.append(top.val)
                queue.pop(0)
                if top.left:
                    queue.append(top.left)
                    in_cnt += 1
                if top.right:
                    queue.append(top.right)
                    in_cnt += 1
                out_cnt -= 1
            res.append(tmp)
            out_cnt = in_cnt
            in_cnt = 0
        return res 
```
# ★ LeetCode 337. House Robber III
The thief has found himself a new place for his thievery again. There is only one entrance to this area, called the "root." Besides the root, each house has one and only one parent house. After a tour, the smart thief realized that "all houses in this place forms a binary tree". It will automatically contact the police if two directly-linked houses were broken into on the same night.

Determine the maximum amount of money the thief can rob tonight without alerting the police.

Example 1:
```
Input: [3,2,3,null,3,null,1]

     3
    / \
   2   3
    \   \ 
     3   1

Output: 7 
Explanation: Maximum amount of money the thief can rob = 3 + 3 + 1 = 7.
```
Example 2:
```
Input: [3,4,5,1,3,null,1]

     3
    / \
   4   5
  / \   \ 
 1   3   1

Output: 9
Explanation: Maximum amount of money the thief can rob = 4 + 5 = 9.
```

分析：递归， 对于每一个节点root，都有偷它或者不偷它两种情况，如果到了root，可以偷它，或者不偷它，你要保存不偷root的结果，用作以后的参考，L表示偷左孩子，L_no表示不偷左孩子。
偷到root的结果是：`max(root.val+L_no+R_no, L+R)`
保存不偷root的结果：`L+R`
定义递归函数helper(root)，每次返回以上两个值。
原函数最终的结果即res[0]，偷到root的结果。

Runtime: 40 ms, faster than 79.51%
```python
    def rob(self, root):
        def recursion(root):
            if not root:
                return 0, 0
            L, no_L = recursion(root.left)
            R, no_R = recursion(root.right)
            return max(root.val+no_L+no_R, L+R), L+R
        res = recursion(root)
        return res[0]
```
# LeetCode 48. Rotate Image
原题：https://leetcode.com/problems/rotate-image/
You are given an n x n 2D matrix representing an image.

Rotate the image by 90 degrees (clockwise).

Note:

You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.

Example 1:
```
Given input matrix = 
[
  [1,2,3],
  [4,5,6],
  [7,8,9]
],

rotate the input matrix in-place such that it becomes:
[
  [7,4,1],
  [8,5,2],
  [9,6,3]
]
```
Example 2:
```
Given input matrix =
[
  [ 5, 1, 9,11],
  [ 2, 4, 8,10],
  [13, 3, 6, 7],
  [15,14,12,16]
], 

rotate the input matrix in-place such that it becomes:
[
  [15,13, 2, 5],
  [14, 3, 4, 1],
  [12, 6, 8, 9],
  [16, 7,10,11]
]
```
分析：旋转二维数组有一个技巧，
顺时针旋转90度：先上下翻转，再沿对角线翻转
```python
# clockwise rotate
# first reverse up to down, then swap the symmetry 
# 1 2 3     7 8 9     7 4 1
# 4 5 6  => 4 5 6  => 8 5 2
# 7 8 9     1 2 3     9 6 3
```
逆时针旋转90度：先左右翻转，再沿对角线翻转
```python
# anticlockwise rotate
# first reverse left to right, then swap the symmetry
# 1 2 3     3 2 1     3 6 9
# 4 5 6  => 6 5 4  => 2 5 8
# 7 8 9     9 8 7     1 4 7
```
Runtime: 24 ms, faster than 69.54%
```python
    def rotate(self, matrix):
        # 先上下翻转
        rows = len(matrix)
        cols = len(matrix[0])
        for i in range(rows//2):
            for j in range(cols):
                matrix[i][j], matrix[rows-1-i][j] = matrix[rows-1-i][j], matrix[i][j]
        # 再沿对角线翻转
        for i in range(rows):
            for j in range(i+1, cols):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
```
# LeetCode 53. Maximum Subarray
原题：https://leetcode.com/problems/maximum-subarray/   
Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

Example:
```
Input: [-2,1,-3,4,-1,2,1,-5,4],
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
```
Follow up:
If you have figured out the O(n) solution, try coding another solution using the divide and conquer approach, which is more subtle.

分析： 最大子序列和，利用动态规划，dp[i]表示数组nums[:i+1]的最大子序列和，dp[0] = nums[0]

Runtime: 72 ms, faster than 26.54%
```python
    def maxSubArray(self, nums):
        N = len(nums)
        dp = [0]*N
        dp[0] = nums[0]
        maxsum = dp[0]
        for i in range(1, N):
            dp[i] = max(dp[i-1]+nums[i], nums[i])
            maxsum = max(maxsum, dp[i])
        return maxsum
```
# 1. Two Sum
Given an array of integers, return indices of the two numbers such that they add up to a specific target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

Example:
```
Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].
```

分析：暴力搜索O(n^2)，如果要缩减时间就要以空间换时间，利用hash有两种方法
第一种，遍历一遍把数字和下标存起来，第二遍判断dic里边是否有target-numbers[i]这个元素。这种方法需要注意的是如[0, 1, -1], 0 这种情况，所以还要加个判断处理
```python
    def twoSum(self, numbers, target):
        if not numbers:
            return []
        N = len(numbers)
        dic = {}
        for i in range(N):
            dic[numbers[i]]=i
        for i in range(N):
            if target-numbers[i] in dic and dic[target-numbers[i]] != i:
                return [i, dic[target-numbers[i]]]
        return []
```

第二种方法比较巧妙，如下
Runtime: 36 ms, faster than 65.40% 
```python
    def twoSum(self, numbers, target):
        if not numbers:
            return []
        N = len(numbers)
        dic = {}
        for i in range(N):
            if numbers[i] in dic:
                return [dic[numbers[i]], i]
            else:
                dic[target-numbers[i]] = i
        return []
```

# 39. Combination Sum
原题：https://leetcode.com/problems/combination-sum/
Given a set of candidate numbers (candidates) (without duplicates) and a target number (target), find all unique combinations in candidates where the candidate numbers sums to target.

The same repeated number may be chosen from candidates unlimited number of times.

Note:

All numbers (including target) will be positive integers.
The solution set must not contain duplicate combinations.
Example 1:
```
Input: candidates = [2,3,6,7], target = 7,
A solution set is:
[
  [7],
  [2,2,3]
]
```
Example 2:
```
Input: candidates = [2,3,5], target = 8,
A solution set is:
[
  [2,2,2,2],
  [2,3,3],
  [3,5]
]
```
分析：用dfs来做
nums:不变
res
target：每次减，等于0退出。由于都是整数，小于0则return
path和i:设j从下标i开始到最后，避免重复

Runtime: 84 ms, faster than 55.82%
```python
    def combinationSum(self, candidates, target):
        res = []
        def dfs(candidates, target, res, path, i):
            if target == 0:
                res.append(path[:])
            if target<0:
                return
            for j in range(i, len(candidates)):
                dfs(candidates, target-candidates[j], res, path+[candidates[j]], j)
        dfs(candidates, target, res, [], 0)
        return res
```


# 300. Longest Increasing Subsequence
原题：https://leetcode.com/problems/longest-increasing-subsequence/
Given an unsorted array of integers, find the length of longest increasing subsequence.

Example:
```
Input: [10,9,2,5,3,7,101,18]
Output: 4 
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4. 
```
Note:

There may be more than one LIS combination, it is only necessary for you to return the length.
Your algorithm should run in O(n2) complexity.
Follow up: Could you improve it to O(n log n) time complexity?

分析：
思路1，设dp数组，每一次遍历到x，从后往前找 $dp[i]=max\{dp[j]+1(0<=j<i，arr[j]<arr[i])\}。$ dp[j]中数值最大的那个+1。

时间复杂度：O(n^2)
Runtime: 1084 ms, faster than 11.25%
```python
    def lengthOfLIS(self, nums):
        if not nums:
            return 0
        N = len(nums)
        dp = [1]*(N)
        for i in range(1, N):
            for j in range(i, -1, -1):
                if nums[i]>nums[j]:
                    dp[i] = max(dp[i], dp[j]+1)
        return max(dp)
```
思路2分析：利用二分查找优化时间
**用一个数组ends保存最长递增序列为k的最小元素**，如
```
nums        2 1 5 3 6 4 8 9 7
dp          1 1 2 2 3 3 4 5 4

endsindex 0 1 2 3 4 5 6 7 8 9         
ends        1 3 4 7 9
```
>ends下标为k，表示原数组中最长递增序列长度为k的这些序列中，最小的元素是ends[k]。下标从1开始是为了方便。

初始，dp[0] = 1，ends[1] = nums[0]，right = 1（有效下标边界），当遍历nums中的数x时，在ends数组中利用二分查找第一个大于等于x的数（所在ends的下标l），**（如果ends中存在这样一个数，返回所在ends下标；如果不存在，则在right右边。不过最后的下标都是l）**

**找到l下标之后，更新ends[l]=x，更新dp[i]=l（l是ends下标即最长长度），更新right为max(l, right)**

时间复杂度O(nlogn)
Runtime: 32 ms, faster than 80.94%
```python
    def lengthOfLIS(self, nums):
        if not nums:
            return 0
        N = len(nums)
        dp = [0] * N
        ends = [0] * (N+1)
        right = 1
        dp[0] = 1
        ends[1] = nums[0]
        for i in range(1, N):# nums中的数字nums[i]
            #在ends里进行二分查找
            l, r = 1, right
            while l<=r:
                mid = (l+r)//2
                if ends[mid]<nums[i]:
                    l = mid+1
                else:
                    r = mid-1
            # l是那个位置
            ends[l] = nums[i] #更新ends[l]
            dp[i] = l #更新dp
            right = max(l, right) #更新right
        return max(dp)
```

# 279. Perfect Squares
https://leetcode.com/problems/perfect-squares/
Given a positive integer n, find the least number of perfect square numbers (for example, 1, 4, 9, 16, ...) which sum to n.

Example 1:
```
Input: n = 12
Output: 3 
Explanation: 12 = 4 + 4 + 4.
```
Example 2:
```
Input: n = 13
Output: 2
Explanation: 13 = 4 + 9.
```
分析：利用动态规划
注意一点：代码中注释部分，不要从1开始到sqrt(N)进行判断，直接根据下标不越界判断，否则超时。

Runtime: 4896 ms, faster than 14.45%
```python
    def numSquares(self, n):
        dp = [n]*(n+1)
        dp[0] = 0
        dp[1] = 1
        for i in range(2, n+1):
            last = int(math.sqrt(i))
            j = 0
            while i-j**2>=0: # 这是一种节约时间的方法
                dp[i] = min(dp[i], 1+dp[i-j**2])
                j += 1
            # for j in range(1, i+1): # 会超时
            #     if i-j**2>0:
            #         dp[i] = min(dp[i], 1+dp[i-j**2])
        return dp[-1]
```
[同样dp的另一种时间复杂度低的写法](https://leetcode.com/problems/perfect-squares/discuss/71512/Static-DP-C%2B%2B-12-ms-Python-172-ms-Ruby-384-ms)
```python
class Solution(object):
    _dp = [0]
    def numSquares(self, n):
        dp = self._dp
        while len(dp) <= n:
            dp += min(dp[-i*i] for i in range(1, int(len(dp)**0.5+1))) + 1,
        return dp[n]
```
# 75. Sort Colors
https://leetcode.com/problems/sort-colors/

Given an array with n objects colored red, white or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white and blue.

Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.

Note: You are not suppose to use the library's sort function for this problem.

Example:
```
Input: [2,0,2,1,1,0]
Output: [0,0,1,1,2,2]
```
Follow up:
```
- A rather straight forward solution is a two-pass algorithm using counting sort.
First, iterate the array counting number of 0's, 1's, and 2's, then overwrite array with total number of 0's, then 1's and followed by 2's.
- Could you come up with a one-pass algorithm using only constant space?
```

分析：对只含0 1 2的数组排序，不能用库函数，也不能用O(2n)遍历两次，要求遍历1次，并且只使用常数级的内存。
直观的想法是，只交换0和2，那么剩下的1自然就排好序了。但实际情况是，如果设首尾两个指针，遇到0和2就交换，很可能是这样`2,1,0,1`前边遇到2后边遇到0，所以交换，变成了`0,1,2,1`，不对。
所以，改进的方法是，用k去遍历数组，i表示i以前的数字都是0，j表示j以后的数字都是2，A[k]碰到0就和A[i]的数字交换，保证A[i]始终为0开头的数字，A[k]遇到2就和A[j]交换，但是由于可能把1或0换到了A[k]，k需要回退。而前者不需要回退的原因是，一直都是从k开始判断的。
[参考链接](https://leetcode.com/problems/sort-colors/discuss/26500/Four-different-solutions)
Runtime: 20 ms, faster than 95.72%
```python
    def sortColors(self, nums):
        i, j = 0, len(nums)-1
        k = -1
        while k<=j:
            if k<0:
                k += 1
                continue
            if nums[k] == 0:
                nums[i], nums[k] = nums[k], nums[i]
                i += 1
            elif nums[k] == 2:
                nums[j], nums[k] = nums[k], nums[j]
                j -= 1
                k -= 1
            k += 1
```
# 200. Number of Islands
https://leetcode.com/problems/number-of-islands/

Given a 2d grid map of '1's (land) and '0's (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

Example 1:
```
Input:
11110
11010
11000
00000

Output: 1
```
Example 2:
```
Input:
11000
11000
00100
00011

Output: 3
```
分析：dfs

Runtime: 128 ms, faster than 54.61%

```python
    def numIslands(self, grid):
        if not grid:
            return 0
        result = 0
        self.M = len(grid)
        self.N = len(grid[0])
        # dfs
        def dfs(grid, i, j):
            if i<0 or j<0 or i>=self.M or j>=self.N or grid[i][j]!='1':
                return
            grid[i][j] = '#'
            dfs(grid, i-1, j)
            dfs(grid, i, j-1)
            dfs(grid, i+1, j)
            dfs(grid, i, j+1)

        for i in range(self.M):
            for j in range(self.N):
                if grid[i][j] == '1':
                    dfs(grid, i, j)
                    result += 1
        return result
```
# 207. Course Schedule
https://leetcode.com/problems/course-schedule/

There are a total of n courses you have to take, labeled from 0 to n-1.

Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]

Given the total number of courses and a list of prerequisite pairs, is it possible for you to finish all courses?

Example 1:
```
Input: 2, [[1,0]] 
Output: true
Explanation: There are a total of 2 courses to take. 
             To take course 1 you should have finished course 0. So it is possible.
```
Example 2:
```
Input: 2, [[1,0],[0,1]]
Output: false
Explanation: There are a total of 2 courses to take. 
             To take course 1 you should have finished course 0, and to take course 0 you should
             also have finished course 1. So it is impossible.
```             
Note:

- The input prerequisites is a graph represented by a list of edges, not adjacency matrices. Read more about how a graph is represented.
- You may assume that there are no duplicate edges in the input prerequisites.

分析：dfs
[参考](https://leetcode.com/problems/course-schedule/discuss/58586/Python-20-lines-DFS-solution-sharing-with-explanation)
Runtime: 72 ms, faster than 70.81%
```python
    def canFinish(self, numCourses, prerequisites):
        graph = [[] for i in range(numCourses)]
        visit = [0 for i in range(numCourses)]
        for x, y in prerequisites:
            graph[x].append(y)

        def dfs(i): # dfs
            if visit[i] == -1:
                return False
            if visit[i] == 1:
                return True
            visit[i] = -1
            for j in graph[i]:
                if not dfs(j):
                    return False
            visit[i] = 1
            return True
        
        for j in range(numCourses):
            if not dfs(j):
                return False
        return True
```