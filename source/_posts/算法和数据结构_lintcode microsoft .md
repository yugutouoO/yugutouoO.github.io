---
title: LintCode Microsoft 
date: 2019-04-13 20:51:18
tags: 
- LintCode
categories: 算法和数据结构
---
# 53. Reverse Words in a String
原题：https://www.lintcode.com/problem/reverse-words-in-a-string/description
Given an input string, reverse the string word by word.

Clarification
What constitutes a word?
A sequence of non-space characters constitutes a word and some words have punctuation at the end.
Could the input string contain leading or trailing spaces?
Yes. However, your reversed string should not contain leading or trailing spaces.
How about multiple spaces between two words?
Reduce them to a single space in the reversed string.

注意，一个string中可能有多个连续空格

Total runtime 1208 ms
Your submission beats 47.00% Submissions!
```python
    def reverseWords(self, s):
        if not s:
            return ''
        lst = []
        for ch in s.split():
            if ch!=' ':
                lst.append(ch)
        print(lst)
        return ' '.join(lst[::-1])
```

# 56. Two Sum
原题：https://www.lintcode.com/problem/two-sum/description
Given an array of integers, find two numbers such that they add up to a specific target number.

The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2. Please note that your returned answers (both index1 and index2) are zero-based.

*You may assume that each input would have exactly one solution*
Example
Example1:
```
numbers=[2, 7, 11, 15], target=9
return [0, 1]
Example2:
numbers=[15, 2, 7, 11], target=9
return [1, 2]
```
Challenge
```
Either of the following solutions are acceptable:
O(n) Space, O(nlogn) Time
O(n) Space, O(n) Time
```
Total runtime 1007 ms
Your submission beats 42.80% Submissions!
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
# 57. 3Sum
原题：https://www.lintcode.com/problem/3sum/description
Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.

Elements in a triplet (a,b,c) must be in non-descending order. (ie, a ≤ b ≤ c)

The solution set must not contain duplicate triplets.
Example
Example 1:
```
Input:[2,7,11,15]
Output:[]
```
Example 2:
```
Input:[-1,0,1,2,-1,-4]
Output:	[[-1, 0, 1],[-1, -1, 2]]
```

分析：把target变为-nums[i]，然后就变成求twoSum的问题了，和twoSum不太一样的地方，这里的结果是数字，twoSum结果要下标

Total runtime 907 ms
Your submission beats 25.20% Submissions!
```python
    def threeSum(self, numbers):
        if not numbers:
            return []
        N = len(numbers)
        numbers.sort()
        res = []
        for i in range(N):
            target = -numbers[i]
            j, k = i+1, N-1
            while j<k:
                if numbers[j]+numbers[k] == target:
                    tmp = [numbers[i], numbers[j], numbers[k]]
                    if tmp not in res:
                        res.append(tmp)
                    j += 1
                elif numbers[j]+numbers[k] < target:
                    j += 1
                elif numbers[j]+numbers[k] > target:
                    k -= 1
        return res
```
# 62. Search in Rotated Sorted Array
原题：https://www.lintcode.com/problem/search-in-rotated-sorted-array/
Suppose a sorted array is rotated at some pivot unknown to you beforehand.

(i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).

You are given a target value to search. If found in the array return its index, otherwise return -1.

*You may assume no duplicate exists in the array.*

Example
Example 1:
```
Input: [4, 5, 1, 2, 3] and target=1, 
Output: 2.
```
Example 2:
```Input: [4, 5, 1, 2, 3] and target=0, 
Output: -1.
```
Challenge
O(logN) time

分析：二分查找
注意代码中的边界
参考[剑指 6.旋转数组的最小数字
](https://www.nowcoder.com/practice/9f3231a991af4f55b95579b44b7a01ba?tpId=13&tqId=11159&rp=1&ru=%2Fta%2Fcoding-interviews&qru=%2Fta%2Fcoding-interviews%2Fquestion-ranking&tPage=1)
Total runtime 101 ms
Your submission beats 100.00% Submissions!
```python
    def search(self, A, target):
        if not A:
            return -1
        N = len(A)
        l = 0
        r = N-1
        mid = (l+r)//2  
        while l+1<r:
            if A[mid] <= A[r]:  # 右半部分有序
                if target>=A[mid] and target<=A[r]:#target在有序的右半部分
                    l = mid
                    mid = (l+r)//2
                else:#target在有pivot的左半部分
                    r = mid
                    mid = (l+r)//2
            if A[l]<=A[mid]:
                if target>=A[l] and target<=A[mid]:#target在有序的左半部分
                    r = mid
                    mid = (l+r)//2
                else:#target在有pivot的左半部分
                    l = mid
                    mid = (l+r)//2
        if A[l] == target:
            return l
        if A[r] == target:
            return r
        return -1
```
# 67. Binary Tree Inorder Traversal
原题：https://www.lintcode.com/problem/binary-tree-inorder-traversal/
Given a binary tree, return the inorder traversal of its nodes' values.

Given binary tree `{1,#,2,3}`,
```
   1
    \
     2
    /
   3
```
return [1,3,2].

Challenge
Can you do it without recursion? 
用栈

Total runtime 61 ms
Your submission beats 99.00% Submissions!
```python
class Solution:
    """
    @param root: A Tree
    @return: Inorder in ArrayList which contains node values.
    """
    def __init__(self):
        self.inorder_list = []
    def inorderTraversal(self, root):
        def inTra(root):
            if not root:
                return
            self.inorderTraversal(root.left)
            self.inorder_list.append(root.val)
            self.inorderTraversal(root.right)
        inTra(root)
        return self.inorder_list
```
# 69. Binary Tree Level Order Traversal
原题：https://www.lintcode.com/problem/binary-tree-level-order-traversal/description
Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).

Example
Example 1:
```
Input：{1,2,3}
Output：[[1],[2,3]]
Explanation：
level traversal
```
Example 2:
```
Input：{1,#,2,3}
Output：[[1],[2],[3]]
Explanation：
level traversal
```
Challenge
Challenge 1: Using only 1 queue to implement it.
Challenge 2: Use BFS algorithm to do it.

Total runtime 101 ms
Your submission beats 96.60% Submissions!
```python
    def levelOrder(self, root):
        if not root:
            return []
        queue = []
        queue.append(root)
        out_cnt = 1
        in_cnt = 0
        res = []
        while queue:
            tmp = []
            while out_cnt > 0:
                out = queue[0]
                queue.pop(0)
                out_cnt -= 1
                tmp.append(out.val)
                if out.left:
                    queue.append(out.left)
                    in_cnt += 1
                if out.right:
                    queue.append(out.right)
                    in_cnt += 1
            out_cnt = in_cnt
            in_cnt = 0
            res.append(tmp)
        return res
```
# 71. Binary Tree Zigzag Level Order Traversal
原题：https://www.lintcode.com/problem/binary-tree-zigzag-level-order-traversal/description
Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between).
Example 1:
```
Input:
{1,2,3}
Output:
[[1],[3,2]]
Explanation:
    1
   / \
  2   3
```
Example 2:
```
Input:
{3,9,20,#,#,15,7}
Output:
[
  [3],
  [20,9],
  [15,7]
]
Explanation:
    3
   / \
  9  20
    /  \
   15   7
```
分析：只是比上题多了个flag

Total runtime 62 ms
Your submission beats 97.00% Submissions!

```python
    def zigzagLevelOrder(self, root):
        if not root:
            return []
        queue = []
        queue.append(root)
        out_cnt = 1
        in_cnt = 0
        res = []
        flag = True # True，从左往右，False，从右往左
        while queue:
            tmp = []
            while out_cnt > 0:
                out = queue[0]
                queue.pop(0)
                tmp.append(out.val)
                if out.left:
                    queue.append(out.left)
                    in_cnt += 1
                if out.right:
                    queue.append(out.right)
                    in_cnt += 1
                out_cnt -= 1
            out_cnt = in_cnt
            in_cnt = 0
            if flag:
                res.append(tmp)
            else:
                res.append(tmp[::-1])
            flag = not flag
        return res
```

# 72. Construct Binary Tree from Inorder and Postorder Traversal
原题：https://www.lintcode.com/problem/construct-binary-tree-from-inorder-and-postorder-traversal/description
Given inorder and postorder traversal of a tree, construct the binary tree.
Example
Example 1:
```
Input:
[1,2]
[2,1]
Output:
{1,#,2}
Explanation:
    1
     \
      2
```
Example 2:
```
Input:
[1,2,3]
[1,3,2]
Output:
{2,1,3}
Explanation:
  2

 /  \

1    3
```

分析：利用递归即可，后序的最后一个元素是根节点root，然后在中序中找到该元素的下标，root.left是左半部分中序和后序的根节点，root.right是右半部分中序和后序的根节点，return root

Total runtime 101 ms
Your submission beats 90.55% Submissions!
```python
class Solution:
    """
    @param inorder: A list of integers that inorder traversal of a tree
    @param postorder: A list of integers that postorder traversal of a tree
    @return: Root of a tree
    """
    def buildTree(self, inorder, postorder):
        def recursive(inorder, postorder):
            if not inorder or not postorder:
                return None
            if set(inorder)!=set(postorder):
                return None
            val = postorder[-1]
            root = TreeNode(val)
            idx = inorder.index(val)
            left_inorder = inorder[:idx]
            right_inorder = inorder[idx+1:]
            left_postorder = postorder[:idx]
            right_postorder = postorder[idx:-1]
            root.left = recursive(left_inorder, left_postorder)
            root.right = recursive(right_inorder, right_postorder)
            return root
        return recursive(inorder, postorder)
```
# 75. Find Peak Element
原题：https://www.lintcode.com/problem/find-peak-element/description
There is an integer array which has the following features:

The numbers in adjacent positions are different.
`A[0] < A[1] && A[A.length - 2] > A[A.length - 1].`
We define a position P is a peak if:

`A[P] > A[P-1] && A[P] > A[P+1]`
Find a peak element in this array. Return the index of the peak.

*It's guaranteed the array has at least one peak.
The array may contain multiple peeks, find any of them.
The array has at least 3 numbers in it.*

分析：二分法。
每次取中间元素，如果大于左右，则这就是peek。
否则取大的一边，两个都大，可以随便取一边。最终会找到peek。
**二分时，选择大的一边**

正确性证明：
题目中说了，肯定存在一个peak元素，而且A[0] < A[1] && A[n-2] > A[n-1]，所以如果A[mid-1]>A[mid]，左半部分肯定有peak，如果A[mid+1]>A[mid]，右半部分肯定有peak
二分时，选择大的一边, 则留下的部分仍然满足1 的条件，即最两边的元素都小于相邻的元素。所以仍然必然存在peek。

注意：
- 二分查找时的一个套路
如果折半时需要i=mid或者j=mid而不是i=mid+1或者j=mid-1，while循环需要到两个间隔为1时即`while i+1<j`，然后出了循环再判断i==j的情况

Total runtime 2621 ms
Your submission beats 13.00% Submissions!
```python
class Solution:
    """
    @param A: An integers array.
    @return: return any of peek positions.
    """
    def findPeak(self, A):
        N = len(A)
        i, j = 0, N-1
        while i+1<j:
            mid = (i+j)//2
            if A[mid]>A[mid-1] and A[mid]>A[mid+1]:
                return mid
            elif A[mid]<A[mid-1]:
                j = mid
            elif A[mid]<A[mid+1]:
                i = mid
        if i==j:
            if A[i]>A[j]:
                return i
            else:
                return j
        return -1
```
# 76. Longest Increasing Subsequence
原题：https://www.lintcode.com/problem/longest-increasing-subsequence/description
Given a sequence of integers, find the longest increasing subsequence (LIS).

You code should return the length of the LIS.

What's the definition of longest increasing subsequence?

The longest increasing subsequence problem is to find a subsequence of a given sequence in which the subsequence's elements are in sorted order, lowest to highest, and in which the subsequence is as long as possible. This subsequence is not necessarily contiguous, or unique.

https://en.wikipedia.org/wiki/Longest_increasing_subsequence

Example
```
Example 1:
	Input:  [5,4,1,2,3]
	Output:  3
	
	Explanation:
	LIS is [1,2,3]


Example 2:
	Input: [4,2,4,5,3,7]
	Output:  4
	
	Explanation: 
	LIS is [2,4,5,7]
```
Challenge
Time complexity O(n^2) or O(nlogn)

分析：
O(n^2) 解法：
dp[i] 表示以第i个数字为结尾的最长上升子序列的长度。
对于每个数字，枚举前面所有小于自己的数字 j，dp[i] = max(dp[i],dp[j] + 1). 如果没有比自己小的，dp[i] = 1;

Total runtime 1166 ms
Your submission beats 63.00% Submissions!

```python
class Solution:
    """
    @param nums: An integer array
    @return: The length of LIS (longest increasing subsequence)
    """
    def longestIncreasingSubsequence(self, nums):
        if not nums:
            return 0
        N = len(nums)
        dp = [0]*N
        dp[0] = 1
        for i in range(1, N):
            # print('dp', dp)
            # print('mn', maxnum)
            for k in range(i):
                if nums[i]>nums[k]:
                    dp[i] = max(dp[i], dp[k]+1)
            if dp[i]==0:
                dp[i] = 1
        return max(dp)
```





