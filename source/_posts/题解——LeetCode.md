---
title: 题解——LeetCode
date: 2019-04-16 16:08:35
tags: 题解
top: 10
categories: Algorithm
---
# 720. Longest Word in Dictionary
Given a list of strings words representing an English Dictionary, find the longest word in words that can be built one character at a time by other words in words. If there is more than one possible answer, return the longest word with the smallest lexicographical order.

If there is no answer, return the empty string.
Example 1:
```
Input: 
words = ["w","wo","wor","worl", "world"]
Output: "world"
Explanation: 
The word "world" can be built one character at a time by "w", "wo", "wor", and "worl".
```
Example 2:
```
Input: 
words = ["a", "banana", "app", "appl", "ap", "apply", "apple"]
Output: "apple"
Explanation: 
Both "apply" and "apple" can be built from other words in the dictionary. However, "apple" is lexicographically smaller than "apply".
```
Note:
```
All the strings in the input will only contain lowercase letters.
The length of words will be in the range [1, 1000].
The length of words[i] will be in the range [1, 30].
```

分析：先对words进行排序，设一个新的空容器 word_set，**初始里边是空字符串**，然后遍历words，如果words[:-1]已经有了，就把它加进去，因为要求逐步的，所以如果是单个字符，它的前一个应该是''，并一直更新最长的字符串。

Runtime: 60 ms, faster than 66.19%

```python
    def longestWord(self, words):
        words.sort()
        word_set, longestword = set(['']), ''
        for word in words:
            if word[:-1] in word_set:
                word_set.add(word)
                if len(word)>len(longestword):
                    longestword = word
        return longestword  
```
# 347. Top K Frequent Elements
原题：https://leetcode.com/problems/top-k-frequent-elements/
Given a non-empty array of integers, return the k most frequent elements.

Example 1:
```
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
```
Example 2:
```
Input: nums = [1], k = 1
Output: [1]
```
```
Note:
You may assume k is always valid, 1 ≤ k ≤ number of unique elements.
Your algorithm's time complexity must be better than O(n log n), where n is the array's size.
```
分析：如果top K问题要求元素有序，则用堆排序
如果不要求元素有序，利用快排

时间复杂度分析：



# 有序的数组中找到某值首次出现的下标
给定一个升序的数组，这个数组中可能含有相同的元素，并且给定一个目标值。要求找出目标值在数组中首次出现的下标。 

分析：题目给出有序数组，应该想到利用二分查找来做。找到左邻居，使其值加一。利用二分查找，算法复杂度为O(logn)
和最长递增子序列时间复杂度O(nlogn)查找ends数组中第一个大于nums[i]的数的下标一样。

二分查找总结：
https://blog.csdn.net/yefengzhichen/article/details/52372407
https://github.com/selfboot/LeetCode/blob/master/BinarySearch/README.md


# 993. Cousins in Binary Tree
https://leetcode.com/problems/cousins-in-binary-tree/
In a binary tree, the root node is at depth 0, and children of each depth `k` node are at depth `k+1`.

Two nodes of a binary tree are cousins if they have the same depth, but have different parents.

We are given the root of a binary tree with unique values, and the values `x` and `y` of two different nodes in the tree.

Return `true` if and only if the nodes corresponding to the values `x` and `y` are cousins.

Example 1:
![](/images/lc993_1.png)
```
Input: root = [1,2,3,4], x = 4, y = 3
Output: false
```

Example 2:
![](/images/lc993_3.png)
```
Input: root = [1,2,3,null,4,null,5], x = 5, y = 4
Output: true
```
Example 3:
![](/images/lc993_2.png)
```
Input: root = [1,2,3,null,4], x = 2, y = 3
Output: false
```
分析：利用DFS遍历树，可以获得某个节点的深度和父节点。
开始是用了两个dfs分别找x和y，时间复杂度高，后来合并优化了一下。

分别寻找，beats 11.42%
```python
    def isCousins(self, root, x, y):
        """
        :type root: TreeNode
        :type x: int
        :type y: int
        :rtype: bool
        """
        def dfs(root, x, depth, parent):
            if not root:
                return
            if root.val==x:
                return depth, parent
            return dfs(root.left, x, 1+depth, root) or dfs(root.right, x, 1+depth, root)
        dx, px = dfs(root, x, 0, None)
        dy, py = dfs(root, y, 0, None)
        if dx!=dy:
            return False
        elif dx==0 and dy==0:
            return False
        elif px==py:
            return False
        elif dx==dy and px!=py:
            return True
```

优化之后，beats 66.56%
```python
    def isCousins(self, root, x, y):
        self.dx, self.px = 0, None
        self.dy, self.py = 0, None
        def dfs(root, x, y, depth, parent):
            if not root:
                return
            if root.val==x:
                self.dx, self.px = depth, parent
            if root.val == y:
                self.dy, self.py = depth, parent
            dfs(root.left, x, y, 1+depth, root)
            dfs(root.right, x, y, 1+depth, root)
        dfs(root, x, y, 0, None)

        print(self.dx, self.dy)
        if self.dx != self.dy:
            return False
        elif self.dx == 0 and self.dy == 0:
            return False
        elif self.px == self.py:
            return False
        elif self.dx == self.dy and self.px != self.py:
            return True
```

# 653. Two Sum IV - Input is a BST
https://leetcode.com/problems/two-sum-iv-input-is-a-bst/
Given a Binary Search Tree and a target number, return true if there exist two elements in the BST such that their sum is equal to the given target.

Example 1:
```
Input: 
    5
   / \
  3   6
 / \   \
2   4   7

Target = 9

Output: True
```

Example 2:
```
Input: 
    5
   / \
  3   6
 / \   \
2   4   7

Target = 28

Output: False
```
分析：深度优先搜索遍历整个二叉树，不过每次把值存到lst里，并每次判断k-val是否存在于lst中，如果存在，说明找到了。
但这种方法没有利用BST的性质...
也可以中序遍历得到序列，再首尾判断。

Runtime: 108 ms, faster than 12.87%
```python
    def findTarget(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: bool
        """
        lst = []
        def dfs(root, k, lst):
            if not root:
                return False
            if k-root.val in lst:
                return True
            lst.append(root.val)
            return dfs(root.left, k, lst) or dfs(root.right, k, lst)
        return dfs(root, k, lst)
```
# 671. Second Minimum Node In a Binary Tree
https://leetcode.com/problems/second-minimum-node-in-a-binary-tree/

Given a non-empty special binary tree consisting of nodes with the non-negative value, where each node in this tree has exactly two or zero sub-node. If the node has two sub-nodes, then this node's value is the smaller value among its two sub-nodes.

Given such a binary tree, you need to output the second minimum value in the set made of all the nodes' value in the whole tree.

If no such second minimum value exists, output -1 instead.

Example 1:
```
Input: 
    2
   / \
  2   5
     / \
    5   7

Output: 5
Explanation: The smallest value is 2, the second smallest value is 5.
```

Example 2:
```
Input: 
    2
   / \
  2   2

Output: -1
Explanation: The smallest value is 2, but there isn't any second smallest value.    
```
分析：这是一种树，根节点是两个孩子节点中较小的那个，考虑到两个孩子节点值相等的情况。
当没有孩子时，返回-1
当根节点和左右孩子节点相等时，最小值是root.val，这时可能不存在第二小的节点，需要继续遍历。
总而言之，可以用先序递归的方法遍历二叉树，如果碰到第二小的元素就更新res的值。

- `res = float('inf)`表示正无穷
- `res = float('-inf')`表示负无穷

Runtime: 16 ms, faster than 98.30% 

```python
    def __init__(self):
        self.res_lc671 = float('inf')
        
    def findSecondMinimumValue(self, root):
        if not root.left and not root.right:
            return -1
        def traverse(node):
            if not node:
                return
            if root.val<node.val<self.res_lc671:
                self.res_lc671 = node.val
            traverse(node.left)
            traverse(node.right)
        traverse(root)
        return -1 if self.res_lc671 == float('inf') else self.res_lc671
```


