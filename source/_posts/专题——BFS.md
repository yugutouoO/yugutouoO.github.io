---
title: 专题——BFS
date: 2019-04-16 15:44:04
tags: 
- BFS
- Tree
top: 10
categories: Algorithm
---
# LeetCode 994. Rotting Oranges
In a given grid, each cell can have one of three values:

the value 0 representing an empty cell;
the value 1 representing a fresh orange;
the value 2 representing a rotten orange.
Every minute, any fresh orange that is adjacent (4-directionally) to a rotten orange becomes rotten.

Return the minimum number of minutes that must elapse until no cell has a fresh orange.  If this is impossible, return -1 instead.

Example 1:
![](/images/lc994oranges.png)
```
Input: [[2,1,1],[1,1,0],[0,1,1]]
Output: 4
```
Example 2:
```
Input: [[2,1,1],[0,1,1],[1,0,1]]
Output: -1
Explanation:  The orange in the bottom left corner (row 2, column 0) is never rotten, because rotting only happens 4-directionally.
```
Example 3:
```
Input: [[0,2]]
Output: 0
Explanation:  Since there are already no fresh oranges at minute 0, the answer is just 0.
```

分析：利用队列进行广度优先搜索，也就是类似树的层次遍历
```python
class Solution(object):
    def orangesRotting(self, grid):
        if not grid:
            return -1
        queue = []
        fresh_cnt = 0
        in_size = 0
        count = 0
        M = len(grid)
        N = len(grid[0])
        for i in range(M):
            for j in range(N):
                if grid[i][j] == 1:
                    fresh_cnt += 1
                elif grid[i][j] == 2:
                    queue.append([i, j])
        out_size = len(queue)
        if out_size == 0 and fresh_cnt!=0:
            return -1
        if out_size==0 and fresh_cnt==0:
            return 0
        if out_size!=0 and fresh_cnt==0:
            return 0
        while queue:
            # print(queue)
            count += 1
            while out_size: # 出队已经腐烂的，入队的是本次队列里可以感染的
                top = queue[0]
                queue.pop(0)
                out_size -= 1
                if self.checkBoundary(top[0], top[1]-1, M, N) and grid[top[0]][top[1]-1]==1:
                    fresh_cnt -= 1
                    queue.append([top[0], top[1]-1])
                    grid[top[0]][top[1]-1] = 2
                    in_size += 1
                if self.checkBoundary(top[0], top[1]+1, M, N) and grid[top[0]][top[1]+1] == 1:
                    fresh_cnt -= 1
                    queue.append([top[0], top[1]+1])
                    grid[top[0]][top[1]+1] = 2
                    in_size += 1
                if self.checkBoundary(top[0]-1, top[1], M, N) and grid[top[0]-1][top[1]] == 1:
                    fresh_cnt -= 1
                    queue.append([top[0]-1, top[1]])
                    grid[top[0]-1][top[1]] = 2
                    in_size += 1
                if self.checkBoundary(top[0]+1, top[1], M, N) and grid[top[0]+1][top[1]] == 1:
                    fresh_cnt -= 1
                    queue.append([top[0]+1, top[1]])
                    grid[top[0]+1][top[1]] = 2
                    in_size += 1
            out_size = in_size
            in_size = 0
        # print(fresh_cnt)
        # print(fresh_cnt2)
        # print(count)
        if fresh_cnt>0:
            return -1
        else:
            return count-1

    def checkBoundary(self, i, j, M, N):
        if i>=0 and i<M and j>=0 and j<N:
            return True
        else:
            return False
```