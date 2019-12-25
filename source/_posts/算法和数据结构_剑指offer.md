---
title: 剑指offer
date: 2019-04-01 22:30:53
tags: 
- 剑指offer
categories: 算法和数据结构
---
原题链接：https://www.nowcoder.com/ta/coding-interviews?page=1

# 剑指 6. 旋转数组的最小数字
把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。 输入一个非减排序的数组的一个旋转，输出旋转数组的最小元素。 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。 NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。

分析：
`[mid]<=[right]`：说明右边非递减数列，最小元素在左边
`[mid]>=[left]`：说明左边是非递减数列，最小元素在右边
- **这里要有等号**
- 另外，因为是非递减数列，所以确定子空间之后，不能-1，应该**用`mid`作为边界**
- 最后，因为**如果left和right相差1**，则mid = left，这时不能直接返回[mid]，需要判断[left]和[right]的值哪个更小，这里必须要考虑相差1的情况，否则无限循环

```python
    def minNumberInRotateArray(self, arr):
		if not arr:
			return
		N = len(arr)
		i = 0
		j = N-1
		mid = 0
		while (i+1)<j:
			mid = (i+j)//2
			if arr[mid]<=arr[j]:
				j = mid
			if arr[mid]>=arr[i]:
				i = mid
		if i+1==j:
			return arr[i] if arr[i]<arr[j] else arr[j]
		return arr[mid]
```
# 剑指 15. 反转链表
输入一个链表，反转链表后，输出新链表的表头
分析：结点反转的思路，
![](/images/reverselist.png)
先定义None节点作为head的前一个节点pre，然后，链表的反转就是直接把head.next这个指向修改，，并同时修改变量，每一次，先保存nxt = head.next，然后修改指向head.next = pre，更新pre和head，pre = head，head = nxt，，注意不要直接在链表head上操作，定义一个新的cur指针

然后，注意判断什么时候结束循环：：temp = head.next，当temp==None时，说明到了最后一个结点，这时把新的链表头标记为head，继续执行反转操作，head = temp变成了temp，会结束循环。	也可以直接返回pre，就是新的头
- **另外，初始的None节点，pre = None，而不要写成pre = ListNode(-1)之类，它就应该是个None节点，否则会死循环！！！**
- 另外，不要直接修改原链表，定义一个cur指针来遍历

# 剑指 27. 
输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。
分析：dfs经典
```python
    def sPermutation(self, ss):
        if not ss:
            return []
        res = []
        def dfs(ss, res, path):
            if not ss:
                res.append(path)
            for i in range(len(ss)):
                # 这里没在函数体里path + ，所以不用path -
                dfs(ss[:i]+ss[i+1:], res, path+ss[i])
        dfs(ss, res, '')
        return sorted(list(set(res))) # 去重，排序
```
如果把代码中dfs()换成下边这三行也是一样的
```python
# path = path + ss[i]
# dfs(ss[:i]+ss[i+1:], res, path)
# path = path[:len(path)-1]
```
# 剑指 29. 最小的K个数
输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4。

分析：用排序或者堆来做

排序，26ms
```python
    def GetLeastNumbers_Solution(self, tinput, k):
        # write code here
        if not tinput or k>len(tinput):
            return []
        return sorted(tinput)[:k]
```
堆，26ms
```python
import heapq
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        # write code here
        if not tinput or k>len(tinput):
            return []
        return heapq.nsmallest(k, tinput)
```

# 剑指 30. 连续数组的最大和
HZ偶尔会拿些专业问题来忽悠那些非计算机专业的同学。今天测试组开完会后,他又发话了:在古老的一维模式识别中,常常需要计算连续子向量的最大和,当向量全为正数的时候,问题很好解决。但是,如果向量中包含负数,是否应该包含某个负数,并期望旁边的正数会弥补它呢？例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。给一个数组，返回它的最大连续子序列的和，你会不会被他忽悠住？(子向量的长度至少是1)

分析：设置一个临时变量，当它的和小于0时，就把它赋值为num，否则它就加上num，然后真正的最大值maxsum一直取max

24ms
```python
    def FindGreatestSumOfSubArray(self, array):
        if not array:
            return
        max_sum = array[0]
        cur_sum = array[0]
        for num in array[1:]:
            if cur_sum < 0:
                cur_sum = num
            else:
                cur_sum += num
            max_sum = max(max_sum, cur_sum)
        return max_sum
```

# 剑指 31. 整数中1出现的次数（从1到n整数中1出现的次数）
求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。

分析：
一、暴力方法
53ms
```python
    def NumberOf1Between1AndN_Solution(self, n):
        if n<1:
            return 0
        if n==1:
            return 1
        res = 1
        for num in range(2, n+1):
            i = num
            while i>0:
                if i%10==1:
                    res += 1
                i = i//10
        return res
```

# 剑指 32. 把数组排成最小的数
输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。

分析：先把int类型的list转换成str类型的list，然后比较list中的字符串两两拼接进行比较，如果ss[j]+ss[i]< ss[i]+ss[j]，就把两个字符串元素交换。

```python
    def jz32(self, numbers):
        if not numbers:
            return 0
        if len(numbers)==1:
            return numbers[0]
        N = len(numbers)
        ss = [str(n) for n in numbers]
        for i in range(N-1):
            for j in range(i+1, N):
                if ss[i] + ss[j] > ss[j] + ss[i]:
                    ss[i], ss[j] = ss[j], ss[i]
        return ''.join(ss)
```

# 剑指 33. 丑数
把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。

分析：一个丑数肯定是之前的某个丑数乘上2 或3 或5得到的

将丑数序列中的最后一个数，即最大的元素，记为bigM。每次往丑数序列中添加元素，将丑数序列中每个元素分别乘2得到一个大于bigM的数，乘3得到一个大于bigM的数，乘5得到一个大于bigM的数，取这3个数中的最小数。
这里有一个优化，你不需要每次都将已有的丑数序列中的每个数进行乘，你这一次添加时，某个元素\*2得到大于bigM的数，下一次就不需要再遍历这个元素之前的元素了，因为不会得到大于bigM的值，所以接着上次遍历到的元素继续遍历就行了。

边界：
- N<=0时，return 0
- 元素\*2或3或5等于bigM是不可以的，需要得到一个大于bigM的数

24ms
```python
    def jz33(self, N):
        if N<=0:
            return 0
        ugly_arr = [1]
        idx2 = 0
        idx3 = 0
        idx5 = 0
        for i in range(1, N):
            ugly_arr.append(min(ugly_arr[idx2]*2, ugly_arr[idx3]*3, ugly_arr[idx5]*5))
            bigM = ugly_arr[-1]
            while ugly_arr[idx2]*2<=bigM:
                idx2 += 1
            while ugly_arr[idx3]*3<=bigM:
                idx3 += 1
            while ugly_arr[idx5]*5<=bigM:
                idx5 += 1
        return ugly_arr[-1]
```

# 剑指 34. 第一个只出现一次的字符
在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 如果没有则返回 -1（需要区分大小写）

分析：先遍历一遍s把每个字符出现的次数存在dict里，然后再遍历一遍s，如果字符在dict里出现了一次就输出下标

注意：
- 第二次也是遍历s而不是dict，字典中元素的存储是无序的

32ms
```python
    def jz34(self, s):
        if not s:
            return -1
        dic = {}
        for ch in s:
            if ch in dic:
                dic[ch] += 1
            else:
                dic[ch] = 1
        for i in range(len(s)):
            if dic[s[i]]==1:
                return i
        return -1
```
# 剑指 35. 数组中的逆序对

分析：暴力搜索O(n^2)只能通过25%，用库函数sorted得到一个有序数组，然后一一比对，只能通过50%。
另一种方法是利用[归并排序](https://yugutouoo.github.io/2019/04/04/%E5%85%AB%E5%A4%A7%E6%8E%92%E5%BA%8F%E7%AE%97%E6%B3%95/)，在归并的过程中计算逆序对的个数。


```python
    def __init__(self):
        self.count = 0
    def InversePairs(self, data):
        if not data:
            return 0
        res = 0
        sorted_data = self.merge_sort(data)
        return self.count % 1000000007

    def merge_sort(self, data):
        if len(data)<=1:
            return data
        mid = len(data)//2
        A = self.merge_sort(data[:mid])
        B = self.merge_sort(data[mid:])
        return self.merge_ab(A, B)

    def merge_ab(self, A, B):
        C = []
        i, j = 0, 0
        M, N = len(A), len(B)
        while i<M and j<N:
            if A[i]<=B[j]:
                C.append(A[i])
                i += 1
            elif B[j]<A[i]:
                C.append(B[j])
                self.count += len(A)-i
                j += 1
        if i==M:
            for num in B[j:]:
                C.append(num)
        if j==N:
            for num in A[i:]:
                C.append(num)
        return C
```
设置了count变量来保存归并排序过程中，左边数组A[i]>右边数组B[j]时的逆序对，如对有序A=[1 4 5]和B=[2 3 8]进行归并排序时，添加到新数组C中去，只有当添加B中元素时才需要计算逆序对，如添加到C=[1]，需要从B中添加元素2了，这时计算数组A中还剩几个元素，就是2对应的逆序对，所以在代码merge_ab()中，只有B[j]>A[i]时才更新了count。
需要注意的是，如果你不在merge_ab()中修改count值，而是利用归并排序得到有序数组，再去一一比较，会超时。

# 剑指 36. 两个链表的第一个公共节点
输入两个链表，找出它们的第一个公共结点。

分析：两个单向链表的如果存在公共节点，那么从第一个公共节点开始后边的所有节点都是相同的，如图所示。因为公共节点除了`val`相同，`next`也相同。

![](/images/jz36.png)

分析：
第一个思路是，先遍历一遍得到两个链表的长度M和N，然后让长的（比如M>N）先走M-N步，然后再两个指针一起走，那么就可以得到第一个公共节点。
29ms
```python
    def jz36(self, pHead1, pHead2):
        if not pHead1 or not pHead2:
            return None
        h1 = pHead1
        h2 = pHead2
        M, N = 0, 0
        dif = M
        while h1:
            M += 1
            h1 = h1.next
        while h2:
            N += 1
            h2 = h2.next
        dif = abs(M-N)
        longhead = pHead1 if M>N else pHead2
        shorthead = pHead2 if M>N else pHead1
        i = 0
        while i<dif:
            longhead = longhead.next
            i += 1
        while longhead!=shorthead:
            print('in')
            longhead = longhead.next
            shorthead = shorthead.next
        return longhead
```
第二个思路是：
<u>1 2 3 6 7 4 5</u> 6 7
<u>4 5 6 7 1 2 3</u> 6 7
两个指针一起走，当p1走到终点时，就让他指向第二个链表的头；当p2走到终点时，就让p2指向第一个链表的头。

**边界控制：如果没有公共节点的话，最后两个指针都会指向None，相同的None节点会退出～**

24ms
```python
    def jz36_2(self, pHead1, pHead2):
        if not pHead1 or not pHead2:
            return None
        p1, p2 = pHead1, pHead2
        while p1!=p2:
            p1 = p1.next if p1 else pHead2
            p2 = p2.next if p2 else pHead1
        return p1
```
# 剑指 37. 数字在排序数组中出现的次数
统计一个数字在排序数组中出现的次数。
分析：题目说了有序，那么会想到二分查找。（如果顺序查找时间复杂度是O(n)）
首先查找key值出现的首位置，然后查找key值出现的末位置，进行计算。

边界：
- `while l<=r`有等号，比如arr=[1]，key=1，就是这种情况
- 当`arr[m]==key`时，**如果m==l或者m左边没有和key相等的了**，那么m就是key出现的首位置；如果不是这种情况下arr[m]==key，则说明m左边还有key值，则更新r边界。
- l和r更新都要+1或-1
22ms
```python
    def GetNumberOfK(self, arr, key):
        if not arr:
            return 0
        first_idx = self.findFirstK(key, arr)
        last_idx = self.findLastK(key, arr)
        if first_idx==-1 or last_idx==-1:
            return 0
        return last_idx - first_idx + 1

    def findFirstK(self, key, arr):
        l = 0
        r = len(arr)-1
        while l<=r:
            m = (l+r)//2
            if key<arr[m]:
                r = m-1
            elif key>arr[m]:
                l = m+1
            else: #arr[m]==key了
                if l==m or arr[m-1]!=key:
                    return m
                else:
                    r = m-1
        return -1
            
    def findLastK(self, key, arr):
        l = 0
        r = len(arr)-1
        while l<=r:
            m = (l+r)//2
            if key>arr[m]:
                l = m+1
            elif key<arr[m]:
                r = m-1
            else:
                if m==r or arr[m+1]!=key:
                    return m
                else:
                    l = m+1
        return -1 
```
# 剑指 38. 二叉树的深度
输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。

分析：求二叉树的深度，最经典的递归。
如果root为None，就返回0，否则它就是一层，返回1+max(左子树深度， 右子树深度)

23ms
```python
    def TreeDepth(self, root):
        if not root:
            return 0
        return 1+max(self.TreeDepth(root.left), self.TreeDepth(root.right))
```
# 剑指 39. 平衡二叉树
输入一棵二叉树，判断该二叉树是否是平衡二叉树。

分析：对于每一个节点，计算它的左右子树的深度，如果差的绝对值超过1则return False；如果差的绝对值不超过1，则对该节点的左子树（以root.left为根节点）进行同样的判断，右子树同理。只有当左右子树都是平衡二叉树时才能return True

26ms
```python
    def IsBalanced_Solution(self, root):
        if not root:
            return True
        l_depth = self.getDepth(root.left)
        r_depth = self.getDepth(root.right)
        if abs(l_depth - r_depth)>1:
            return False
        return self.IsBalanced_Solution(root.left) and self.IsBalanced_Solution(root.right)

    def getDepth(self, root):
    '''计算以root为根节点的二叉树的深度
    '''
        if not root:
            return 0
        return 1+max(self.getDepth(root.left), self.getDepth(root.right))
```
# 剑指 40. 数组中只出现一次的数字
一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。

分析：
第一种思路，利用dict，先遍历一次记录次数，再遍历一次找到出现次数为1的数字

27ms
```python
    def FindNumsAppearOnce(self, arr):
        dic = {}
        res = []
        for num in arr:
            if num in dic:
                dic[num]+=1
            else:
                dic[num]=1
        for k, v in dic.items():
            if v==1:
                res.append(k)
                if len(res)==2:
                    return res
        return res
```
第二种思路，如果是一个数组中只有1个数只出现了一次，其它数字都出现了2次，那么利用异或可以很容易的解出来。
那如果是有两个数字只出现了1次呢？
还是利用异或，当所有数异或完之后，剩下的就是只出现一次的两个数字AB异或的结果。假如这两个数字是2和8，这时候，我们得到这个异或的结果中，二进制倒数第1个不相同的位，如
> 5: 0 1 0 0
8: 1 0 0 0

5和8的倒数第3位不相同，我们就依据每个数字的二进制的倒数第3位是1还是0进行分组，每组内进行异或，由于相同数字异或的结果肯定是0，这就变成了上边的问题。这里的重点是怎么根据第idx位的二进制位进行分组？
如上例idx=3，将num右移2位，然后与1进行&操作，就可以判断啦～
> 7: 0 1 1 1
7>>2: 0 0 1 1 

`(7>>2)&1`的结果是1，所以把7分到idx位为1的那一组

23ms
```python
    def jz40_2(self, arr):
        if not arr:
            return []
        xor = 0
        idx = 0
        for num in arr:
            xor ^= num
        while xor&1!=1:
            idx += 1
            xor = xor>>1
        num1 = 0
        num2 = 0
        for num in arr:
            #判断num的第idx位是不是1
            if (num >> idx)&1 :
                num1 ^= num
            else:
                num2 ^= num
        return [num1, num2]
```
# 剑指 41. 和为S的连续正数序列
小明很喜欢数学,有一天他在做数学作业时,要求计算出9~16的和,他马上就写出了正确答案是100。但是他并不满足于此,他在想究竟有多少种连续的正数序列的和为100(至少包括两个数)。没多久,他就得到另一组连续正数和为100的序列:18,19,20,21,22。现在把问题交给你,你能不能也很快的找出所有和为S的连续正数序列? Good Luck!
输出描述:
输出所有和为S的连续正数序列。序列内按照从小至大的顺序，序列间按照开始数字从小到大的顺序

分析：滑动窗口
先让i指向第一个数，j指向第二个数，**csum一直作为滑动窗口的和**。
- 当滑动窗口内元素的和小于tsum时，j后移，csum加上窗口后移增加的元素；
- 当滑动窗口内元素的和等于tsum时，先在res中增加这个窗口，再将csum中减去i，然后将i前移（顺序不能错）；
- 当滑动窗口内的和大于tsum时，先从滑动窗口内减去i，再将i前移（顺序不能错）。
后两步操作可以合并。

25ms
```python
    def jz41(self, tsum):
        if tsum<3:
            return 0
        i, j = 1, 2
        res = []
        csum = i + j
        while i<tsum//2+1 and j<tsum//2+1:
            if csum<tsum:
                j += 1
                csum += j
            else:
                if csum==tsum:
                    res.append([num for num in range(i, j+1)])
                csum -= i
                i += 1
        return res
```

# 剑指 42. 和为S的两个数字
输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。
输出描述:
对应每个测试案例，输出两个数，小的先输出。

分析：设头尾两个指针

26ms
```python
    def FindNumbersWithSum(self, array, tsum):
        if not array:
            return []
        i, j = 0, len(array)-1
        while i<j:
            if array[i]+array[j]<tsum:
                i += 1
            elif array[i]+array[j]>tsum:
                j -= 1
            else:
                return [array[i], array[j]]
        return []
```
# 剑指 43. 左旋转字符串
汇编语言中有一种移位指令叫做循环左移（ROL），现在有个简单的任务，就是用字符串模拟这个指令的运算结果。对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。是不是很简单？OK，搞定它！

分析：字符串操作用python做很简单

24ms
```python
    def LeftRotateString(self, s, n):
        if not s:
            return ''
        if n==0:
            return s
        return s[n:]+s[:n]
```

# 剑指 44. 翻转单词顺序列
牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，写些句子在本子上。同事Cat对Fish写的内容颇感兴趣，有一天他向Fish借来翻看，但却读不懂它的意思。例如，“student. a am I”。后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。Cat对一一的翻转这些单词顺序可不在行，你能帮助他么？

分析：python操作字符串很简单
三步，1.把s以空格分割变成list，`lst = list(s.split(' '))`
    2.将list反转，`rev = lst[::-1]`
    3.将反转后的list以空格连接，合并为字符串，`' '.join(rev)`

30ms
```python
    def ReverseSentence(self, s):
        if not s:
            return ''
        lst = list(s.split(' '))
        reversed_lst = lst[::-1]
        return ' '.join(reversed_lst)
```

# 剑指 45. 扑克牌顺子
LL今天心情特别好,因为他去买了一副扑克牌,发现里面居然有2个大王,2个小王(一副牌原本是54张^_^)...他随机从中抽出了5张牌,想测测自己的手气,看看能不能抽到顺子,如果抽到的话,他决定去买体育彩票,嘿嘿！！“红心A,黑桃3,小王,大王,方片5”,“Oh My God!”不是顺子.....LL不高兴了,他想了想,决定大\小 王可以看成任何数字,并且A看作1,J为11,Q为12,K为13。上面的5张牌就可以变成“1,2,3,4,5”(大小王分别看作2和4),“So Lucky!”。LL决定去买体育彩票啦。 现在,要求你使用这幅牌模拟上面的过程,然后告诉我们LL的运气如何， 如果牌能组成顺子就输出true，否则就输出false。为了方便起见,你可以认为大小王是0。

分析：问题是判断给定的牌s是不是顺子。先把s进行排序，然后遍历对0进行计数，同时计算相邻数字之间的差，如0 0 1 4 5，0有2个，1和4之间差2个数，4和5之间不差数。最后判断0的个数是否足够填补所有相邻数字之间的差。
这中间如果碰到两个相等的数字，则肯定不是顺子

24ms
```python
    def IsContinuous(self, numbers):
        if not numbers:
            return False
        numbers.sort()
        gaps = 0
        zeros = 0
        for i in range(len(numbers)-1):
            if numbers[i] == 0:
                zeros += 1
            elif numbers[i+1]==numbers[i]:
                return False
            else:
                gap = numbers[i+1]-numbers[i]-1
                gaps += gap
        return True if gaps<=zeros else False
```

# 剑指 46. 孩子们的游戏(圆圈中最后剩下的数)
每年六一儿童节,牛客都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此。HF作为牛客的资深元老,自然也准备了一些小游戏。其中,有个游戏是这样的:首先,让小朋友们围成一个大圈。然后,他随机指定一个数m,让编号为0的小朋友开始报数。每次喊到m-1的那个小朋友要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,从他的下一个小朋友开始,继续0...m-1报数....这样下去....直到剩下最后一个小朋友,可以不用表演,并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。请你试着想下,哪个小朋友会得到这份礼品呢？(注：小朋友的编号是从0到n-1)

分析：约瑟夫环问题
**递推公式： $p_i = (p_{i-1}+m)\%i$**，**第一个p的值为1，不管n个人的下标从0开始到n-1还是从1开始到n，都是求到$p_n$**

24ms
```python
    def LastRemaining_Solution(self, n, m):
        if m<=0 or n<=0:
            return -1
        res = 1
        for i in range(1, n+1):
            res = (res+m)%i
        return res
```

# 剑指 47. 求1+2+3+...+n
求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。

分析：$f(n) = n + f(n-1)$，但是n==0时需要返回0，这样才能求出f(1)，利用 `n and ...`操作来实现当n==0时return 0

23ms
```python
    def Sum_Solution(self, n):
        # write code here
        return n and self.Sum_Solution(n - 1) + n
```

# 剑指 48. 不用加减乘除做加法
写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。

分析：那就只能用位运算了
^ 相当于两个数相加
&操作的结果左移1位 相当于两个数相加过程中的进位

但是，在python中，当一个整数和一个负数相加时出现了死循环，其实问题出在tmp = num1^num2这条语句中。
实际上，在进行负数的按位加法时，有可能发生在最高位还要向前进一位的情形，正常来说，这种进位因为超出了一个int可以表示的最大位数，应该舍去才能得到正确的结果。因此，对于Java，c，c++这样写是正确的。而对于Python，却有点不同。
改正的代码，可以每次都把num1规定成一个32位的数
32个1也就是一个int可表示的无符号整数为4294967295，对应的有符号为-1。因此最后我们可以判断符号位是否为1做处理。
Python代码，30ms
```python
    def Add(self, num1, num2):
        # write code here
        while num2 != 0:
            temp = num1 ^ num2
            num2 = (num1 & num2) << 1
            num1 = temp & 0xFFFFFFFF
        return num1 if num1 >> 31 == 0 else num1 - 4294967296
```
Java代码，13ms
```java
    public int Add(int num1,int num2) {
        while(num2 != 0){
            //无进位，和为1的位
            int temp = num1 ^ num2;
            //有进位，即两个都为1，算出来以后左移1位
            num2 = (num1 & num2) << 1;
            num1 = temp;
        }
        return num1;
    }
```

# 剑指 49. 把字符串转换成整数
将一个字符串转换成一个整数(实现Integer.valueOf(string)的功能，但是string不符合数字要求时返回0)，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0。
输入描述:
> 输入一个字符串,包括数字字母符号,可以为空

输出描述:
> 如果是合法的数值表达则返回该数字，否则返回0

**示例1**
输入
> +2147483647
    1a33

输出
> 2147483647
    0

分析：这里只考虑了数字位和符号位。把字符'0'和数字0放在字典里。这个题主要是考虑不合理的情况，包括
- 首位置出现了符号位，后边又一次出现
- 符号位出现了一次，但是在不是首位置的其它地方
- 字符串s中只有一个符号位，没有数字
- 出现了字母，直接return 0
最后计算的时候，如果首字符是'-'则对结果取负，如果是'+'或者没有符号位，就直接返回该值。

30ms
```python
def StrToInt(s):
    # 函数里用全局变量，需先声明它是个全局变量
    if not s:
        return 0
    dic = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}
    res = []
    flag = False
    for ch in s:
        if ch in dic: #数字
            res.append(dic[ch])
        elif ch=='+' or ch=='-': # 符号位的各种不合理情况
            if flag:
                return 0
            elif s.index(ch)!=0:
                return 0
            else:
                flag = True
        else:
            return 0
    if flag and len(res)==0: # 只有一个符号位没有数字
        return 0
    resnum = 0
    for i in res:
        resnum = resnum*10+i
    return -resnum if s[0]=='-' else resnum
```
# 剑指 50. 数组中重复的数字
在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。
分析：
思路1，利用hash
21ms
```python
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
def duplicate(numbers, duplication):
    dic = {}
    for num in numbers:
        if num not in dic:
            dic[num] = 1
        else:
            dic[num] += 1
            duplication[0] = num
            return True
    return False
```
思路2，排序，相邻元素相同
思路3，题目说了数字范围在1-N-1，可以利用数字和下标映射，数字i对应下标N-1-i

26ms
```python
def duplicate_2(numbers, duplication):
    N = len(numbers)
    visit = [0]*N
    for num in numbers:
        idx = N-1-num
        if visit[idx]==1:
            duplication[0] = num
            return True
        else:
            visit[idx] = 1
    return False
```

# 剑指 51. 构建乘积数组
给定一个数组A`[0,1,...,n-1]`,请构建一个数组B`[0,1,...,n-1]`,其中B中的元素`B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]`。不能使用除法。

分析：把B[i]分成两部分， 一部分是`A[0,…,i-1]`的连乘，记为C[i]，一部分是`A[i+1,…,n-1]`的连乘，记为D[i]，所以，`C[i]=C[i-1] A[i-1]`， `D[i]=D[i+1] A[i+1]`。

![](/images/jz51.png)

21ms
```python
def multiply(A):
    N = len(A)
    B = [1]*N
    for i in range(1, N):
        B[i] = B[i-1] * A[i-1]
    tmp = 1
    for i in range(N-2, -1, -1):
        tmp = tmp * A[i+1]
        B[i] = B[i] * tmp
    return B
```

# 剑指 54. 字符流中第一个不重复的字符
请实现一个函数用来找出字符流中第一个只出现一次的字符。例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。

分析：Insert函数依次读取字符串中的每个字符，FirstAppearingOnce函数可以得到读取到字符串的哪个字符了就返回到它为止的第一个只出现一次的字符， 即读取到'g'返回'g'，读取到'go'返回'g'，读取到'goo'返回'g'，读取到'goog'返回'#'，读取到'googl'返回'l'

你爱读到哪里读到哪里，我用一个字典保存读到这里的每个字符的个数，然后遍历字符串到这个字符，中间碰到val为1的字符就return
可以用字典， 更简单的是用Counter大法好
```python
    def __init__(self):
        self.words = []
    # 到读取到的这个字符为止，return第一个只出现一次的字符
    def FirstAppearingOnce(self):
        c = Counter(self.words)
        for ch in self.words:
            if c[ch] == 1:
                return ch
        return '#'
    # 读取字符
    def Insert(self, char):
        self.words.append(char)
```
# 剑指 55. 链表中环的入口结点
给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null。
如何判断是否有环？？？
利用快慢指针找到环中的任意一个结点meetNode。（一个每次走一步，一个每次走两步，只要快慢指针不等就一直走，如果有环最后肯定会相遇在环内的某个节点，如果没有环最后两个都会指向None）
然后，如何根据环中任意位置的结点meetNode，找到环的入口？？
将meetNode记录，用一个指针去遍历，获得环的长度N
那么，有环并且知道环的长度N了，怎么确定环的入口？？
在链表的头节点head处设一个指针cur，先让它走N步，然后在head处再设一个指针slow，让cur和slow一起走，相遇的结点就是环的入口。原理如图，最好记住：
![](/images/jz55.png)

38ms
```python
    def EntryNodeOfLoop(self, pHead):
        # 先判断是否有环
        if not pHead or not pHead.next:
            return None
        fast = pHead.next
        slow = pHead
        while fast!=slow:
            fast = fast.next.next
            slow = slow.next
        # 如果有环，slow==fast也就是环内的一个节点
        # 先遍历一遍得到环的长度N
        if not fast:
            return None
        cur = slow.next
        N = 1
        while cur!=slow:
            cur = cur.next
            N += 1
        # 然后cur指向head，先走N步
        cur = pHead
        while N>0:
            cur = cur.next
            N -= 1
        # 然后两个指针一起走，一个从head走，一个从cur停在的地方走
        fast = pHead
        while fast!=cur:
            fast = fast.next
            cur = cur.next
        return cur
```

UPDATE简化写法，不再需要判断环的长度了，，，，并且slow和fast的初始位置是一样的，养成好习惯
```python
    def EntryNodeOfLoop(self, pHead):
        if not pHead or not pHead.next:
            return None
        slow = pHead
        fast = pHead
        while fast:
            slow = slow.next
            fast = fast.next.next
            if slow==fast:
                break
        print(slow.val)
        cur = pHead
        while cur!=slow:
            cur = cur.next
            slow = slow.next
        return cur
```






# 剑指 56. 删除链表中重复的结点
在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5

分析： 利用有序， 遍历链表，如果用pre存上一个节点，cur是当前节点，才能实现删除。
如果当前结点的值与下一个结点的值相同，那么它们就是重复的结点，**但是还不可以马上删除**，因为如果再后边一个结点的值也相同，也要删，所以遇到相同的了还要一直往后找。
注意：
- 代码中的`while cur and cur.next`，不能只判断cur.next而不判断cur，会出错的
- 当存在相同节点时，往后遍历直到不同，再更新两个节点的指向；当不存在节点时，直接更新pre和cur节点。

24ms
```python
    def deleteDuplication(self, pHead):
        if not pHead:
            return None
        if not pHead.next:
            return pHead
        cur = pHead
        precur = ListNode(-1)
        precur.next = cur
        pre = precur
        while cur and cur.next:
            if cur.val == cur.next.val:
                while cur.next and cur.val==cur.next.val:
                    cur = cur.next
                pre.next = cur.next
                cur = cur.next
            else:
                pre = cur
                cur = cur.next
        return precur.next
```

一开始没看清题目中的有序链表，是想着遍历一遍链表把元素和个数存在dic里，然后再遍历一遍链表专门删除val>1的节点。

24ms
```python
    def deleteDuplication(self, pHead):
        dic = {}
        if not pHead:
            return None
        if not pHead.next:
            return pHead
        cur = pHead
        while cur:
            if cur.val in dic:
                dic[cur.val] += 1
            else:
                dic[cur.val] = 1
            cur = cur.next
        cur = pHead
        precur = ListNode(-1)
        precur.next = cur
        pre = precur
        while cur:
            if dic[cur.val] > 1:
                pre.next = cur.next
            else:
                pre = cur
            cur = cur.next
        return precur.next
```
# 剑指 57. 二叉树的下一个结点
给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。

分析：三种情况，
+ 当前节点有右子树的话，中序下一个结点是**右子树中的最左子节点**；
+ 当前节点无右子树但是是父节点的左子节点，中序下一个节点是**当前结点的父节点**；
+ 当前节点无右子树而且是父节点的右子节点，则一直向上遍历，如果‘我’的父节点也是它的父节点的右孩子，就**一直向上，直到找到某个pNode结点它不是父节点的右子树**。（如图所示，I结点是符合第三种情况的结点，我们要把pNode定位到B，**如果B存在父节点**，那么B的父节点就是中序的下一个节点）
![](/images/jz57.png)
注意：
- 代码中if语句用到node.next.**left**属性时，需要先判断node.next不为空，否则会出错的
- 最后一种情况要注意，**当定位节点存在父节点时**，pNext才更新为其父节点，否则pNext为None
31ms
```python
    def GetNext(self, pNode):
        if not pNode:
            return None
        node = pNode
        pNext = None
        # node有右子树：则中序下一个节点是右子树的最左
        if node.right:
            node = node.right
            while node.left:
                node = node.left
            pNext = node
        # node没右子树，但是是父节点的左孩子：则中序下一个节点是父节点
        else:
            if node.next and node.next.left == node:
                pNext = node.next
        # node没右子树，但是是父节点parent的右孩子：则中序下一个节点要这么找：一直往上找，直到node不再是它的父节点的右孩子为止，位于xnode
        # 这时候，如果xnode有父节点，则返回父节点，否则返回None
            elif node.next and node.next.right == node:
                while node.next and node.next.right == node:
                    node = node.next
                if node.next:
                    pNext = node.next
        return pNext
```
# 剑指 58. 对称的二叉树
请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。

分析：判断二叉树是不是对称的，也就是二叉树和其镜像是否相同
利用递归来做，初始二叉树和其镜像的根节点都是root，然后比较原始二叉树root的left和镜像root的right是不是相同，并对每个节点root做相同判断
26ms
```python
    def isSymmetrical(self, pRoot):
        return self.isSymHelper(pRoot, pRoot)
    def isSymHelper(self, root1, root2):
        if not root1 and not root2:
            return True
        if not root1 or not root2:
            return False
        if root1.val != root2.val:
            return False
        return self.isSymHelper(root1.left, root2.right) and self.isSymHelper(root1.right, root2.left)
```
# 剑指 59. 按之字形顺序打印二叉树
请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。

分析：level traverse的变形，层次遍历用到队列，还有一个把每个层次的元素分开的题[LeetCode 102. Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)，和这个题是一样的，只是要设一个flag指示是存每层的从左往后还是从右往左存，每次更新flag

```python
    def Print(self, pRoot):
        if not pRoot:
            return []
        queue = []
        res = []
        queue.append(pRoot)
        out_cnt = 1
        in_cnt = 0
        flag = False # flag为False，从左往右；flag为True，从右往左
        while queue:
            tmp = []
            while out_cnt > 0:
                top = queue[0]
                tmp.append(top.val)
                queue.pop(0)
                out_cnt -= 1
                if top.left:
                    queue.append(top.left)
                    in_cnt += 1
                if top.right:
                    queue.append(top.right)
                    in_cnt += 1 
            out_cnt = in_cnt
            in_cnt = 0
            if not flag:
                res.append(tmp)
            else:
                res.append(tmp[::-1]) # python 666
            flag = not flag
        return res
```
# 剑指 60. 把二叉树打印成多行
从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。

分析：比59题还简单，[LeetCode 102. Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)原题

26ms
```python
    def Print(self, pRoot):
        if not pRoot:
            return []
        queue = []
        res = []
        queue.append(pRoot)
        out_cnt = 1
        in_cnt = 0
        while queue:
            tmp = []
            while out_cnt > 0:
                top = queue[0]
                tmp.append(top.val)
                queue.pop(0)
                out_cnt -= 1
                if top.left:
                    queue.append(top.left)
                    in_cnt += 1
                if top.right:
                    queue.append(top.right)
                    in_cnt += 1 
            out_cnt = in_cnt
            in_cnt = 0
            res.append(tmp)
        return res
```

# 剑指 61. 序列化二叉树
请实现两个函数，分别用来序列化和反序列化二叉树

分析：二叉树的序列化就是把节点存到一个字符串里，空节点对应'#'，这里节点如果是整数还需要转换成str再存；你可以用某种特定遍历顺序来实现序列化和反序列化。在这里用某个字符' '或','分隔字符串中的值都行
序列化：用先序递归遍历
反序列化有一丢丢难：也是用递归

原题 [LeetCode 297. Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)

24ms
```python
    def Serialize(self, root):
        vals = []
        def preTraverse(root):
            if not root:
                vals.append('#')
            else:
                vals.append(str(root.val)) # root.val是整数，需要先变成str
                preTraverse(root.left)
                preTraverse(root.right)
        preTraverse(root)
        return ' '.join(vals) # 这里以空格分隔字符

    def Deserialize(self, s):
        vals = [v for v in s.split()]
        def build():
            if vals:
                val = vals[0]
                vals.pop(0)
                if val == '#':
                    return None
                root = TreeNode(int(val)) # val是字符型的，需要还原int
                root.left = build()
                root.right = build()
                return root
        return build()
```
# 剑指 62. 二叉搜索树的第k个结点

给定一棵二叉搜索树，请找出其中的第k小的结点。例如， （5，3，7，2，4，6，8）    中，按结点数值大小顺序第三小结点的值为4。

分析：中序遍历把节点存到一个list里，再取第k个就好了
当k<=0或大于树中节点个数return None

34ms
```python
    def KthNode(self, pRoot, k):
        if k<=0:
            return None
        res = []
        def inTraverse(root):
            if not root:
                return
            inTraverse(root.left)
            res.append(root)
            inTraverse(root.right)
        inTraverse(pRoot)
        N = len(res)
        if k>N:
            return None
        return res[k-1]
```
# 剑指 63. 数据流中的中位数
如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。我们使用Insert()方法读取数据流，使用GetMedian()方法获取当前读取数据的中位数。


# 剑指 64. 滑动窗口的最大值
给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个： {[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}， {2,3,[4,2,6],2,5,1}， {2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}。

分析：固定大小的滑动窗口很简单

```python
    def maxInWindows(self, num, size):
        if not num or size == 0:
            return []
        N = len(num)
        K = size
        res = []
        for i in range(N-K+1):
            res.append(max(num[i:i+K]))
        return res
```
# ★ 剑指 65. 矩阵中的路径
请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。如果一条路径经过了矩阵中的某一个格子，则之后不能再次进入这个格子。 例如 a b c e s f c s a d e e 这样的3 X 4 矩阵中包含一条字符串"bcced"的路径，但是矩阵中不包含"abcb"路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子。

分析：dfs经典
需要设一个visit数组来表示某元素是否已经被访问过
遍历到这个元素了，如果ch和判断到s的第那个元素匹配，就用ch，并把ch的visit置为True。然后对这个元素的上下左右都做同样的判断，hasPath表示上下左右中是不是有一个符合的，如果一个符合的都没有，hasPath是False，如果有一个符合的，hasPath是True
1.将matrix字符串模拟映射为一个字符矩阵(但并不实际创建一个矩阵)
2.取一个boolean[matrix.length]标记某个字符是否已经被访问过,用一个布尔矩阵进行是否存在该数值的标记。
3.如果没找到结果，需要将对应的boolean标记值置回false,返回上一层进行其他分路的查找。

43ms
```python
    def hasPath(self, matrix, rows, cols, path):
        if not matrix or not path:
            return False
        visit = [False]*(rows*cols)
        for row in range(rows):
            for col in range(cols):
                if self.dfs_hasPath(matrix, path, 0, rows, cols, row, col, visit):
                    return True
        return False
    def dfs_hasPath(self, matrix, path, pathlength, rows, cols, row, col, visit):
        if pathlength == len(path):
            return True
        hasPath = False
        if row<rows and row>=0 and col<cols and col>=0 and matrix[row*cols+col]==path[pathlength] and visit[row*cols+col]==False:
            pathlength += 1
            visit[row*cols+col] = True
            hasPath = self.dfs_hasPath(matrix, path, pathlength, rows, cols, row+1, col, visit)\
                or self.dfs_hasPath(matrix, path, pathlength, rows, cols, row-1, col, visit)\
                or self.dfs_hasPath(matrix, path, pathlength, rows, cols, row, col+1, visit)\
                or self.dfs_hasPath(matrix, path, pathlength, rows, cols, row, col-1, visit)
            if not hasPath:
                pathlength -= 1
                visit[row*cols+col] = False
        return hasPath
```
# 剑指 66. 机器人的运动范围
地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，但是不能进入行坐标和列坐标的数位之和大于k的格子。 例如，当k为18时，机器人能够进入方格（35,37），因为3+5+3+7 = 18。但是，它不能进入方格（35,38），因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？

分析： dfs经典问题
把方格看成一个`m*n`的矩阵，从`（0，0）`开始移动。当准备进入坐标`(i, j)`是，通过检查坐标的数位来判断机器人能否进入。如果能进入的话，接着判断四个相邻的格子。

32ms
```python
    def movingCount(self, threshold, rows, cols):
        visit = [[0 for i in range(cols)] for j in range(rows)]
        res = self.dfs_movingCount(threshold, rows, cols, visit, 0, 0)
        return res
    def dfs_movingCount(self, threshold, rows, cols, visit, i, j):
        count = 0
        if i<rows and i>=0 and j<cols and j>=0:
            if self.isCanMove(threshold, i, j) and visit[i][j] == 0:
                visit[i][j] = 1
                count = 1 + self.dfs_movingCount(threshold, rows, cols, visit, i+1, j)\
                    + self.dfs_movingCount(threshold, rows, cols, visit, i-1, j)\
                    + self.dfs_movingCount(threshold, rows, cols, visit, i, j+1)\
                    + self.dfs_movingCount(threshold, rows, cols, visit, i, j-1)
        return count
    def isCanMove(self, threshold, row, col): # 能否进入
        res = 0
        while row>0:
            mod = row % 10
            res += mod
            row = row//10
        while col > 0:
            mod = col % 10
            res += mod
            col = col//10
        if res > threshold:
            return False
        else:
            return True
```