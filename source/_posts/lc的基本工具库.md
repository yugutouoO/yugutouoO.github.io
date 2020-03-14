---
title: lc的基本工具库
date: 2020-03-12 11:15:49
tags:
---
<!-- [[toc]] -->

## 求a,b的最大公约数

```python
def gcd(self,a,b):
    '''求a,b的最大公约数
    '''
    return a if b==0 else self.gcd(b, a%b)
```

**附 辗转相除法**  
两个正整数a和b（a>b），它们的最大公约数等于a除以b的余数c和b之间的最大公约数。