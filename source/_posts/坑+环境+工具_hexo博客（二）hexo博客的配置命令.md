---
title: hexo博客（二）hexo博客的配置命令
date: 2019-04-01 19:28:37
tags:
- Hexo
- 环境配置
categories: 坑+环境+工具
---

在 Hexo 中有两份主要的配置文件，其名称都是 _config.yml。 其中，一份位于站点根目录下，主要包含 Hexo 本身的配置；另一份位于主题目录下，这份配置由主题作者提供，主要用于配置主题相关的选项。

为了描述方便，在以下说明中，将前者称为 **站点配置文件**， 后者称为 **主题配置文件**。
> 注意：在冒号后面一定要加上一个空格，否则在生成静态文件的时候会报错，并且也不能将其成功推送到github。

# 发布新的博客
+ 新建
`hexo new 'blogname'`
在 `source/_posts`文件夹下就会出现'blogname.md'的文档

+ 发布
```
hexo clean
hexo generate
hexo deploy
```
后边可以简写成`hexo d -g`

# MarkDown教程
参考http://xianbai.me/learn-md/index.html

# NexT主题配置
## 修改网站图标
将ico图标设置为 32*32 大小的图片，并将名称改为`favicon.ico`，然后把图标放在`/themes/next/source/images`里，并且修改主题配置文件：`medium: /images/favicon.ico`

## 阅读全文
在主题配置文件中找到auto_excerpt属性进行配置
```
auto_excerpt:
  enable: true #改写为true
  length: 150 #默认展示的高度
```
你也可以在自己的博文中添加`<!--more-->`来决定在首页展示到什么位置

## 添加分类
+ 生成“分类”页并添加type属性
在terminal终端进入博客所在的文件夹，执行
`hexo new page categories`
就可以在`/source文件夹下看到`categories`文件夹中的`index.md`，打开后添加`type: categories`
```
title: 分类
date: 2019-04-01 20:34:55
type: "categories"
```
保存关闭文件

+ 给文章添加“categories”属性
```
title: '博客题目'
date: 2019-04-01 19:28:37
categories:
- 工具
```
这样分类模块就做好了+.+
## 添加标签
和添加分类同理
+ 生成“标签”页并添加type属性
在terminal终端进入博客所在的文件夹，执行
`hexo new page tags`
就可以在`/source文件夹下看到`tags`文件夹中的`index.md`，打开后添加`type: tags`
```
title: 分类
date: 2019-04-01 20:34:55
type: "tags"
```
保存关闭文件

+ 给文章添加“tags”属性
```
title: '博客题目'
date: 2019-04-01 19:28:37
tags:
- 技巧
- 环境配置
```

## 目录设置全展开
在 `~/themes/next/source/css/_custom/custom.styl`中添加以下代码：

`.post-toc .nav .nav-child { display: block; }`

## 使文章多级目录自动展开，而不是默认折叠
[参考](https://github.com/iissnan/hexo-theme-next/issues/710)  
如果你想实现默认展开全部目录的功能，可以在themes/next/source/css/_custom/custom.styl文件中添加以下自定义样式规则：
```
.post-toc .nav .nav-child { 
    display: block; 
}
```
但是通常文章内会出现多级标题，对应的目录里就会有多级导航出现，这时候一些原本你不希望出现的次要标题也会在目录中出现并且无法折叠。可以通过以下样式实现默认只展开两级目录，这样以来就完美解决了该问题。
```
.post-toc .nav .nav-level-1>.nav-child { 
   display: block; 
}
```
## 分页显示问题
[参考](https://github.com/hexojs/hexo/issues/3794)

问题：如图，使用的Next主题，分页这里有些问题，分页显示1，2，3，……9，
```
<i class="fa fa-angle-right"></i>
```
解决方案：在主题文件夹目录`hexo-theme-next/layout/_partials/pagination.swig`下，修改`escape=false`   
即：
```
{%- if page.prev or page.next %}
  <nav class="pagination">
    {{
      paginator({
        prev_text: '<i class="fa fa-angle-left" aria-label="' + __('accessibility.prev_page') + '"></i>',
        next_text: '<i class="fa fa-angle-right" aria-label="' + __('accessibility.next_page') + '"></i>',
        mid_size : 1,
        escape   : false
      })
    }}
  </nav>
{%- endif %}
```
## 插入图片
+ 在`/source`目录下新建`images`文件夹，用于存放图片（如果你想在编辑MarkDown时自动提醒，需要把images文件夹添加到工作区）
![](/images/hexo_image.png)
+ 然后在`.md`中插入图片时，写如下代码：
`![](/images/hexo_image.png)`

## 插入公式
原生hexo并不支持数学公式，需要安装插件[mathJax](https://www.mathjax.org/)，用markdown语法来写公式。
- 安装
`$ npm install hexo-math --save`
- 在站点配置文件 _config.yml 中添加
```
math:
  engine: 'mathjax' # or 'katex'
  mathjax:
    # src: custom_mathjax_source
    config:
      # MathJax config
```
- 在 next 主题配置文件中 themes/next-theme/_config.yml 中将 mathJax 设为 true:
```
mathjax:
  enable: true
  per_page: false
  cdn: //cdn.bootcss.com/mathjax/2.7.1/latest.js?config=TeX-AMS-MML_HTMLorMML
```
- 公式的使用
`$w_i$`得到$w_i$




















