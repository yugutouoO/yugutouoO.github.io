---
title: hexo博客（一）macOS下使用Hexo+Github搭建自己的博客
date: 2019-04-01 16:53:53
tags: 
- Hexo
- 环境配置
categories: 坑+环境+工具
---

# 安装环境配置
- 安装git，使用homebrew安装
- 安装node.js，直接去官网下载安装包比较快也省事[node.js](https://nodejs.org/en/)

# 安装hexo
- 安装hexo
`sudo npm install -g hexo`
- 初始化，终端cd到一个指定目录，我的在`/Users/jingzhang/blogs`
`hexo init /User/jingzhang/blogs`
- 终端cd到该目录下，安装npm
`npm install`
- 此时开启hexo服务就可在本地预览博客主页了
`hexo s`

# 关联已有的github账号
- 首先在github中新建仓库，仓库名需要和你的账号对应，格式：`yourname.github.io`，拿我的举例：`yugutouoO.github.io`

- 在finder中打开hexo创建文件夹下的`_config.yml`配置文件，拉到最底下修改成这样
```
deploy:
  type: git
  repository: https://github.com/yugutouoO/yugutouoO.github.io.git
  branch: master
```
(注意所有的：后面要加上一个空格，否则hexo命令会报错)

- 在hexo文件夹目录下执行
`hexo g`
- 如果报错，则执行
`npm install hexo --save`
- 再执行命令
`hexo d`
- 如果报错，则执行
`npm install hexo-deployer-git --save`
然后再执行hexo g和hexo d
此时终端会提示输入github的用户名和密码，输入完成后就已经成功链接了github上的repository，这时浏览器输入yugutouoO.github.io就可以看到博客页面了

# 建立ssh
- 本地生成SSH密钥，命令行：
`ssh-keygen -t rsa -C buptzj615@gmail.com`

主要C后面的邮箱地址替换成自己的。然后应该回让设置密码，可以全部为空。然后我在我的/Users/用户名/.ssh路径下看到了两个新生成的东西：id_rsa和id_rsa.pub，我们需要复制.pub里面的全部内容，最好用sublime打开。

- 在github到account设置中，找到ssh设置，点击New SSH key,然后粘贴.pub里面的全部内容进去，然后Add SSH key.

- 然后在hexo文件夹内找到_config.yml配置文件，sublime打开编辑，下拉到最后，修改如下
```
deploy:
  type: git
  repository: https://github.com/yugutouoO/yugutouoO.github.io
  branch: master
```
# 修改主题
http://theme-next.iissnan.com/getting-started.html


