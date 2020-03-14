---
title: 搭建Vultr VPS 开启SSR科学上网
date: 2020-02-19 19:53:33
tags:
- Hexo
- 环境配置
categories: 坑+环境+工具
---

**搭建服务器科学上网，如Google、Youtube、Pixiv等网站。**

# 注册Vultr账号

**[点击这个链接进去注册](https://www.vultr.com/?ref=8444456)，会送给你100刀，但是你必须最少充值10刀才能把这100刀提到你的账户上，相当于把账号激活了。（如果不充值就能白用100刀的话商家得赔死O(∩_∩)O哈哈~）。**

- **必须要先充值10刀，才能用那100刀**。可以选择PayPal，支付宝，微信支付等方式，这个就不细说了。
- **就我个人而言，我选择的是每月10刀的服务器，所以第一次注册后充值10刀加上送的100刀，可以以70RMB的价格用10个月，非常划算。[注意点我进去注册](https://www.vultr.com/?ref=8444456)才能有送的100刀哦~**
注册完成之后，会给你发一封确认邮件，你要去你的邮箱里去确认一下~

![](/images/0-1.png "图0-1 充值之前点击Billing的状态，$100以积分显示")

![](/images/0-2.png "图0-2 充值之后点击Billing的状态，以余额显示（因为我已经用了几天所以不是$110了）")

# 选择、购买服务器
   **如果是小白（跟我一样）用来Google~ Youtube~等服务的话，按照下面的选择就好。（基本是默认的）**

   ![](/images/1.png "图1. 选择服务器，服务器地址")
   
   ![](/images/2.png "图2. 选择服务器的操作系统，选择配置")

   ![](/images/3.png "图3. 开启IPv6，其它都留空，然后点击Deploy Now")
   
   **注意：**
   - 国内用的话，机房地址可以选择日本和洛杉矶
   - 可以勾选IPv6（不用白不用嘛\~不勾选也可以，后面可以在设置里面改的\~）
   - 最后一张图的地方都可以留空，系统会有默认值的
   - 点击 Deploy Now之后要等的时间可能会久一点，大概几分钟，耐心等等\~成功后如下图，变成绿色的running\~
  ![](/images/4.png "图4. 部署成功")

# 在服务器上搭建ssr环境
   
   **你需要在服务器上搭建好ssr环境，才能用服务器科学上网。**
   
   1. 首先需要在终端远程连接服务器，macOS可以直接在终端做，Windows需要下载putty
   ```
   ssh root@你的ip地址
   ```
   
   2. 从本地的终端进入服务器后，分开执行下面三行代码
   ```
   wget --no-check-certificate -O shadowsocks-all.sh https://raw.githubusercontent.com/teddysun/shadowsocks_install/master/shadowsocks-all.sh 
   chmod +x shadowsocks-all.sh
   ./shadowsocks-all.sh 2>&1 | tee shadowsocks-all.log
   ```
   
   3. 在执行上面的第三行代码后，就进入了文本编辑模式，这里我们进行ssr的配置
   
   （1）首先选择服务类型，我选的是2，ShadowsocksR
   ```
   Which Shadowsocks server you'd select:
    1) Shadowsocks-Python
    2) ShadowsocksR
    3) Shadowsocks-Go
    4) Shadowsocks-libev
   Please enter a number (Default Shadowsocks-Python):
   ```
  
  （2）然后是密码，自己设置：
    ```
    Please enter password for ShadowsocksR
    (Default password: teddysun.com):
    ```

  （3）然后是端口号，我设置的8080：
  ```
  Please enter a port for ShadowsocksR [1-65535]
  (Default port: 11400):
  ```

  （4）最后是加密方式，我选择的是2，aes-256-cfb：
  ```
  Please select stream cipher for ShadowsocksR:
  1) none
  2) aes-256-cfb
  3) aes-192-cfb
  4) aes-128-cfb
  5) aes-256-cfb8
  6) aes-192-cfb8
  7) aes-128-cfb8
  8) aes-256-ctr
  9) aes-192-ctr
  10) aes-128-ctr
  11) chacha20-ietf
  12) chacha20
  13) salsa20
  14) xchacha20
  15) xsalsa20
  16) rc4-md5
  Which cipher you'd select(Default: aes-256-cfb):
  ```

  （5）然后是协议，我选择的是3， auth_sha1_v4
  ```
  Please select protocol for ShadowsocksR:
  1) origin
  2) verify_deflate
  3) auth_sha1_v4
  4) auth_sha1_v4_compatible
  5) auth_aes128_md5
  6) auth_aes128_sha1
  7) auth_chain_a
  8) auth_chain_b
  9) auth_chain_c
  10) auth_chain_d
  11) auth_chain_e
  12) auth_chain_f
  Which protocol you'd select(Default: origin):
  ```

  （6）然后是混淆，我选择的1， plain
  ```
  Please select obfs for ShadowsocksR:
  1) plain
  2) http_simple
  3) http_simple_compatible
  4) http_post
  5) http_post_compatible
  6) tls1.2_ticket_auth
  7) tls1.2_ticket_auth_compatible
  8) tls1.2_ticket_fastauth
  9) tls1.2_ticket_fastauth_compatible
  Which obfs you'd select(Default: plain):
  ```

  （7）上一步做完后，出现如下内容：
  ```
  obfs = plain

  Press any key to start...or Press Ctrl+C to cancel
  ```

  （8）再按回车键，等一会儿后，出现如下内容，就是安装成功了！
  ```
  ShadowsocksR (pid 1117) is already running...

  Congratulations, ShadowsocksR server install completed!
  Your Server IP        :  xxxxx 
  Your Server Port      :  8080 
  Your Password         :  xxxxx 
  Your Protocol         :  auth_sha1_v4 
  Your obfs             :  plain 
  Your Encryption Method:  aes-256-cfb 

  Your QR Code: (For ShadowsocksR Windows, Android clients only)
      xxxxx
  Your QR Code has been saved as a PNG file path:
      xxxxx
  Welcome to visit: https://teddysun.com/486.html
  Enjoy it!
```

**附：我安装的时候发生的一个问题**

我买的$10/month的服务器，默认是CentOS 8，在执行上面的部署步骤出错了，Python环境相关的错误，半天没解决，我把它回退到了CentOS 7，就正常了。步骤参考：

![](/images/5.png "图5. 服务器操作系统版本回退到centOS7")

进入自己的服务器，选择 Settings， 然后选择 Change OS，选择CentOS 7版本，然后从第2步执行三行代码重新开始就好了。

# 配置电脑端、手机端的ssr然后愉快地上网~

**1. 下载链接**

  [**SSR for mac**](https://github.com/qinyuhang/ShadowsocksX-NG-R/releases/download/1.4.4-r8/ShadowsocksX-NG-R8.dmg)

  [**SSR for windows**](https://github.com/shadowsocksrr/shadowsocksr-csharp/releases/download/4.9.2/ShadowsocksR-win-4.9.2.zip)

  [**SSR for android**](https://github.com/shadowsocksrr/shadowsocksr-android/releases/tag/3.5.4) **（不要用ss）**

**2. 配置**

- **配置照抄刚刚配置的就行，注意一定不要填错哦，错了没法上网的~~**
- **一般选择PAC自动模式**

![](/images/6.png "图6. SSR客户端配置")

**如果本文对你有用，请关注我，后面会更新利用VPS上北邮人BT的方法（IPv6）。**

**如果有任何问题，欢迎评论讨论交流。**