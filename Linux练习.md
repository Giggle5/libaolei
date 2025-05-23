Linux 安装
![](https://github.com/Giggle5/libaolei/blob/main/images/linux.png)

vi/vim 的使用
基本上 vi/vim 共分为三种模式，命令模式（Command Mode）、输入模式（Insert Mode）和命令行模式（Command-Line Mode）。

命令模式
用户刚刚启动 vi/vim，便进入了命令模式。
此状态下敲击键盘动作会被 Vim 识别为命令，而非输入字符，比如我们此时按下 i，并不会输入一个字符，i 被当作了一个命令。

以下是普通模式常用的几个命令：

i -- 切换到输入模式，在光标当前位置开始输入文本。
x -- 删除当前光标所在处的字符。
: -- 切换到底线命令模式，以在最底一行输入命令。
a -- 进入插入模式，在光标下一个位置开始输入文本。
o：在当前行的下方插入一个新行，并进入插入模式。
O -- 在当前行的上方插入一个新行，并进入插入模式。
dd -- 剪切当前行。
yy -- 复制当前行。
p（小写） -- 粘贴剪贴板内容到光标下方。
P（大写）-- 粘贴剪贴板内容到光标上方。
u -- 撤销上一次操作。
Ctrl + r -- 重做上一次撤销的操作。
:w -- 保存文件。
:q -- 退出 Vim 编辑器。
:q! -- 强制退出Vim 编辑器，不保存修改。
若想要编辑文本，只需要启动 Vim，进入了命令模式，按下 i 切换到输入模式即可。

命令模式只有一些最基本的命令，因此仍要依靠底线命令行模式输入更多命令。

输入模式
在命令模式下按下 i 就进入了输入模式，使用 Esc 键可以返回到普通模式。

在输入模式中，可以使用以下按键：

字符按键以及Shift组合，输入字符
ENTER，回车键，换行
BACK SPACE，退格键，删除光标前一个字符
DEL，删除键，删除光标后一个字符
方向键，在文本中移动光标
HOME/END，移动光标到行首/行尾
Page Up/Page Down，上/下翻页
Insert，切换光标为输入/替换模式，光标将变成竖线/下划线
ESC，退出输入模式，切换到命令模式
底线命令模式
在命令模式下按下 :（英文冒号）就进入了底线命令模式。

底线命令模式可以输入单个或多个字符的命令，可用的命令非常多。

在底线命令模式中，基本的命令有（已经省略了冒号）：

:w：保存文件。
:q：退出 Vim 编辑器。
:wq：保存文件并退出 Vim 编辑器。
:q!：强制退出Vim编辑器，不保存修改。
按 ESC 键可随时退出底线命令模式。

![](https://github.com/Giggle5/libaolei/blob/main/images/vim.png)


yum常用命令
1. 列出所有可更新的软件清单命令：yum check-update
![](https://github.com/Giggle5/libaolei/blob/main/images/yum.png)
2. 更新所有软件命令：yum update
![](https://github.com/Giggle5/libaolei/blob/main/images/yum2.png)
3. 仅安装指定的软件命令：yum install <package_name>
![](https://github.com/Giggle5/libaolei/blob/main/images/yum4.png)
4. 仅更新指定的软件命令：yum update <package_name>

5. 列出所有可安裝的软件清单命令：yum list
![](https://github.com/Giggle5/libaolei/blob/main/images/yum5.png)
6. 删除软件包命令：yum remove <package_name>
![](https://github.com/Giggle5/libaolei/blob/main/images/yum3.png)
7. 查找软件包命令：yum search <keyword>

8. 清除缓存命令:

yum clean packages: 清除缓存目录下的软件包
yum clean headers: 清除缓存目录下的 headers
yum clean oldheaders: 清除缓存目录下旧的 headers
yum clean, yum clean all (= yum clean packages; yum clean oldheaders) :清除缓存目录下的软件包及旧的 headers
![](https://github.com/Giggle5/libaolei/blob/main/images/yum6.png)

