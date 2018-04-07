# Python 语法笔记

## python时间,日期,时间戳处理
https://blog.csdn.net/u011279649/article/details/70242132

## Python的format函数

自python2.6开始，新增了一种格式化字符串的函数str.format()，此函数可以快速处理各种字符串。  
>语法  
它通过```{}```和```:``` 来代替``` %```。

请看下面的示例，基本上总结了format函数在python的中所有用法：
```python
# 通过位置
In [52]: '{0},{1}'.format('CHJ',24)
Out[52]: 'CHJ,24'

In [53]: '{},{}'.format('CHJ',24)
Out[53]: 'CHJ,24'

In [54]: '{1},{0},{1}'.format('CHJ',24)
Out[54]: '24,CHJ,24'

# 通过关键字参数
In [55]: '{name},{age}'.format(age = 24,name = 'CHJ')
Out[55]: 'CHJ,24'

In [56]: class Person:
    ...:     def __init__(self,name,age):
    ...:         self.name = name
    ...:         self.age = age
    ...:     def __str__(self):
    ...:         return 'This guy is {self.name},is {self.age} old'.format(self = self)
    ...:     

In [57]: print(str(Person('CHJ',24)))
This guy is CHJ,is 24 old

# 通过映射 list
In [58]: a_list = ['CHJ',24,'China']
In [59]: print('My name is {0[0]}, from {0[2]}, age is {0[1]}'.format(a_list))
My name is CHJ, from China, age is 24

# 通过映射 dict
In [60]: b_dict = {'name':'CHJ', 'age':24, 'province':'Jianxi'}
    ...:
In [61]: print('my name is {name}, age is {age}, from {province}'.format(**b_dict))
my name is CHJ, age is 24, from Jianxi

# 填充与对齐
In [62]: print('{:>7}'.format('189'))
    189

In [63]: print('{:0>7}'.format('189'))
0000189

In [64]: print('{:0>1}'.format('189'))
189

In [65]: print('{:0>5}'.format('189'))
00189

In [66]: print('{:0>8}'.format('189'))
00000189

In [67]: print('{:a>8}'.format('189'))
aaaaa189

# 精度与类型f
In [68]: print('{:.2f}'.format(321.34546))
321.35

In [69]: print('{:.2f}'.format(321.34446))
321.34

# 用来做金额的千位分隔符
In [70]: print('{:,}'.format(1234560569085))
1,234,560,569,085

# 其他类型 主要就是进制了，b、d、o、x分别是二进制、十进制、八进制、十六进制。
In [76]: print('{:b}'.format(18))  #二进制 10010
10010

In [77]: print('{:d}'.format(18)) #十进制 18
18

In [78]: print('{:o}'.format(18)) #八进制 22
22

In [79]: print('{:X}'.format(18)) #十六进制12
12

```





## 文件读写与存储

### 7.2. 读写文件
open()返回一个文件对象，最常见的用法带有两个参数：open(filename, mode)。

```>>> f = open('workfile', 'w')```


第一个参数是一个包含文件名的字符串。第二个参数是另一个包含几个字符的字符串，用于描述文件的使用方式。mode为```'r'```时表示只是读取文件；w表示只是写入文件（已经存在的同名文件将被删掉）；```'a'```表示打开文件进行追加；写入到文件中的任何数据将自动添加到**末尾**。```'r+'```表示打开文件进行读取和写入。mode参数是可选的；如果省略，则默认为```'r'```。

通常，文件以文本模式打开，它表示你从文件读取以及向文件写入的字符串是经过特定的编码的。如果没有指定编码，则默认为与平台有关（参见open()）。在mode后面附加```'b'```将以二进制模式打开文件：现在数据以字节对象的形式读取和写入。这个模式应该用于所有不包含文本的文件。

在文本模式中，读取的默认行为是将平台相关的换行（Unix上的```\n```、Windows上的```\r\n```）仅仅转换为```\n```。当在文本模式中写入时，默认的行为是将```\n```转换为平台相关的换行。这种对文件数据的修改对文本文件没有问题，但会损坏JPEG或EXE这样的二进制文件中的数据。在读取和写入这样的文件时要非常小心地使用二进制模式。

### 7.2.1. 文件对象的方法

本节中的示例将假设文件对象```f```已经创建。

要读取文件内容，可以调用```f.read(size)``` ，该方法读取若干数量的数据并以字符串（在文本模式中）或字节对象（在二进制模式中）形式返回它。size是一个可选的数值参数。当 size被省略或者为负数时，将会读取并返回整个文件；如果文件大小是你机器内存的两倍时，这会产生问题。否则，最多读取并返回size字节。如果到了文件末尾，```f.read()``` 会返回一个空字符串(```" "```)。

例如：
__文件名f.txt__  

```
This is the entire file.
vnfdgntgkhngh.
ngirg tng Hahaahahahahaha
vfnggoh
```
```python
In [87]: f = open('f.txt')

In [88]: f.read()
Out[88]: 'This is the entire file.\nvnfdgntgkhngh.\nngirg tng Hahaahahahahaha\nvfnggoh'

In [89]: f.read()
Out[89]: ''

In [90]: f.close()
```

```f.readline()```从文件读取一行数据；字符串结尾会带有一个换行符 (```\n```) ，只有当文件最后一行没有以换行符结尾时才会省略。这样返回值就不会有混淆；如果```f.readline()```返回一个空字符串，那就表示已经达到文件的末尾，而如果返回一个只包含一个换行符的字符串```'\n'```，则表示遇到一个空行。
```python
In [91]: f = open('f.txt')

In [92]: f.readline()
Out[92]: 'This is the entire file.\n'

In [93]: f.readline()
Out[93]: 'vnfdgntgkhngh.\n'

In [94]: f.readline()
Out[94]: 'ngirg tng Hahaahahahahaha\n'

In [95]: f.readline()
Out[95]: 'vfnggoh'

In [96]: f.readline()
Out[96]: ''

In [97]: f.close()
```

对于从文件中读取行，可以在文件对象上循环。这是内存高效，快速，并导致简单的代码：
```python
In [98]: f = open('f.txt')

In [99]: for line in f:
    ...:     print(line, end='')
    ...:     
This is the entire file.
vnfdgntgkhngh.
ngirg tng Hahaahahahahaha
vfnggoh
```


如果你想要读取的文件列表中的所有行的数据，你也可以使用 ```list(f) ```或``` f.readlines()```。
```f.write(string) ```将 字符串 的内容写入到该文件，返回写入的字符数。

f.write(string) 将 字符串 的内容写入到该文件，返回写入的字符数。
```python
In [105]: f = open('f.txt','a')

In [106]: f.write('This is test line.\n')
Out[106]: 19
```
其他类型的对象,在写入之前则需要转换成 字符串 （在文本模式下） 或 字节对象 （以二进制模式）
```python

In [107]: value = ('the answer', 42)

In [108]: s = str(value)  # convert the tuple to string

In [109]: f.write(s)
Out[109]: 18
```
```f.tell()```返回一个整数，代表文件对象在文件中的当前的位置，在二进制模式中该数值表示自文件开头到指针处的字节数，在文本模式中则是不准确的。

使用完一个文件后，调用f.close()可以关闭它并释放其占用的所有系统资源。调用f.close()后，再尝试使用该文件对象将自动失败。

```python

In [110]: f.close()

In [111]: f.read()
Traceback (most recent call last):

  File "<ipython-input-111-571e9fb02258>", line 1, in <module>
    f.read()

ValueError: I/O operation on closed file.
```

处理文件对象是使用with关键字是很好的做法。这具有的优点是，在其套件完成后，文件被正确关闭，即使在路上出现异常。它还比编写一个等同的try-finally语句要短很多：

```python

In [114]: with open('f.txt','r') as f:
     ...:     read_data = f.read()
     ...:     print(read_data)
     ...:     
This is the entire file.
vnfdgntgkhngh.
ngirg tng Hahaahahahahaha
vfnggohThis is test line.
('the answer', 42)

In [115]: f.close()

```
补充实例：```'r+'```表示打开文件进行读取和写入
```python
In [124]: f = open('f.txt','r+')

In [125]: f.write('chj 2018-4-5')
Out[125]: 12

In [126]: f.write('AAABBBCCC')
Out[26]: 9

In [127]: f.close()
```
看看f.txt内容：
```
chj 2018-4-5AAABBBCCCle.
vnfdgntgkhngh.
ngirg tng Hahaahahahahaha
vfnggohThis is test line.
('the answer', 42)
```

### 7.2.2. 使用json存储结构化数据
字符串可以轻松地写入和读取文件。数值就要多费点儿周折，因为read ()方法只会返回字符串，应将其传入int()这样的函数，就可以将'123'这样的字符串转换为对应的数值123。当您想要保存更复杂的数据类型（如嵌套列表和字典）时，手动解析和序列化变得复杂。

Python允许您使用名为JSON（JavaScript Object Notation）的流行数据交换格式，而不是让用户不断地编写和调试代码以将复杂的数据类型保存到文件中。标准模块json可以接受 Python 数据结构，并将它们转换为字符串表示形式；此过程称为序列化。从字符串表示重构数据称为反序列化。在序列化和反序列化之间，表示对象的字符串可能已经存储在文件或数据中，或者通过网络连接发送到一些远程机器。

>注意JSON格式通常由现代应用程序使用以允许数据交换。许多程序员已经熟悉它，这使它成为互操作性的不错选择。
![](https://images2018.cnblogs.com/blog/1245030/201804/1245030-20180405224359017-1082518410.png)

