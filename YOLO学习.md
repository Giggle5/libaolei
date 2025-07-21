https://zhuanlan.zhihu.com/p/94986199
## YOLO是什么？

YOLO是目标检测模型。

目标检测是计算机视觉中比较简单的任务，用来在一张图篇中找到某些特定的物体，目标检测不仅要求我们识别这些物体的种类，同时要求我们标出这些物体的位置。

显然，类别是离散数据，位置是连续数据。


上面的图片中，分别是计算机视觉的三类任务：分类，目标检测，实例分割。

很显然，整体上这三类任务从易到难，我们要讨论的目标检测位于中间。前面的分类任务是我们做目标检测的基础，至于像素级别的实例分割，太难了别想了。

YOLO在2016年被提出，发表在计算机视觉顶会CVPR(Computer Vision and Pattern Recognition)上

YOLO的全称是you only look once，指只需要浏览一次就可以识别出图中的物体的类别和位置。

因为只需要看一次，YOLO被称为Region-free方法，相比于Region-based方法，YOLO不需要提前找到可能存在目标的Region。

也就是说，一个典型的Region-base方法的流程是这样的：先通过计算机图形学（或者深度学习）的方法，对图片进行分析，找出若干个可能存在物体的区域，将这些区域裁剪下来，放入一个图片分类器中，由分类器分类。

因为YOLO这样的Region-free方法只需要一次扫描，也被称为单阶段（1-stage）模型。Region-based方法方法也被称为两阶段（2-stage）方法。

YOLO之前的世界
YOLO之前的世界，额，其实是R-CNN什么的，也就是我们前面说的Region-based方法，但是感觉还是太高端了。我们从用脚都能想到的目标检测方法开始讲起。

如果我们现在有一个分类器：


现在我们的追求升级了，我们不仅仅想处理这种一张图片中只有一个物体的图片，我们现在想处理有多个物体的图片。

## 我们该什么做呢？

首先有几点我们要实现想到：首先物体的位置是不确定的，你没办法保证物体一定在最中间；其次，物体的大小是不确定的，有的物体比较大，也有的物体比较小，注意，这里不是说大象一定更大，猫咪一定更小，毕竟还有近大远小嘛；然后，我们还没办法保证物体的种类，假设我们有一个可以识别100中物体的分类器，那么起码图片中出现了这100种物体我们都要识别出来。

挺难的，是吧？

最naive的方法是滑窗法，就是用滑动窗口去识别一个个物体。

比如这样：


上图的红色框框就是所谓的滑窗。如果一个物体正好出现在一个滑窗中，那么我们就可以把它检测出来了，这个滑窗的位置也就是我们认为这个物体所在的位置。

等下，如果物体没有正好出现在一个滑窗中呢？

我们管滑窗每次滑动的距离叫做步长，如果我们把步长设置的特别小，如果步长仅仅为一个像素点，那一定可以保证物体可以正好出现在某个窗口中了。

那如果某个物体特别大，或者特别小呢？

例如在上图中，每个窗口和汽车差不多大小，但是如果我们要识别一辆卡车，一个窗口可能就不够大了。

显然，我们可以设计不同大小的窗口，我们可以设计几十中不同大小的窗口，让他们按照最小的步长滑动，把窗口里的所有图片都放入分类器中。

但是这样太太太浪费时间了。

到这里R-CNN同学出现了，他说，你这样用滑窗法可能最后得到了几十万个窗口，而我可以提前扫描一下图片，得到2000个左右的Region（其实就是前面的窗口），这样不就节省了很多时间？


R-CNN同学管这个叫做Region Proposal，并且提出了一个叫做Selective Search的算法。（吐槽一下这个名字起得太大众了）

但是R-CNN被YOLO打脸了，YOLO说，我更快。

## YOLO原理

在这之前，我们再重申一下我们的任务。我们的目的是在一张图片中找出物体，并给出它的类别和位置。目标检测是基于监督学习的，每张图片的监督信息是它所包含的N个物体，每个物体的信息有五个，分别是物体的中心位置(x,y)和它的高(h)和宽(w)，最后是它的类别。

YOLO 的预测是基于整个图片的，并且它会一次性输出所有检测到的目标信息，包括类别和位置。

就好像捕鱼一样，R-CNN是先选好哪里可能出现鱼，而YOLO是直接一个大网下去，把所有的鱼都捞出来。

先假设我们处理的图片是一个正方形。

YOLO的第一步是分割图片，它将图片分割为 
 个grid，每个grid的大小都是相等的，像这样：


如果我们让每个框只能识别出一个物体，且要求这个物体必须在这个框之内，那YOLO就变成了很蠢的滑窗法了。

YOLO的聪明之处在于，它只要求这个物体的中心落在这个框框之中。

这意味着，我们不用设计非常非常大的框，因为我们只需要让物体的中心在这个框中就可以了，而不是必须要让整个物体都在这个框中。

具体怎么实现呢？

我们要让这个 
 个框每个都预测出B个bounding box，这个bounding box有5个量，分别是物体的中心位置(x,y)和它的高(h)和宽(w)，以及这次预测的置信度。

每个框框不仅只预测B个bounding box，它还要负责预测这个框框中的物体是什么类别的，这里的类别用one-hot编码表示。

注意，虽然一个框框有多个bounding boxes，但是只能识别出一个物体，因此每个框框需要预测物体的类别，而bounding box不需要。

也就是说，如果我们有 
 个框框，每个框框的bounding boxes个数为B，分类器可以识别出C种不同的物体，那么所有整个ground truth的长度为：


先看这些bounding box显示出来是什么样的：


在上面的例子中，图片被分成了49个框，每个框预测2个bounding box，因此上面的图中有98个bounding box。

可以看到大致上每个框里确实有两个bounding box。

可以看到这些BOX中有的边框比较粗，有的比较细，这是置信度不同的表现，置信度高的比较粗，置信度低的比较细。

在详细的介绍confidence之前，我们先来说一说关于bounding box的细节。

bounding box可以锁定物体的位置，这要求它输出四个关于位置的值，分别是x,y,h和w。我们在处理输入的图片的时候想让图片的大小任意，这一点对于卷积神经网络来说不算太难，但是，如果输出的位置坐标是一个任意的正实数，模型很可能在大小不同的物体上泛化能力有很大的差异。

这时候当然有一个常见的套路，就是对数据进行归一化，让连续数据的值位于0和1之间。

对于x和y而言，这相对比较容易，毕竟x和y是物体的中心位置，既然物体的中心位置在这个grid之中，那么只要让真实的x除以grid的宽度，让真实的y除以grid的高度就可以了。

但是h和w就不能这么做了，因为一个物体很可能远大于grid的大小，预测物体的高和宽很可能大于bounding box的高和宽，这样w除以bounding box的宽度，h除以bounding box的高度依旧不在0和1之间。

解决方法是让w除以整张图片的宽度，h除以整张图片的高度。

下面的例子是一个448*448的图片，有3*3的grid，展示了计算x,y,w,h的真实值（ground truth）的过程：


接下来，我们好好说道说道这个confidence。

confidence的计算公式是：


这个IOU的全称是intersection over union，也就是交并比，它反应了两个框框的相似度。


 的意思是预测的bounding box和真实的物体位置的交并比。

 是一个grid有物体的概率，在有物体的时候ground truth为1，没有物体的时候ground truth为0.

这个
 非常有意思，因为它的groun truth不是确定的，这导致虽然 
 的ground truth是确定的，但是bounding box的confidence的ground truth是不确定的。

一个不确定的ground truth有什么用呢？

想象这样的问题：老师问小明1+1等于几。小明说等于2，老师又问你有多大的把握你的回答是对的，小明说有80%

这里的80%就是confidence。

confidence主要有两个作用，在后面我会一一介绍。

现在，我们根据上面大雁的图片计算一下样本的groun truth：


首先，这里有9个grid，每个grid有两个bounding box，每个bounding box有5个预测值，假设分类器可以识别出3中物体，那么ground truth的总长度为 

我们假定大雁的类别的one-hot为100，另外两个是火鸡和特朗普，分别是010和001.

我们规定每个grid的ground truth的顺序是confidence, x, y, w, h, c1, c2, c3

那么第一个（左上角）grid的ground truth应该是：0, ?, ?, ?, ?, ?, ?, ?

实际上除了最中间的grid以外，其他的grid的ground truth都是这样的。

这里的"?"的意思是，随便是多少都行，我不在乎。在下面我们会看到，我们不会对这些值计算损失函数。

中间的ground truth应该是：

iou, 0.48, 0.28, 0.50, 0.32, 1, 0, 0

iou要根据x, y, w, h的预测值现场计算。

这样看似可以让每个grid找到负责的物体，并把它识别出来了。但是还存在一个不得不考虑的问题，如果物体很大，而框框又很小，一个物体被多个框框识别了怎么办？

这里，我们要用到一个叫做非极大值抑制Non-maximal suppression(NMS)的技术。

这个NMS还是基于交并比实现的。


例如在上面狗狗的图里，B1,B2,B3,B4这四个框框可能都说狗狗在我的框里，但是最后的输出应该只有一个框，那怎么把其他框删除呢？

这里就用到了我们之前讲的confidence了，confidence预测有多大的把握这个物体在我的框里，我们在同样是检测狗狗的框里，也就是B1,B2,B3,B4中，选择confidence最大的，把其余的都删掉。

也就是只保留B1.

但是这里还有一个引人深思的问题，为什么confidence的定义是 
 ，直接用 
 不行吗，直接用 
 的话就可以把ground truth确定下来，训练的时候就方便多了。

这里有一个非常非常鸡贼的技巧！

理论上只用 
 也可以选出应该负责识别物体的grid，但是可能会不太精确。这里我们训练的目标是预测 
 ，我们的想法是让本来不应该预测物体的grid的confidence尽可能的小，既然 
 的效果不太理想，那我就让 
 尽可能小。

为什么真正的最中间的grid的confidence往往会比较大呢？

因为我们的bounding boxes是用中点坐标+宽高表示的，每个grid预测的bounding box都要求其中心在这个grid内，那么如果不是最中间的grid，其他的grid的IOU自然而言就会比较低了，因此相应的confidence就降下来了。

现在，我们知道了哪个是应该保留的bounding boxes了，但是还有一个问题，我们是怎么判断出这几个bounding boxes识别的是同一个物体的呢？

这里用到NMS的技巧，我们首先判断这几个grid的类别是不是相同的，假设上面的B1，B2，B3和B4识别的都是狗狗，那么进入下一步，我们保留B1，然后判断B2，B3和B4要不要删除。

我们把B1成为极大bounding box，计算极大bounding box和其他几个bounding box的IOU，如果超过一个阈值，例如0.5，就认为这两个bounding box实际上预测的是同一个物体，就把其中confidence比较小的删除。

最后，我们结合极大bounding box和grid识别的种类，判断图片中有什么物体，它们分别是什么，它们分别在哪。


我们刚才说confidence有两个功能，一个是用来极大值抑制，另一个就是在最后输出结果的时候，将某个bounding box的confidencd和这个bounding box所属的grid的类别概率相乘，然后输出。




# 论文精读
哔哩哔哩博主 同济子豪兄
yolov1

## 测试阶段 
输出7*7*30（5+5+20）张量 5是指位置坐标x,y,h,w,置信度，20个类别的条件概率。

## 训练阶段
人工标注框，使最终
每一个gridcell有两个bbox  中心点落在gridcell里面，只能预测一个物体。置信度最大的  

## 预测后处理  
非极大值抑制
过滤


# 源代码试运行及其理解

## data.py
我们逐行解释代码的功能和含义。这个代码定义了一个自定义数据集类`MyDataset`，用于加载图像和标签，并支持训练集和验证集的划分。
### 1. 导入必要的库
```python
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import torch
from PIL import Image
import torchvision.transforms as transforms
```
- `Dataset`和`DataLoader`：PyTorch中用于构建数据集和数据加载的工具。
- `numpy`：用于数值计算，这里用于读取标签文件（csv格式）。
- `os`：用于操作系统相关的功能，如文件路径操作。
- `random`：用于生成随机数，这里用于设置随机种子以保证可重复性。
- `torch`：PyTorch深度学习框架。
- `PIL.Image`：Python图像处理库，用于读取图像文件。
- `torchvision.transforms`：提供常用的图像预处理和增强方法。
### 2. 定义`MyDataset`类，继承自`Dataset`
```python
class MyDataset(Dataset):
```
### 3. 初始化方法`__init__`
```python
    def __init__(self, dataset_dir, seed=None, mode="train", train_val_ratio=0.9, trans=None):
```
- `dataset_dir`：数据集所在的目录路径。
- `seed`：随机种子，用于划分训练集和验证集时保证可重复性。
- `mode`：数据集模式，可以是`"train"`（训练）、`"val"`（验证）或`"test"`（测试）。注意，在代码中，如果`mode`为`"val"`，会先将其改为`"train"`，这是因为训练和验证集都是从同一个总训练集中划分的，它们使用同一个`train.txt`和`train.csv`。
- `train_val_ratio`：训练集占整个训练集（包括验证集）的比例，默认0.9，即训练集:验证集=9:1。
- `trans`：数据预处理和增强的变换组合。
#### 初始化中的代码：
```python
        if seed is None:
            seed = random.randint(0, 65536)
        random.seed(seed)
```
如果未提供随机种子，则生成一个随机种子，并设置随机种子以保证可重复性。
```python
        self.dataset_dir = dataset_dir
        self.mode = mode
        if mode=="val":
            mode = "train"
```
保存`dataset_dir`和`mode`。注意，当`mode`为`"val"`时，将`mode`变量改为`"train"`，这是因为验证集的数据文件实际上和训练集是同一个（即`train.txt`和`train.csv`），后续通过索引划分。
```python
        img_list_txt = os.path.join(dataset_dir, mode+".txt")  # 储存图片位置的列表
        label_csv = os.path.join(dataset_dir, mode+".csv")  # 储存标签的数组文件
```
构建图像列表文件和标签文件的路径。例如，如果`mode`是`"train"`（注意此时`val`已经被改成了`train`），那么图像列表文件为`train.txt`，标签文件为`train.csv`。
```python
        self.img_list = []
        self.label = np.loadtxt(label_csv)  # 读取标签数组文件
```
初始化图像路径列表`self.img_list`，并使用`numpy.loadtxt`读取标签文件（csv格式）到`self.label`。
```python
        with open(img_list_txt, 'r') as f:
            for line in f.readlines():
                self.img_list.append(line.strip())
```
打开图像列表文件（每行是一个图像路径），读取每一行并去除首尾空格，然后添加到`self.img_list`。
```python
        self.num_all_data = len(self.img_list)
        all_ids = list(range(self.num_all_data))
        num_train = int(train_val_ratio*self.num_all_data)
```
计算总数据量`self.num_all_data`，生成所有数据的索引列表`all_ids`，并计算训练集的数量（根据比例`train_val_ratio`）。
```python
        if self.mode == "train":
            self.use_ids = all_ids[:num_train]
        elif self.mode == "val":
            self.use_ids = all_ids[num_train:]
        else:
            self.use_ids = all_ids
```
根据`self.mode`（注意，这里用的是`self.mode`，它可能是`"train"`、`"val"`或`"test"`）来划分使用的索引：
- 训练模式：使用前`num_train`个索引。
- 验证模式：使用剩余索引（从`num_train`开始到最后）。
- 测试模式（或其它模式）：使用全部索引。
```python
        self.trans = trans
```
保存传入的数据增强变换。
### 4. `__len__`方法
```python
    def __len__(self):
        return len(self.use_ids)
```
返回数据集的大小，即使用的样本数量（根据`use_ids`的长度）。
### 5. `__getitem__`方法
```python
    def __getitem__(self, item):
        id = self.use_ids[item]
        label = torch.tensor(self.label[id, :])
        img_path = self.img_list[id]
        img = Image.open(img_path)
```
- 根据索引`item`获取实际数据索引`id`（从`use_ids`中取）。
- 根据`id`从`self.label`中取出对应的标签，并转换为`torch.tensor`。
- 根据`id`从`self.img_list`中取出图像路径，然后用`PIL.Image.open`打开图像。
```python
        if self.trans is None:
            trans = transforms.Compose([
                # transforms.Resize((112,112)),
                transforms.ToTensor(),
            ])
        else:
            trans = self.trans
```
如果未提供数据变换，则使用默认变换（仅转换为张量）。否则，使用提供的变换。
```python
        img = trans(img)  # 图像预处理&数据增广
        # transforms.ToPILImage()(img).show()  # for debug
        # print(label)
        return img, label
```
对图像进行变换，然后返回图像张量和标签。
### 6. 调试代码
```python
if __name__ == '__main__':
    dataset_dir = r"D:\VOC2012\voc2012_forYolov1"
    dataset = MyDataset(dataset_dir)
    dataloader = DataLoader(dataset, 1)
    for i in enumerate(dataloader):
        input("press enter to continue")
```
在直接运行该脚本时，会执行以下操作：
- 指定数据集路径`dataset_dir`。
- 实例化`MyDataset`（默认参数：`mode="train"`，`train_val_ratio=0.9`）。
- 创建`DataLoader`，批大小为1。
- 遍历数据加载器，每取一个批次就暂停，等待用户按回车继续。这可以用于调试，查看每个样本是否正确。
### 注意点：
1. 当`mode`为`"val"`时，在初始化中会将`mode`变量改为`"train"`，这是因为验证集和训练集使用同一个源文件（`train.txt`和`train.csv`），然后通过索引划分。但注意，类内部保存的`self.mode`仍然是原来的模式（`"val"`），用于确定使用哪部分索引。
2. 在测试模式下（`mode="test"`），会使用整个数据集（不划分）。
3. 数据预处理：如果没有提供`trans`，则只进行`ToTensor`转换（将图像转换为张量，并归一化到[0,1]）。在实际使用中，可能需要根据任务添加其他预处理（如归一化、缩放等）。
### 可能的问题：
- 在验证模式下，代码将`mode`变量改成了`"train"`，这样读取的是`train.txt`和`train.csv`。如果验证集有单独的文件（如`val.txt`和`val.csv`），则这种做法不合适。但根据代码上下文，作者意图是训练和验证集来自同一个大训练集，所以这样处理。
- 标签读取为`numpy`数组，然后按行索引。注意标签文件必须是每一行对应一个样本的标签，且每行有多个数值（因为用了`self.label[id, :]`）。
- 在调试部分，批大小为1，每次只取一个样本。`enumerate`会返回批次索引和批次数据（每个批次包含一个图像和一个标签）。
### 改进建议：
- 在读取图像时，可以检查图像是否成功打开，避免因文件损坏导致错误。
- 对于大型数据集，使用`np.loadtxt`可能效率较低，可以考虑使用`pandas`或分块读取。
- 如果标签文件很大，可以考虑只读取需要用到的部分（但这里先全部读取，然后按索引取，对于大内存机器可行）。
- 在`__getitem__`中，如果图像是灰度图，可能需要转换为RGB（根据任务需求）。
- 默认变换中没有归一化，可能需要根据模型要求添加归一化步骤（如`transforms.Normalize`）。
下面是对代码的逐行详细解释：

python
from torch.utils.data import Dataset, DataLoader
导入PyTorch中用于创建自定义数据集和数据加载器的核心类。Dataset是基类，需要继承并实现特定方法；DataLoader用于批量加载数据。

python
import numpy as np
导入NumPy库，用于科学计算，这里主要用于处理标签数据。

python
import os
导入操作系统接口库，用于处理文件和目录路径。

python
import random
导入随机数生成库，用于数据集的随机划分。

python
import torch
导入PyTorch深度学习框架。

python
from PIL import Image
导入Python图像处理库(PIL)的Image模块，用于加载和处理图像文件。

python
import torchvision.transforms as transforms
导入PyTorch的torchvision.transforms模块，提供常用的图像转换和数据增强方法。

python
class MyDataset(Dataset):
定义一个名为MyDataset的自定义数据集类，继承自PyTorch的Dataset基类。

python
    def __init__(self, dataset_dir, seed=None, mode="train", train_val_ratio=0.9, trans=None):
类的初始化方法，接收以下参数：

dataset_dir: 数据集根目录路径

seed: 随机种子（确保训练/验证集划分可复现）

mode: 数据集模式（"train"/"val"/"test"）

train_val_ratio: 训练集占整个数据集的比例

trans: 数据预处理/增强函数

python
        if seed is None:
            seed = random.randint(0, 65536)
        random.seed(seed)
如果没有提供随机种子，则生成一个随机种子并设置，确保结果可复现。

python
        self.dataset_dir = dataset_dir
        self.mode = mode
保存数据集目录和模式到实例变量。

python
        if mode=="val":
            mode = "train"
特殊处理：当模式为"val"（验证集）时，将模式临时改为"train"，因为验证集和训练集使用相同的源文件。

python
        img_list_txt = os.path.join(dataset_dir, mode+".txt")
        label_csv = os.path.join(dataset_dir, mode+".csv")
构建图像路径列表文件和标签文件的完整路径（如：train.txt, train.csv）。

python
        self.img_list = []
        self.label = np.loadtxt(label_csv)
初始化图像路径列表，并从CSV文件加载标签数据（NumPy数组格式）。

python
        with open(img_list_txt, 'r') as f:
            for line in f.readlines():
                self.img_list.append(line.strip())
读取图像路径文件，将每行（一个图像路径）添加到img_list中，并去除首尾空白字符。

python
        self.num_all_data = len(self.img_list)
        all_ids = list(range(self.num_all_data))
        num_train = int(train_val_ratio*self.num_all_data)
计算数据集总样本数，生成所有样本的索引列表，计算训练集应包含的样本数。

python
        if self.mode == "train":
            self.use_ids = all_ids[:num_train]
        elif self.mode == "val":
            self.use_ids = all_ids[num_train:]
        else:
            self.use_ids = all_ids
根据当前模式选择使用的样本索引：

训练模式：使用前num_train个样本

验证模式：使用剩余样本

其他模式（如测试）：使用全部样本

python
        self.trans = trans
保存传入的数据预处理/增强函数。

python
    def __len__(self):
        """获取数据集数量"""
        return len(self.use_ids)
实现Dataset要求的__len__方法，返回数据集大小（当前模式下的样本数）。

python
    def __getitem__(self, item):
        """
        获取单个样本
        1. 按索引获取样本
        2. 加载图像并预处理
        3. 获取对应标签
        4. 转换为张量并返回
        """
        id = self.use_ids[item]
        label = torch.tensor(self.label[id, :])
根据给定索引item获取实际样本ID，并将对应的标签转换为PyTorch张量。

python
        img_path = self.img_list[id]
        img = Image.open(img_path)
获取图像路径，并用PIL加载图像。

python
        if self.trans is None:
            trans = transforms.Compose([
                # transforms.Resize((112,112)),  # 可选的图像缩放
                transforms.ToTensor(),           # 转换为张量并归一化到[0,1]
            ])
        else:
            trans = self.trans
如果没有提供预处理函数，则创建默认转换（仅转换为张量）。否则使用提供的转换函数。

python
        img = trans(img)
对图像应用预处理/增强转换。

python
        # transforms.ToPILImage()(img).show()  # 调试用：显示图像
        # print(label)                        # 调试用：打印标签
        return img, label
返回处理后的图像张量和标签张量（注释部分用于调试）。

python
if __name__ == '__main__':
    # 调试用，依次取出数据看看是否正确
    dataset_dir = r"D:\VOC2012\voc2012_forYolov1"
主程序入口：当直接运行此脚本时执行的代码。设置数据集路径（使用原始字符串避免转义）。

python
    dataset = MyDataset(dataset_dir)
创建数据集实例（使用默认参数：mode="train"）。

python
    dataloader = DataLoader(dataset, 1)
创建数据加载器，批量大小为1（即每次加载一个样本）。

python
    for i in enumerate(dataloader):
        input("press enter to continue")
遍历数据加载器：

enumerate返回(索引, 数据)元组

每次迭代暂停，等待用户按回车继续

用于逐样本检查数据加载是否正确

关键点总结：
灵活的数据集划分：通过mode参数自动处理训练/验证/测试集的划分

随机种子控制：确保每次运行划分结果一致

图像预处理流水线：支持自定义转换，默认只做张量转换

按需加载：仅在__getitem__中加载图像，节省内存

调试友好：提供可视化检查的注释代码

此数据集类特别适用于：

图像分类任务

目标检测任务（如YOLO）

需要训练/验证划分的场景

需要数据增强的实验

## model.py
python
import torch
import torch.nn as nn
import torchvision.models as tvmodel
导入PyTorch核心库、神经网络模块和torchvision模型库。

python
from prepare_data import GL_CLASSES, GL_NUMBBOX, GL_NUMGRID
from util import calculate_iou
从自定义模块导入全局变量：

GL_CLASSES: 目标检测的类别数量

GL_NUMBBOX: 每个网格预测的边界框数量（YOLO v1为2）

GL_NUMGRID: 网格划分数量（YOLO v1为7×7）

calculate_iou: 计算两个边界框交并比的函数

python
class MyNet(nn.Module):
    """
    @ YOLOv1 with ResNet34 backbone
    """
定义YOLOv1模型类，使用ResNet34作为主干网络。

python
    def __init__(self):
        super(MyNet, self).__init__()
        resnet = tvmodel.resnet34(pretrained=None)  # 调用torchvision里的resnet34预训练模型
初始化网络：

继承nn.Module基类

加载ResNet34模型（不使用预训练权重）

python
        resnet_out_channel = resnet.fc.in_features  # 记录resnet全连接层之前的网络输出通道数
获取ResNet34最后一层卷积的输出通道数（512）

python
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # 去除resnet的最后两层
移除ResNet34的最后两层（全局池化层和全连接层），保留特征提取部分

python
        # 以下是YOLOv1的最后四个卷积层
        self.Conv_layers = nn.Sequential(
            nn.Conv2d(resnet_out_channel, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),  # 为了加快训练，这里增加了BN层，原论文里YOLOv1是没有的
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
        )
定义YOLOv1特有的4个卷积层：

1024通道的3×3卷积（保持尺寸）

1024通道的3×3卷积（步长为2，下采样）

1024通道的3×3卷积（保持尺寸）

1024通道的3×3卷积（保持尺寸）
每层后接批归一化(BatchNorm)和LeakyReLU激活

python
        # 以下是YOLOv1的最后2个全连接层
        self.Conn_layers = nn.Sequential(
            nn.Linear(GL_NUMGRID * GL_NUMGRID * 1024, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4096, GL_NUMGRID * GL_NUMGRID * (5*GL_NUMBBOX+len(GL_CLASSES))),
            nn.Sigmoid()  # 增加sigmoid函数是为了将输出全部映射到(0,1)之间
        )
定义YOLOv1的全连接部分：

展平特征图后连接4096个神经元

输出层：7×7网格×(每个网格5个坐标×2个边界框+类别数)

使用Sigmoid激活将输出限制在(0,1)范围内

python
    def forward(self, inputs):
        x = self.resnet(inputs)
        x = self.Conv_layers(x) #使用resnet的输出14x14x512
        x = x.view(x.size()[0], -1)  #对7x7x1024进行展平处理
        x = self.Conn_layers(x) #全连接线性分类器得到1x1470
        self.pred = x.reshape(-1, (5 * GL_NUMBBOX + len(GL_CLASSES)), GL_NUMGRID, GL_NUMGRID)
        return self.pred
前向传播：

通过ResNet主干网络

通过4个卷积层

展平特征图

通过全连接层

重塑输出为(batch_size, 30, 7, 7)格式（30=5×2+20类）

python
    def calculate_loss(self, labels):
        self.pred = self.pred.double()
        labels = labels.double()
计算损失函数，确保使用双精度浮点数

python
        num_gridx, num_gridy = GL_NUMGRID, GL_NUMGRID  # 划分网格数量
        noobj_confi_loss = 0.  # 不含目标的网格损失
        coor_loss = 0.  # 含有目标的bbox的坐标损失
        obj_confi_loss = 0.  # 含有目标的bbox的置信度损失
        class_loss = 0.  # 含有目标的网格的类别损失
        n_batch = labels.size()[0]  # batchsize的大小
初始化损失分量和批次大小

python
        # 遍历每个样本、每个网格
        for i in range(n_batch):  # batchsize循环
            for n in range(num_gridx):  # x方向网格循环
                for m in range(num_gridy):  # y方向网格循环
三层循环：遍历批次中的每个样本，以及7×7网格中的每个位置

python
                    if labels[i, 4, m, n] == 1:  # 如果包含物体
                        # 将预测的bbox转换为(x1,y1,x2,y2)格式
                        bbox1_pred_xyxy = (...)
                        bbox2_pred_xyxy = (...)
                        # 将真实bbox转换为(x1,y1,x2,y2)格式
                        bbox_gt_xyxy = (...)
如果当前网格包含物体：

将预测的两个边界框从(中心x,中心y,宽,高)转换为(x1,y1,x2,y2)格式

将真实边界框同样转换

python
                        # 计算两个预测框与真实框的IoU
                        iou1 = calculate_iou(bbox1_pred_xyxy, bbox_gt_xyxy)
                        iou2 = calculate_iou(bbox2_pred_xyxy, bbox_gt_xyxy)
计算两个预测边界框与真实边界框的交并比(IoU)

python
                        # 选择iou大的bbox作为负责物体
                        if iou1 >= iou2:
                            # 计算坐标损失
                            coor_loss += ... 
                            # 负责框的置信度损失
                            obj_confi_loss += ...
                            # 非负责框的置信度损失
                            noobj_confi_loss += ...
                        else:
                            # 类似处理第二个边界框
                            ...
选择IoU较大的边界框作为"负责"预测物体的框：

计算负责框的坐标损失（中心点误差+宽高误差）

计算负责框的置信度损失（目标是IoU值）

计算非负责框的置信度损失（目标是另一个IoU值）

python
                        # 类别损失
                        class_loss += ...
无论哪个框负责，都计算类别预测损失

python
                    else:  # 如果不包含物体
                        # 两个边界框的置信度损失
                        noobj_confi_loss += ...
如果网格不包含物体，两个边界框的置信度目标都是0

python
        # 总损失
        loss = coor_loss + obj_confi_loss + noobj_confi_loss + class_loss
        return loss / n_batch
汇总所有损失分量，并返回批次平均损失

python
    def calculate_metric(self, preds, labels):
        """计算评估指标（此处实现似乎不完整）"""
        preds = preds.double()
        labels = labels[:, :(self.n_points*2)]
        l2_distance = torch.mean(torch.sum((preds-labels)**2, dim=1))
        return l2_distance
计算评估指标的方法（当前实现似乎与YOLO任务不匹配）

python
if __name__ == '__main__':
    # 调试代码
    x = torch.zeros(5,3,448,448)  # 创建5张448x448的黑色图像
    net = MyNet()  # 初始化网络
    a = net(x)  # 前向传播
    labels = torch.zeros(5, 30, 7, 7)  # 创建全零标签
    loss = net.calculate_loss(labels)  # 计算损失
    print(loss)  # 打印损失值
    print(a.shape)  # 打印输出形状
测试代码：

创建模拟输入数据（5张448×448图像）

初始化网络

进行前向传播

计算损失

打印损失值和输出形状

YOLOv1损失函数关键点：
坐标损失：只针对"负责"预测物体的边界框

中心坐标使用平方误差

宽高使用平方根后的平方误差（减轻大物体和小物体之间的不平衡）

加权系数5，强调位置精度的重要性

置信度损失：

有物体网格的负责框：目标是IoU值

有物体网格的非负责框：目标是另一个IoU值（或0）

无物体网格：目标是0

无物体损失加权系数0.5，减少负样本的贡献

类别损失：有物体网格的类别预测误差（平方误差）

设计特点：

每个网格预测2个边界框但只选择IoU最大的负责预测

平衡不同损失分量的权重

使用双精度计算确保数值稳定性

这个实现结合了ResNet34的特征提取能力和YOLOv1的检测头，同时添加了批归一化来改善训练稳定性。




