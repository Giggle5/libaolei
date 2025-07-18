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
