# transformer

## 一、简介

### 1.1 what's transformer

简单地说，transformer就是seq2seq with "self-attention"。

关于简单的seq2seq with attention ，我在前篇文章已经有过介绍，如果还不知道seq2seq with attention 的话，可以先去看一下那篇文章。

###  1.2 why use transformer

#### 1.2.1 传统RNN	

为了更好地理解transformer,我们首先需要理解，为什么要用transformer，它与之前的方法相比有什么优势。

​	传统情况下，我们处理NLP任务一般是使用RNN网络

![RNN](https://gitee.com/in_the_wind_ghx/markdownImageUpload/raw/master/img/RNN.jpg)

<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">图1.传统RNN</center> 

​	传统RNN除了在前面讲解seq2seq时提到的N-M问题之外，还有一个问题就是**难以并行化(parallelized)**，如上图，我们需要先算出$h_1$然后才能再算$h_2$，然后才能算$h_3$，后面的任务总是需要等待前面的任务完成后才能开始。这就导致了**效率不够高**的问题。

#### 1.2.2 CNN replace RNN

​	针对这个问题，有人提出了用CNN去替代RNN的想法(CNN是比较容易并行化的)。

​	![CNNreplaceRNN](https://gitee.com/in_the_wind_ghx/markdownImageUpload/raw/master/img/CNNreplaceRNN.png)

<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">图2.CNN替代RNN实现并行化</center> 

​	但是，这样仍然存在一定的问题，每个CNN层的感受野是一定的，只能捕捉感受野范围内的信息，如果想要捕捉到某个文本的信息，就需要不断地堆叠CNN层，让前面的多个CNN捕捉到的信息作为输入，喂给下一个CNN层，这样经过一些堆叠后，高层的CNN也能捕捉到整个文本的信息。

​	用CNN去替代RNN,想要捕捉到全局的信息就必须要去堆叠，就显得有些繁琐，因此，就有人又提出了transformer。

#### 1.2.3 transformer

![image-20200729235445819](https://gitee.com/in_the_wind_ghx/markdownImageUpload/raw/master/img/image-20200729235445819.png)

<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">图3.transformer实现并行化</center> 

transformer想要做的，就是用self-attention 去替代RNN，就像在《什么是transformer》一节中我们提到的，transformer就是seq2seq with "self-attention"。

## self-attention

### self-attention计算方式

首先，X是input层，作为最初的输入。

经过一层Embedding层，输出为A(各$a^i$组成的向量)

我们会有3个向量，q,k,v

这3个向量都是分别由三个参数矩阵$W^q$,$W^k$,$W^v$与 $a^i$相乘得到的。



![image-20200730000318481](https://gitee.com/in_the_wind_ghx/markdownImageUpload/raw/master/img/image-20200730000318481.png)

<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">q、k、V向量</center> 

然后我们将每个q与每个k做attention，原始论文中，使用的是Scaled Dot-Product Attention，也就是

$α_{i,j}=q^i·k^i/\sqrt{d}$ ,其中q与k的乘积是做点积运算，这样，我们就可以得到所有的$α_{i,j}$

![image-20200730000419700](https://gitee.com/in_the_wind_ghx/markdownImageUpload/raw/master/img/image-20200730000419700.png)

<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">得到α向量</center> 

再将刚刚得到的各个$α_{i,j}$拼成一个向量作为输入，经过一个**softmax层**,就得到了各个$\hat{α}_{i,j}$

![image-20200730001410580](https://gitee.com/in_the_wind_ghx/markdownImageUpload/raw/master/img/image-20200730001410580.png)

<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">得到hat α向量</center> 

然后，我们通过 $b_j=\sum_{i}{\hat{a}_{j,i}·v^i}$ 来计算出各个 $b_j$ 的值



![image-20200730001818840](https://gitee.com/in_the_wind_ghx/markdownImageUpload/raw/master/img/image-20200730001818840.png)

<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">计算b^1</center> 

![image-20200730002517278](https://gitee.com/in_the_wind_ghx/markdownImageUpload/raw/master/img/image-20200730002517278.png)

<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">计算b^2</center> 

以此类推...

这样，我们就做到了并行化地计算

![image-20200730002715324](https://gitee.com/in_the_wind_ghx/markdownImageUpload/raw/master/img/image-20200730002715324.png)

<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">也正是最开始介绍transformer出现的图</center> 

### self-attention如何并行化

对于q,k,v来说，我们可以直接转化为矩阵运算

​													$$Q=W^q·I$$

​													$$K=W^k·I$$

​													$$V=W^v·I$$

其中，I指的是Input,实际上就是初始的文本生成字典后的向量经过一层Embedding的结果。

![image-20200730003547708](https://gitee.com/in_the_wind_ghx/markdownImageUpload/raw/master/img/image-20200730003547708.png)

对于计算α

![image-20200730004107702](https://gitee.com/in_the_wind_ghx/markdownImageUpload/raw/master/img/image-20200730004107702.png)

![image-20200730004202997](https://gitee.com/in_the_wind_ghx/markdownImageUpload/raw/master/img/image-20200730004202997.png)

只需令$$A=KQ$$,即可

而后，A经过softmax层即可得到$\hat{A}$



对于b

![image-20200730004306130](https://gitee.com/in_the_wind_ghx/markdownImageUpload/raw/master/img/image-20200730004306130.png)

也只需令$$O=VA$$即可(O指的是Output)



我们可以看到，整个self-attention，全部都可以用矩阵运算，这样也就可以并行化，得到更高的运行速度。



![image-20200730004519932](https://gitee.com/in_the_wind_ghx/markdownImageUpload/raw/master/img/image-20200730004519932.png)



### Multi-head Self-attention

Multi-head Self-attention与普通的self-attention只有一点不同，就是每个$a_i$后面的q,v,j向量均不止1个。下图是以2 heads为例，那么每个q,k,v都有2个。

但是其计算的时候，始终是各个q,k,v中同一位置的做attention。

$q^{i,1}只会与各个a_i的第1个k向量做attention$ 

![image-20200730004654904](https://gitee.com/in_the_wind_ghx/markdownImageUpload/raw/master/img/image-20200730004654904.png)

![image-20200730004933556](https://gitee.com/in_the_wind_ghx/markdownImageUpload/raw/master/img/image-20200730004933556.png)

![image-20200730005221105](https://gitee.com/in_the_wind_ghx/markdownImageUpload/raw/master/img/image-20200730005221105.png)

## Positional Encodeing

还有一点需要注意的是，回顾上面self-attention的计算过程，我们可以发现各个$x_i$的位置是没有考虑到的，但是显然一个字在文本中的位置是会影响文本的意思的，因此又提出了Positional Encoding的概念，希望给每个字一个位置向量p，给予模型各个字符位置的表示。

在原始论文中，作者是这样做的: 给每个位置一个独立的位置向量(人工设置而非作为超参数学习)，p是一个独热编码的向量，$W^p$是我们人为设置的，像$W^p*p_i赋值给e_i,e_i+a_i作为真正的a_i$

![](https://gitee.com/in_the_wind_ghx/markdownImageUpload/raw/master/img/image-20200730103300129.png)

## transformer

![image-20200730102414665](https://gitee.com/in_the_wind_ghx/markdownImageUpload/raw/master/img/image-20200730102414665.png)

![image-20200730102450401](https://gitee.com/in_the_wind_ghx/markdownImageUpload/raw/master/img/image-20200730102450401.png)





## Reference

这里理论主要参考了李宏毅 transformer讲解;原始的PPT以及PDF形式也一并上传，可以自行下载。

