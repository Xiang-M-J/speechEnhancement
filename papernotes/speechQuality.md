## 一些想法

CNN + 注意力（受限的注意力，Restricted Self-Attention） + 时间编码

**受限的注意力不太行**

将注意力放在最后全局平均池化中（好像不太行）

生成对抗网络（一个随机加减高斯噪声，一个根据加减情况与上一次的检测结果对应，这种措施可能用于微调）

在使用Quality-Net 训练语音增强模型时，使用联合训练的方式

在训练语音增强模型时，语音增强模型生成增强后的语音，Quality-Net 作为鉴别器判断语音质量（GAN网络）

将 MSE 损失换成交叉熵损失，具体思路是将 1.0-5.0 看成若干个类别，如每隔 0.2 进行一个分类

（如果不修改原来的网络，单纯使用avgS的输出即4×1维的特征来计算EMD损失的效果不好）

考虑 topk 的损失（扩大分数区间，如将分数区间扩大到0.4或者0.5，将top k （或者全部）的概率对应的点的概率乘上对应的点与真实值计算损失，压低非top k 的点对应的概率）

训练语音质量网络和语音增强模型的数据应该需要事先分开

在训练classifier时并不是准确率越高，预测效果越好

lstmClassifier：step 0.2，EDMLoss + FrameLoss，smooth=True

cnn训练时，使用两个1×1卷积+1个3×3卷积比2个3×3卷积的效果好，卷积块中的池化层选择，128, 64

SENet的效果似乎更好

Focal Loss: 
$$
{\left( {{p_t}} \right)^\gamma }\log \left( {{p_t} + 1} \right)
$$
其中，$p_t$ 为计算得到的 EMD 损失，$p_t$​ 越大，表示分类越不自信，所以为其分配更大的损失。对avgloss使用focal、对Frame使用EDMLoss，效果更好。

CNN 应该也能计算帧损失

训练Hubert 需要较小的学习率，如训练LSTM使用1e-3，训练Hubert则需要使用1e-4这样的学习率

在训练完语音增强模型后，finetune时将未加噪的语音和增强后的语音分别通过QualityNet，减小这两者的差距



**在训练QualityNet时进行如下规定**

1. 对于分数，计算归一化输入输出，不在模型中添加sigmoid 函数，在计算损失时加上sigmoid函数
2. 对于分类器，不在模型中添加 Softmax 函数，如果需要在损失计算时添加
3. 平均分数的输出为 (N, 1), 帧分数的输出为 （N L）
4. 平均分类的输出为（N, C），帧分类的输出为（）



对于 CNN1D 如果需要计算注意力，长度不能太短

对于CNNMAttn 最后直接使用 nn.Linear(128, 1) 比用 nn.Linear(128, 50) nn.Linear(50, 1) 效果好

将mos指标用于语音增强模型的训练，训练时根据mos分数考虑为每条语音分配权重

实验中发现直流分量的损失总是最大的

![image-20240601100950420](https://cdn.jsdelivr.net/gh/Xiang-M-J/MyPic@img/img/image-20240601100950420.png)



## 多维度语音质量评价

MOS_COL：频谱染色

MOS_DISC：连续性

MOS_NOISE：降噪

MOS_REVERB：混响

MOS_LOUD：响度

MOS_SIG：整体语音质量

MOS_OVRL：整体质量

上述指标越大越好



## 实验发现

> [!IMPORTANT]
>
> 训练 QualityNet 时特别需要注意损失函数，损失函数必须要正确，不能使用 weight_decay，否则会导致训练难以进行

> [!CAUTION]
>
> 如果使用 CNN 进行训练，Frame Loss 可能需要设置为 0，但是效果比较差

> [!TIP]
>
> 在训练很大的数据集时，可能需要将学习率降低的步长减小

> [!IMPORTANT]
>
> 在训练CNN模型时，使用 Adam 优化器

> [!WARNING]
>
> 在编写函数时最好不要直接修改参数，否则可能会影响后面的操作

> [!NOTE]
>
> 如果训练时发现损失突然变为 Nan，检查计算损失时是否包含了sqrt操作（sqrt求导时根号下的值会放在分母中），需要保证sqrt中的值为正值（加上一个很小的值如 1e-6）

> [!NOTE]
>
> 使用CNN2d会大量增加计算量，同时对模型的增益效果有限

> [!CAUTION]
>
> 使用语音质量模型计算损失时，**千万不能使用 with torch.no_grad()** 这会导致pytorch不再计算梯度，导致语音增强模型无法更新参数，同时需要注意对于语音质量模型来说，如果希望其不更新参数，那么唯一要做的是冻结其参数，其它的和正常的模型类似，该在 model.train() 时就用 model.train()，该在 model.eval() 时就用 model.eval()。





## 基本知识

### 主观方法测量语音质量

语音增强理论与实践 Ch10

#### 相对偏好法

**等优先测试**：设置不同的强度级别的测试信号，并被不同级别的加性噪声破坏，通过一个带限滤波器（模拟语音传输系统模拟特性）组成的系统。实验对象听了成对的测试信号，选择他们喜欢的一个。测试结果在语音强度水平与噪声水平的二维空间上以“等偏好轮廓”的形式报告。等优等高线上的任意两点指定对两个测试信号产生相等优先级的参数值对。

还有一种方法不需要确定等优轮廓，而是基于使用五种失真语音作为参考信号(该方法由IEEE主观测量小组委员会推荐)。

有许多不同信号被用于生成参考信号，如调制噪声参考单元（MNRU）
$$
r(n) = x(n)\left[ {1 + {{10}^{ - Q/20}}d(n)} \right]
$$
d(n) 为随机噪声，x(n) 为输入的语音信号，Q 为信噪比。



#### 绝对类别评定法

**平均意见得分（MOS）**：分为两个阶段，训练和评估，在训练阶段，听者听到一组参考信号，这些信号代表了高、低和中等判断类别。这个阶段，也被称为“锚定阶段”，是非常重要的，因为它需要平衡所有听众的主观质量评分范围。也就是说，训练阶段原则上应该使所有听众的“善良”尺度相等，以尽可能确保一个听众认为“好”的东西被其他听众认为是“好”的。在报告MOS分数时，需要使用和描述一组标准的参考信号。在评估阶段，受试者聆听测试信号，并根据5个质量类别(1-5)对信号的质量进行评分（分数越高越好）。MOS 的参考信号可以通过将 MNRU 中的 Q 值设置为 5-35 之间获得。

**诊断可接受度测量（DAM）**：DAM测试在三个不同的尺度上评估语音质量，分别是参数化、超度量和等距度量。这三个尺度总共产生了16个测量语音质量，涵盖了信号和背景的几个属性。超尺度和等距尺度代表了传统的类别判断方法，其中语音是相对于“可理解性”、"愉悦性"和"可接受性"进行评级的。



### 客观方法测量语音质量

语音增强理论与实践 Ch11

首先将语音分成 10-30 ms的帧，然后计算原始和处理后的信号之间的失真度量，通过平均每个语音帧的失真度量来计算一个单一的、全局的语音失真度量。失真度量可以在时域（信噪比）或者频域（LPC系数）计算。

#### 分段信噪比测量

可以在时域或者频域进行测量，时域测量是最简单的客观方法之一，需要将原始信号和处理过后的信号在时间上对齐，定义为
$$
SN{R_{seg}} = {{10} \over M}\sum\limits_{m = 0}^{M - 1} {{{\log }_{10}}{{\sum\nolimits_{n = Nm}^{Nm + N - 1} {{x^2}(n)} } \over {{{\sum\nolimits_{n = Nm}^{Nm + N - 1} {\left( {x(n) - \hat x(n)} \right)} }^2}}}}
$$
M 是帧的数量，N为帧长（15-20ms）。信噪比估计的一个潜在问题是，语音信号中静默间隔期间的信号能量将非常小，导致很大的负值，这将使整体测量产生偏差。解决这一问题的一种方法是，通过将短时间能量测量值与阈值进行比较，或将SNRseg值调到一个较小的值，将静默帧从式11.1中的总和中排除。可以将信噪比限制在[-10, 35 dB]的范围内，从而避免了语音静默检测。还有一种方法是在log中的数加 1，从而避免负值的存在。

频域的 SNR 计算如下
$$
fwSN{R_{seg}} = {{10} \over M}\sum\limits_{m = 0}^{M - 1} {{{\sum\nolimits_{j = 1}^K {{W_j}{{\log }_{10}}\left[ {{X^2}(j,m)/{{\left( {X(j,m) - \hat X(j,m)} \right)}^2}} \right]} } \over {\sum\nolimits_{j = 1}^K {{W_j}} }}}
$$
$W_j$ 是第 j 个频带的权重，K 为频带数

频域 SNR 相比于时域 SNR 的一个优点是，可以为频谱中不同频带加权。



#### 基于LPC的谱距离测量

这种测量方式认为短时语音可以表示为 p 阶的全极点模型
$$
x(n) = \sum\limits_{i = 1}^p {{a_x}(i)x(n - i)}  + {G_x}u(n)
$$
$G_x$ 是滤波器增益，$u(n)$​ 是单位方差的白噪声激励

最常用的两种评价语音增强算法的方法为对数似然比 LLR 和 Itakura–Saito IS 测量。



#### 感知动机测量

**加权谱斜率距离测量**：心理声学的研究表明，人对于共振峰频率不同的元音对的差异感受更明显。这种测量首先需要找到干净频谱和增强过的加噪频谱中每个频带的谱斜率（作一阶差分），然后对谱斜率的差异进行加权，首先根据波段是否靠近谱峰或谷，其次根据该峰是否为光谱中的最大峰。频带 k 的权重记为 W(k)，计算公式如下：
$$
W(k) = {{{K_{max}}} \over {\left[ {{K_{max}} + {C_{max}} - {C_x}(k)} \right]}}{{{K_{locmax}}} \over {\left[ {{K_{locmax}} + {C_{locmax}} - {C_x}(k)} \right]}}
$$
$C_{max}$ 是所有频带中最大的幅度，$C_{locmax}$ 是最接近频带 k 的峰值

$K_{max}$ 和 $K_{locmax}$ 都是常数，可以使用回归分析来调整以最大化主观听力测试与客观方法之间的关系，$K_{max}$ 可以取 20，$K_{locmax}$ 可以取 1。

加权谱斜率测量的公式为
$$
{d_{WSM}}({C_x},{{\bar C}_x}) = \sum\limits_{i = 1}^p {W(k){{\left( {{S_x}(k) - {{\bar S}_{\hat x}}(k)} \right)}^2}}
$$
**Bark 失真测量**：该方法基于下面的事实：人耳的频率分辨率不是均匀的，即声信号的频率分析不是基于线性频率标度；耳朵的灵敏度与频率有关；响度与信号强度呈非线性关系

人的听觉被建模为一系列声信号的变换。原始信号和经过增强的信号都经过这一系列的变换，从而得到所谓的响度谱。Bark 谱失真(Bark spectral distortion, BSD)测量方法使用这些谱之间的距离作为主观质量度量。

**语音质量测量的感知评价（PESQ）**：上述客观措施只适用于评估有限范围的失真，这些失真不包括语音通过电信网络时常见的失真（如丢包，信号延迟和编解码器失真等）。PESQ 被提出，同时作为 ITU-T 推荐的语音质量评价标准。PESQ 度量的结构如下图所示

![image-20240503150922823](D:\TyporaImages\image-20240503150922823.png)

原始和加噪信号首先被均衡到一个标准的收听电平，并通过一个响应类似于标准电话听筒的滤波器进行滤波。将信号对齐以校正时间延迟，然后通过类似于 BSD 的听觉变换进行处理，得到响度谱。响度谱之间的差异，称为干扰，并按时间和频率平均，以产生主观MOS评分的预测。

*预处理*

被测系统的增益不是先验已知的，可能根据电话连接的类型而有很大的变化。因此，有必要将原始信号和加噪信号的电平均衡到标准收听电平。增益是基于带通滤波(350-3250 Hz)语音的均方根值计算的。这些增益应用于原始信号和增强信号，产生这些信号的缩放版本。在增益均衡之后，信号由一个响应类似于电话听筒的滤波器滤波。中间参考系统(IRS)接收所使用的电话听筒的特性。计算了原始语音信号和降级语音信号的IRS滤波版本，并将其用于时间对齐块和感知模型中。

*时间对齐*

时间对齐块提供时延值给感知模型，使得对应的原始信号和失真信号可以比较。

1. 粗略延迟估计：对原始信号和失真信号作互相关，分辨率为 4 ms。信号波形是基于由log(max(Ek /ET, 1))确定的归一化帧能量值来计算的，其中Ek是4毫秒长帧k中的能量，ET是由语音活动检测器确定的阈值
2. 话语分割和对齐：使用估计的延迟将原始信号分成若干个子部分，这些子部分被称为话语。进一步细化以确定话语与最近样本的准确对齐，这分两个阶段完成：a 基于包络的话语延迟估计。这种处理只提供原始信号和失真信号的语音之间的粗略对齐。b 在基于包络的对齐之后，原始信号和处理信号的帧(持续时间为64 ms)加汉宁窗再作互相关。最大相关指数给出了每一帧的延迟估计，而相关性的最大值(提高到0.125次方)被用作每一帧中对齐的置信度的度量。这些延迟估计随后在基于直方图的方案中使用，以确定到最近样本的延迟。

精细时间对齐过程的输出是每个话语的延迟值和延迟置信度。通过拆分和重新调整每个话语的时间间隔来测试语音中的延迟变化。在每个话语的几个点上重复分割过程，并确定产生最大概率的分裂。从PESQ结构图可以看出，在应用感知模型之后，具有非常大的干扰平衡(大于阈值)的部分被识别并使用互相关重新对齐。PESQ 的时间对齐部分随着匹配的开始和结束样本产生每个时间间隔的延迟。使得在感知模型中识别每帧的延迟。



*感知模型*

听觉转换（Auditory transform）块基于一系列类似于BSD测量中使用的转换，将信号映射为感知响度的表示。计算响度谱的步骤

1. Barl 谱估计
2. 频率均衡
3. 增益均衡
4. 响度谱计算



*干扰计算和时间频率平均*

干扰为增强和原始幅度谱之差，n 代表时间（帧）
$$
r_n(b) = S_n(b)-\bar S_n(b)
$$
PESQ 并不将正响度谱差异和负响度谱差异看成一样。正差异表示噪声增加，而负差异表示丢失或者严重衰减，与加噪相比，由于屏蔽效应，丢失的部分不容易察觉。因此，正和负的差异会有不同的权重（非对称）。



#### 宽带 PESQ

宽带（50-7000Hz） PESQ 与 PESQ 相比，有两个小改动。首先，去掉了最初用于电话耳机响应建模的IRS滤波器。改为使用平坦响应超过100 Hz 的 IIR 滤波器。其次，使用逻辑类型函数映射 PESQ 原始输出值，以更好地拟合主观平均意见得分。



#### 综合测量

综合多种客观测量方法，每种方法可以捕获失真信号不同的特征。可以使用回归分析最大化相关来计算客观方法的最优结合。这里的客观测量方法可以是分段信噪比、PESQ、LPC谱距离等。



#### 非侵入式客观质量测量

之前所提到的都是侵入式测量，因为需要使用干净语音。有些非侵入式方法是基于将输出信号与从适当码本导出的人工参考信号进行比较。其他方法使用声道模型来识别失真。后一种方法首先从信号中提取一组声道形状参数，然后评估这些参数是否存在违反声音产生规律，即这些参数是否可能由人类语音产生系统生成。当声道参数产生不合理时，识别出扭曲。ITU-T P.563 采用了声道法的一种变体作为非侵入性语音质量评价标准。



#### 客观测量的指标数值

什么使某种客观度量优于其他客观度量。一些客观测量对于特定类型的失真是“优化的”，而对于另一种类型的失真可能没有意义。在广泛的失真中评估客观测量的有效性的任务是巨大的。一个建议遵循的过程是创建一个以各种方式失真的语音的大型数据库，并评估数据库中每个文件和每种失真类型的客观测量。与此同时，失真的数据库需要由人类听众使用前面描述的主观听力测试（例如，MOS测试）之一进行评估。需要使用统计分析来评估主观得分与客观测量值之间的相关性。为了使客观测量有效和有用，它需要与主观听力测试很好地相关。

主观听力分数和客观测量之间的相关可以使用 Pearson 相关系数衡量，相关系数 ρ 可以用来根据客观测量预测主观分数：
$$
{P_k} = {\mu _P} + \rho {{{\sigma _P}} \over {{\sigma _O}}}\left( {{O_k} - {\mu _O}} \right)
$$
$O_k$ 为客观测量，$P_k$ 为主观听力分数。

通过使用客观测量来预测主观听力分数而获得的误差标准偏差的估计为
$$
{\sigma _e} = {\sigma _P}\sqrt {1 - {\rho ^2}}
$$
$\sigma_e$ 是估计的标准差。另一种可选的指标值是每个条件的平均客观测量和主观评分之间的均方根误差(RMSE)
$$
RMSE = \sqrt {{{\sum\nolimits_{i = 1}^M {{{\left( {{{\bar S}_i} - {{\bar O}_i}} \right)}^2}} } \over M}}
$$
其中下标 i 表示第 i 个条件，$\bar S$ 表示均值。



### POLQA

POLQA 的流程图如下所示，下面称 Degraded signal 为失真信号，

<img src="D:\TyporaImages\image-20240504132955029.png" alt="image-20240504132955029" style="zoom:67%;" />

#### 时间对齐

时间对齐的基本概念有

1. 将失真信号分成等距的帧，并计算每帧的延时。延迟表示样本中可以找到最佳匹配参考信号部分的偏移量；
2. 只要可能，在参考信号中搜索失真信号部分的匹配对应，反之亦然；
3. 逐步细化每帧延迟以避免长搜索范围（长搜索范围需要大量计算，并且与时间缩放输入信号相结合是至关重要的）

时间对齐包括主要的块滤波、预对齐、粗对齐、细对齐和分段组合。失真的输入信号被分割成等长的宏帧，宏帧的长度取决于输入采样率。通过宏帧计算参考信号相对于失真信号的延迟（算法通常在参考信号中搜索失真信号的部分）。

预对齐确定信号的活动语音部分，计算每个宏帧的初始延迟估计和每个宏帧延迟所需的估计搜索范围（检测到的初始延迟的理论最小和最大延迟变化）。

粗对齐执行每帧延迟的迭代细化，使用多维搜索和类似viterbi的回溯算法来过滤检测到的延迟。为了保持所需的相关长度和较小的搜索范围，粗对齐的分辨率逐级提高。

细对齐最终确定输入信号中每一帧的精确延迟，以最大可能的分辨率。这一步的搜索范围由粗对齐最后一次迭代的精度决定。

在最后一步中，所有具有几乎相同延迟的语音段被组合成所谓的“段信息”。

一般计算延迟的方法是计算两个信号之间的互相关。先计算两个信号之间的互相关，将峰值放在一个直方图中，然后将两个信号都移动一小段距离，重复计算互相关，并将峰值放在直方图内。当直方图中包含了足够的值，确定其峰值，这个峰值对应的位置便是两个信号的延迟偏移。

时间对齐的第一步是**滤波**（这里滤波的结果仅用于时间对齐），对于全带宽信号，带通滤波器的频率为 320-3400Hz，对于窄带信号，带通滤波器的频率为 290-3300Hz。

**预对齐**首先识别失真信号中的重分析点。重分析点是指信号从语音暂停过渡到主动语音的位置。重分析点标记了活动语音段的开始，而重分析段描述了从重分析点开始的整个活动语音段。对于每个这样的重分析点，将计算重分析段信息。该段信息存储了该段的开始和结束的位置，以及该段的延迟的初始值，表示所发现的延迟的可靠性及其准确性，即期望找到准确延迟的上限和下界。

当信号有着固定的或者分段的固定偏移，存在一个快速对齐方法

**粗对齐**逐步细化每帧延迟。将每个信号细分为小的子部分(“特征帧”)并为每个子部分计算一个特征值(“特征”)，得到的向量称为特征向量。特征帧同样是等距的，长度在迭代中不断减少（与宏帧长度无关，通常比宏帧的长度短）。长度的减少提高了每次迭代估计延迟的准确性，但同时减少了可靠使用的搜索范围。计算多个特征向量，对于每个宏帧，使用最合适的特征来确定当前帧的最终延迟值。

**细对齐**直接在参考信号和失真信号上以最大可能的分辨率进行操作，并确定每帧的精确延迟，以样本表示。由于前面的对齐步骤，所需的搜索范围被限制到一个较小的范围。因此，可以使用非常短的相关性来预测准确的延迟值，且不会影响预测的准确性。

#### 以固定的延迟连接段

在此步骤中，将具有相同延迟的所有部分组合在一起，这意味着为整个部分存储一组信息（延迟、可靠性、启动、停止、语音活动）。

在第二步中，n+1段与n段结合：

+ 如果n+1段包含主动语音，并且两个部分的延迟相差小于0.3 ms

+ 如果n+1段包含语音暂停，并且两个部分的延迟相差小于15毫秒。得到的切片信息被传递给心理声学模型



#### 采样率比检测

需要采样率比检测来补偿参考信号和失真信号在播放速度上的感知无关差异。这种差异可能有各种各样的原因，它们可能是有意的（例如，由于抖动缓冲器自适应造成的时间缩放），也可能不是有意的（例如，由于部分模拟设备中的A/D或D/A转换器不同步）。在任何情况下产生的效果都是相同的，并且可以描述为两个信号的采样率在非常少的百分比范围内的差异。重要的是要注意，这不是关于标称采样率，而不是相对于另一个信号的有效采样率。



#### 重采样

如果标称采样率与检测到的采样率之差大于0.5%，则对高采样率的信号进行下采样，重新开始整个处理。这种情况最多发生一次，以避免在不能以可靠的方式确定采样率比的信号的情况下过度循环。



#### 电平，频率响应和时间校准预处理

ITU-T P.863算法旨在考虑重放电平对感知质量的影响，因此需要一个校准因子，将以dBov表示的数字信号表示电平映射到以dB(a)表示的重放电平。当数字表示在-26 dBov电平时，以73 dB(a)播放的信号(双耳接收的信号相同)选择该校准因子为2.8。对于数字信号电平和重放电平的相互组合，校准因子可由下式确定
$$
C = 2.8*{10^{\left( { - 26 - dBov} \right)/20}}*{10^{\left( {73 - db(A)} \right)/20}}
$$
ITU-T P.863算法支持窄带模式和全带模式两种工作模式。在窄带模式下，参考信号和失真信号都用IRS接收滤波器进行预滤波，IRS接收滤波器代表一种听的情况，在这种情况下，受试者判断在单音模式下通过IRS接收手持设备或在单音模式下通过IRS接收耳机的语音信号的质量。在全频带模式下，参考信号和失真信号都没有被过滤，这代表了一种听的情况，在这种情况下，受试者在浑浊模式下通过漫射场均衡耳机判断语音信号的质量。



#### 感知模型

当采样率为 36-72kHz 时，傅里叶变换的窗大小为2048。



## 论文

### Quality-Net

大多数客观语音质量评估都基于干净语音和处理后的语音之间的比对，但是在现实场景中难以得到用于参考的干净语音。人可以无需参考语音便可以对语音质量进行评估。这篇论文提出了一种端到端、非侵入式（不需要干净语音参考）的语音质量评估方法 Quality-Net，该方法基于双向LSTM，Quality-Net 中话语水平的评估基于帧层次的评估，采用帧约束和遗忘门偏差的合理初始化，从话语级质量标签中学习有意义的帧级质量评估。

使用 Quality-Net 来预测 PESQ。模型需要将 $u\in R^{T(u)}$ 映射到 $Q\in R^1$ 上，$u$ 为输入语音，Q 为预测的质量分数，T(u) 可以是 $u$ 在时域波形上的采样点或者帧数。

使用幅度谱作为输入，Quality-Net 输出语音质量得分，除此之外，为了避免 Quality-Net 变成一个黑箱，尽管训练数据的质量标签是在话语水平，Quality-Net 被设计成自动学习合理的帧水平质量，质量分数 Q 是逐帧分数 $q_t$ 的全局平均。

我们只有话语水平的语音质量标签，当噪声非平稳时，帧之间的语音失真程度并不相同，不能直接将话语水平的质量标签赋给每个独立的帧。一般来说，人们认为的高质量的语音应该没有任何的失真，即所有帧都会分配一个很高的分数，基于这样的事实，作者提出了 Quality-Net 的目标函数中的条件逐帧约束，目标函数如下
$$
O={1 \over S}\sum\limits_{s = 1}^S {\left[ {{{\left( {{{\hat Q}_s} - {Q_s}} \right)}^2} + \alpha \left( {{{\hat Q}_s}} \right)\sum\limits_{t = 1}^{T({u_s})} {{{\left( {{{\hat Q}_s} - {q_{s,t}}} \right)}^2}} } \right]}
$$
$\alpha(\hat Q_s)$ 是加权因子。
$$
\alpha \left( {{{\hat Q}_s}} \right) = {10^{\left( {{{\hat Q}_s} - {{\hat Q}_{MAX}}} \right)}}
$$
S 为训练话语的总数，$\hat Q_s$ 和 $Q_s$ 分别是第 s 条话语的真实和预测质量分数。$q_{s,t}$ 是第 s 条话语的第 t 帧的预测帧质量。$\hat Q_{MAX}$ 是指标中最大的分数（如MOS中 $\hat Q_{MAX}=5$ ，而 PESQ 中 $\hat Q_{MAX}=4.5$）

由于 BLSTM 对长程信息的捕捉，在一条话语中，干净帧的质量分数会受到噪声污染帧的影响而偏低，这会导致逐帧质量分数不再逐帧，因此需要控制BLSTM的遗忘门，将遗忘门的偏置初始化为一个较小的数值，使其更加关注当前帧。



### Learning With Learned Loss Function: Speech Enhancement With Quality-Net to Improve Perceptual Evaluation of Speech Quality



这篇论文中 Quality-Net 是一种侵入式方法，同时将原本 Quality-Net 中的 BLSTM 换成 CNN，有利于梯度下降。将 Quality-Net 级联在语音增强的模型后面。为了近似PESQ，首先计算训练数据（干净语音和加噪语音）的 PESQ 分数，之后使用 MSE 损失训练  Quality-Net 最小化估计和真实的分数之间的差异。因为框架在变长语音上执行，所以需要在 CNN 后使用全局平均操作来处理变长数据。

训练好 Quality-Net 后，将其级联在语音增强模型后面，训练语音增强模型时，固定 Quality-Net 的参数，最大化估计的质量分数，可以将 Quality-Net 看成是一个损失函数。为了防止语音增强模型生成的增强谱中存在额外的 artifacts（副产品），输出为 ratio mask（值为0-1）。因此，最优增强模型 $G^*$ 可以通过解下面的优化问题求解
$$
{G^*} = \arg \mathop {\min }\limits_G \sum\limits_{u = 1}^U {{{\left( {1 - Q\left( {{N_u} \otimes G\left( {{N_u}} \right),{C_u}} \right)} \right)}^2}}
$$
U 为训练话语的总数，$N_u$ 和 $C_u$ 为带噪语音和干净语音的幅度谱，Q 表示 Quality-Net ，$ \otimes$ 表示按元素乘。（注意 ${G\left( {{N_u}} \right)}$ 生成的是一个 mask）

训练时，语音增强模型先使用 MSE 损失预训练，再使用 Quality-Net 损失微调。语音增强模型为 BLSTM，Quality-Net 是四个二维卷积层组成的 CNN，再加上全连接层。



### A Deep Learning Loss Function Based on the Perceptual Evaluation of the Speech Quality

为语音质量评估提出了一个感知指标，可以作为损失函数，该损失函数在 MSE 损失的基础上加上两个扰动项，分别表示听觉掩蔽和阈值效应。

受到 PESQ 算法的启发，一个对称和非对称的扰动，都是逐帧计算。对称扰动 $D_t^{(s)}$ 考虑增强后的频谱与干净频谱之间的绝对差。非对称扰动 $D_t^{(a)}$ 通过对称扰动计算，但是因为掩蔽效应，人们对于负差异（省略或衰减的频谱成分）与正差异（加性噪声）的感知不同，所以为正和负响度差异赋予不同的权重。最终的损失函数为
$$
J = {1 \over T}\sum\limits_t {\left( {MS{E_t} + \alpha D_t^{(s)} + \beta D_t^{(a)}} \right)}
$$
T 为训练批次中帧的数量。

对称和非对称扰动都是在响度谱中计算，响度谱是通过将stft变换后的结果转到 Bark 谱中，再进行 Zwicker 律转换获得响度。对称扰动是通过计算响度谱之间的绝对差异
$$
d_t^{(s)} = \max \left( {\left| {{{\hat s}_t} - {s_t}} \right| - {m_t},0} \right),\quad {m_t} = 0.25 \cdot \min \left( {{{\hat s}_t},{s_t}} \right)
$$
这里的 $s_t$ 是对应帧的响度谱。非对称扰动的计算为 $d_t^{(a)} = d_t^{(s)} \odot {r_t}$，$\odot$ 表示按位相乘，$r_t$ 是一个非对称比率，由 Bark 谱计算得到
$$
{R_{t,q}} = {\left( {{{{{\hat B}_{t,q}} + \varepsilon } \over {{B_{t,q}} + \varepsilon }}} \right)^\lambda }
$$
这里的 $\varepsilon$ 和 $\lambda$ 分别设置为 50 和 1.2。

最后的 $D_t^{(s)}$ 和 $D_t^{(a)}$ 的计算分别为
$$
\eqalign{
  & D_t^{(s)} = \left\| w \right\|_1^{1/2} \cdot {\left\| {w \odot d_t^{(s)}} \right\|_2}  \cr 
  & D_t^{(a)} = {\left\| {w \odot d_t^{(a)}} \right\|_1} = {w^T} \cdot d_t^{(a)} \cr}
$$
为了复用 PESQ 中已有的感知常数和值，需要进行预处理，因此在 Bark 转换之前，两个信号的电平应等于标准电平，预处理如下
$$
{{\bar x}_t} = {x_t} \cdot {{{P_c}} \over {{1 \over T}\sum\limits_t {\left( {{g^T} \cdot {x_t}} \right)} }}
$$




### Non-Intrusive Binaural Speech Intelligibility Prediction From Discrete Latent Representations



从双耳信号进行非侵入式在许多应用中非常有用，大多数现有的基于信号的方法应用于单信道信号，而考虑双耳性质的经常需要使用干净语音。本文使用矢量量化（VQ）和对比预测编码（CPC）实现非侵入式预测语音可理解性。VQ-CPC 的特征提取不依赖于听觉系统的任何模型，而是通过训练来最大化输入信号和输出特征之间的互信息。将计算得到的 VQ-CPC 特征输入到由神经网络建模的预测函数中。考虑了两个预测函数，在具有各向同性噪声的模拟双耳信号上训练特征提取器和预测函数。

语音可理解度 (SI) 预测旨在预测普通听者在信号中理解语音的能力-可能被噪音，混响或处理伪影破坏。SI 定义为可以被评估者正确辨别的单词或词素，一般来说测试 SI 是一件非常耗时的事情，所以基于信号的预测方法十分必要，这些方法可以被分为侵入式和非侵入式。

如果只使用 CPC，将会限制在单通道音频输入。

VQ-CPC 用于预测 M 个通道的信号 $x_m(n)$，长度为 N，采样频率为 $f_s$。m 和 n 分别表示信道下标和采样索引
$$
x_m(n)=s(n)*h_m(n)+v_m(n)
$$
这些 VQ-CPC 特征的计算由三个主要部分组成：非线性编码器、VQ 码本和自回归聚合器。

首先非线性编码器 f 将输入 x 映射为中间隐层表示，然后应用 VQ 码本 q 将隐层表示映射成嵌入向量，最后通过一个自回归聚合器 g 得到输出 c。

f，q 和 g 的训练是通过端到端最大化互信息
$$
I(x;c) = \sum\limits_{x,c} {p(x,c)\log \left( {{{p(x|c)} \over {p(x)}}} \right)} 
$$
损失函数为
$$
L = \beta  \cdot {L_{vq}} + {1 \over k}\sum\limits_{i = 1}^k {{L_i}}
$$

### Reﬁnement and validation of the binaural short time objective intelligibility measure for spatially diverse conditions

提出了短期客观可理解性（STOI）的双耳版本（DBSTOI）。



### RemixIT: Continual self-training of speech enhancement models via bootstrapped remixing

提出了自监督语音增强方法 RemixIT，RemixIT 基于一种连续的自训练方案，在该方案中，使用域外数据预训练的教师模型推断出域内混合的估计伪目标信号。然后，通过对估计的干净信号和噪声信号进行排列并将它们混合在一起，生成一组新的自举（bootstrapped）混合数据和相应的伪目标，用于训练学生网络。反之亦然，教师定期使用最新学生模型的更新参数来改进其估计。RemixIT 可以与任何分离模型相结合，并适用于任何半监督和无监督域自适应任务。

RemixIT 训练语音增强模型来分离干净语音和观测噪声。对于初始的教师模型，可以使用在域外数据集上训练的任意语音增强模型，输出为语音分量和一个或多个估计噪声。教师模型的输出中的预测语音和噪声会被随机组合生成新的混合数据来训练学生模型。每隔一段时间，教师模型会利用学生模型的参数来更新自身参数。

每次优化时，RemixIT 尝试最小化学生模型估计和教师模型估计结果在信号水平的损失。



### HASA-NET: A NON-INTRUSIVE HEARING-AID SPEECH ASSESSMENT NETWORK

已有的一些网络：

预测 PESQ：Quality-Net

预测 MOS：MOSNet、MBNet

语音可理解性：STOI-Net



本文提出的 HASA-Net 可以同时估计给定语音的质量和可理解性，即预测 HASQI（语音质量）和 HASPI（语音可理解性）分数（多任务学习）



在 BLSTM 中加入了注意力



LCC：[scipy.stats.pearsonr — SciPy v0.14.0 Reference Guide](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pearsonr.html)

SRCC：[scipy.stats.spearmanr — SciPy v0.14.0 Reference Guide](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.spearmanr.html)



### NIMA: Neural Image Assessment

图像评分模型，使用基本的卷积网络（VGG16、Inception-V2 和 MobileNet）提取特征，通过全连接层，再经过 softmax 得到分数落在的区间的概率，最终分数可以直接取区间乘以区间概率，可以使用交叉熵损失，但是更好的选择为EMD损失，EMD 定义为将一个分布搬移到另一个分布的最小代价，即
$$
EMD\left( {p,\hat p} \right) = {\left( {{1 \over N}\sum\limits_{k = 1}^N {{{\left| {CD{F_p}(k) - CD{F_{\hat p}}(k)} \right|}^r}} } \right)^{1/r}}
$$
其中 $CDF_p(k)$ 为累积分布函数 $\sum_{i=1}^k p_{s_i}$，注意  $\sum_{i=1}^N p_{s_i} = \sum_{i=1}^N \hat p_{s_i} = 1$，r = 2，N 为分类数。

代码为

```python
class EDMLoss(nn.Module):
    def __init__(self):
        super(EDMLoss, self).__init__()

    def forward(self, p_target: torch.Tensor, p_estimate: torch.Tensor):
        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]
        cdf_target = torch.cumsum(p_target, dim=1)
        # cdf for values [1, 2, ..., 10]
        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2)))
        return samplewise_emd.mean()
```



### A TIME-RESTRICTED SELF-ATTENTION LAYER FOR ASR


在 TDNN 中加入注意力，将 TDNN + LSTM 架构中的 LSTM 替换为注意力层，这里使用的注意力为 TIME-RESTRICTED 自注意力


考虑单头的情况，将 $x_t$ 转为查询向量 $q_t$、键向量 $k_t$ 和值向量 $v_t$，输出 $y_t$ 为

$$
{y_t} = \sum\limits_{\tau  = t - L}^{t + R} {{c_t}(\tau ){v_t}}
$$

其中 $c_t(\tau) = \exp(q_t\cdot k_{\tau}) / Z_t$，$Z_t$ 用来保证 $\sum_{\tau} c_t(\tau) = 1$。


为了了解 key 和 query 的相对关系，需要加上位置编码，给 x 加上一个向量，该向量除了位置 $\tau + L - t$ 为 1，其余全部为0。

单头的表达能力可能不足，所以需要扩展到多头。
