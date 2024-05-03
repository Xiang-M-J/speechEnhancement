### Quality-Net

大多数客观语音质量评估都基于干净语音和处理后的语音之间的比对，但是在现实场景中难以得到用于参考的干净语音。人可以无需参考语音便可以对语音质量进行评估。这篇论文提出了一种端到端、非侵入式（不需要干净语音参考）的语音质量评估方法 Quality-Net，该方法基于双向LSTM，Quality-Net 中话语水平的评估基于帧层次的评估，采用帧约束和遗忘门偏差的合理初始化，从话语级质量标签中学习有意义的帧级质量评估。

使用 Quality-Net 来预测 PESQ。模型需要将 $u\in R^{T(u)}$ 映射到 $Q\in R^1$ 上，$u$ 为输入语音，Q 为预测的质量分数，T(u) 可以是 $u$ 在时域波形上的采样点或者帧数。

使用幅度谱作为输入，Quality-Net 可以为语音质量评估提供一个分数，除此之外，为了避免 Quality-Net 变成一个黑箱，尽管训练数据的质量标签是在话语水平，Quality-Net 被设计成自动学习合理的帧水平质量。质量分数 Q 通过全局平均逐帧分数 $q_t$。

我们只有话语水平的语音质量标签，当噪声非平稳时，帧之间的语音失真程度并不相同，不能直接将话语水平的质量标签赋给每个独立的帧。考虑这样的情况，人认为高质量的语音应该没有任何的失真，所有帧都会分配一个很高的分数，基于此，作者提出了 Quality-Net 的目标函数中的条件逐帧约束，目标函数如下
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



将原本 Quality-Net 中的 BLSTM 换成 CNN，有利于梯度下降。将 Quality-Net 级联在语音增强的模型后面。为了近似PESQ，首先计算训练数据（干净语音和加噪语音）的 PESQ 分数，之后使用 MSE 损失训练  Quality-Net 最小化估计和真实的分数之间的差异。因为框架在变长语音上执行，所以需要在 CNN 后使用全局平均操作来处理变长数据。

训练好 Quality-Net 后，将其级联在语音增强模型后面，为了训练语音增强模型，固定 Quality-Net 的参数，最大化估计的质量分数，可以将 Quality-Net 看成是一个损失函数。为了防止语音增强模型生成的增强谱中存在额外的 artifacts，输出为 ratio mask（值为0-1）。因此，最优增强模型 $G^*$ 可以通过解下面的优化问题求解
$$
{G^*} = \arg \mathop {\min }\limits_G \sum\limits_{u = 1}^U {{{\left( {1 - Q\left( {{N_u} \otimes G\left( {{N_u}} \right),{C_u}} \right)} \right)}^2}} 
$$
U 为训练话语的总数，$N_u$ 和 $C_u$ 为带噪语音和干净语音的幅度谱，Q 表示 Quality-Net ，$ \otimes$ 表示按元素乘。（注意 ${G\left( {{N_u}} \right)}$ 生成的是一个 mask）

训练中，语音增强模型先使用 MSE 损失预训练，再使用 Quality-Net 损失微调。语音增强模型为 BLSTM，Quality-Net 是四个二维卷积层组成的 CNN，再加上全连接层。



### A Deep Learning Loss Function Based on the Perceptual Evaluation of the Speech Quality

为语音质量评估提出了一个感知指标，可以作为损失函数，该损失函数在 MSE 损失的基础上加上两个扰动项，分别表示听觉掩蔽和阈值效应。

受到 PESQ 算法的启发，一个对称和非对称的扰动，都是逐帧计算。对称扰动 $D_t^{(s)}$ 考虑增强后的频谱与干净频谱之间的绝对差。非对称扰动 $D_t^{(a)}$​ 通过对称扰动计算，但是为正和负响度差异赋予不同的权重，这是因为由于掩蔽效应，负差异（省略或衰减的频谱成分）与正差异（加性噪声）的感知不同。最终的损失函数为
$$
J = {1 \over T}\sum\limits_t {\left( {MS{E_t} + \alpha D_t^{(s)} + \beta D_t^{(a)}} \right)} 
$$
T 为训练批次中帧的数量，

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



