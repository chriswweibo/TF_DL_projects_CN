# 第10章 基于增强学习的游戏

与输入输出一一对应的监督学习不同，增强学习是另一类最大化问题：在给定环境下，寻找出一个行为策略以达到回报最大化（行为会互相作用，甚至会改变环境）。增强学习算法的目标不是一个明确而严格的结果（译者注：如分类回归问题的具体值，是一个明确而固定的数值），而是最大化最终得到的回报。此类算法会通过自由的反复试错来达成目标。如同幼童学步一般，算法会在实验环境中分析行为带来的反馈，然后找出实现最优回报的方式。这也类似人类尝试新游戏的情形：尝试寻求最优策略，在此之后尝试许多方法，然后最终决定行为准则。

迄今为止，没有增强学习算法能够在通用学习上能够媲美人类。较之算法，人类可以从更少的输入中学习，并且能够在非常复杂多样，结构有无的许多环境中进行学习。但是增强学习在一些特定问题中表现出了超出人类的能力（是的，他们的表现优于人类）。在特定游戏中，假设训练时间充足，增强学习算法能够给出令人惊叹的结果（比如AlphaGo https://deepmind.com/research/alphago/ ——第一个在围棋这种需要长期战略与直觉的复杂游戏中打败了世界冠军的程序）。

在本章中，我们将会呈现给您一个富有挑战性的项目：使用增强学习在雅达利游戏机上的“登月飞行器”游戏中学习如何正确使用指令。鉴于此游戏具有较少的指令，并可以根据少数几个数值描述游戏场景（甚至不用去看屏幕图像就能理解需要做什么。事实上此游戏第一版诞生于20世纪60年代，而且是纯文本的）并完成游戏，并且现在增强学习算法能够成功的解决它。

神经网络和增强学习刚刚开始交融。在20世纪90年代早期，IBM公司的格里·特索罗（Gerry Tesauro，应为Gerald Tesauro）研究员编写了著名的TD-Gammon，结合了前馈网络与时间差分学习（一种蒙特卡洛与动态规划的结合算法）来训练TD-Gammon进行西洋双陆棋游戏。西洋双陆棋是一个双人使用若干个骰子进行的游戏，如果读者希望进一步了解这个游戏，可以通过以下美国双陆棋联盟的网站来了解它的规则：http://usbgf.org/learn-backgammon/backgammon-rules-and-terms/rulesof-backgammon/ 当时TD-Gammon在西洋双陆棋上有着较好的表现是受益于西洋双陆棋是一个基于掷骰子的非确定性游戏，但当时无法在更具有确定性的游戏中获得较好的结果。近些年来，得益于谷歌深度学习的研究者，神经网络被证明能够帮助解决西洋双陆棋以外的问题，并且它在任何人的计算机上运行。最近，强化学习被列于未来深度学习乃至机器学习的爆点榜单之首。读者可以从下述链接访问谷歌大脑的人工智能科学家Ian Goodfellow的榜单，并看到强化学习位于榜首：https://www.forbes.com/sites/quora/2017/07/21/whats-next-for-deep-learning/#6a8f8cd81002



## 关于游戏

“登月飞行器”是一个1979年左右在雅达利游戏机上的街机游戏。游戏发布载体为特殊设计的匣子，游戏画面基于黑白矢量图形，为一个月球着陆舱接近月球表面的侧视图。着陆舱需要在若干个指定的地点之一着陆。由于地形差异，这些着陆点具有不同的宽度和难度，在不同的着陆点着陆也会具有不同的得分。通过界面玩家能够得知飞船的高度、速度、燃料余量、得分以及已用时间。在月球引力作用下，玩家需要通过消耗燃料调节着陆舱的旋转与反推（需要考虑惯性）来使得着陆舱着陆。燃料是这个游戏的关键。

当着陆舱燃料耗尽并接触到月面时游戏结束。在燃料耗尽之前，就算着陆舱已经坠毁，玩家也能继续游戏。玩家可用的指令有四个：左转、右转、推进，以及放弃着陆。当玩家使用推进命令时，着陆器将使用底部的推进器将着陆舱向其前方加速。而使用放弃着陆命令时，飞船将会调整姿态为头部朝上，并进行一次强力的推进以避免坠毁。

这个游戏的有趣之处在于代价与回报非常清楚，但有些是显而易见的（比如尝试着陆中消耗的油料），有些则被推迟到了着陆舱着陆的时候（只有当着陆完全停止时，你才知道着陆成功与否）。操作着陆舱着陆需要经济的规划燃料的使用，尽量不要过于浪费。着陆会提供一个得分，当着陆越困难、越安全的时候，这个得分会越高。



## OpenAI版游戏

OpenAI Gym是一个开发和比较强化学习算法的工具包，在其网站上提供了描述文档(https://gym.openai.com/)。  该工具包包括一个运行于Python 2或Python 3的Python包及其网站API，API用于上传您自己的算法的性能结果，并将它们与其他算法进行比较（在此将不深入探讨工具包的此用途）。  

这个工具包体现了强化学习的要素：一个环境和一个智能体：智能体可以在环境中执行或不执行操作，而环境将以新的状态(表示环境中的情况)和奖励来进行响应，奖励是一个分数，用于告诉智能体何种情况更优。Gym工具包提供了这个环境，因此您必须使用算法来编码智能体，让智能体在环境之中作出反应。环境由`env`处理，`env`是一个类，它带有一些关于强化学习中的环境的方法。当您使用语句为特定游戏场景创建它时，它会被实例化： `gym.make('environment')`。以下是官方文档中的一个示例： 

```python
import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
	observation = env.reset()
	for t in range(100):
		env.render()
		print(observation)
		# taking a random action
		action = env.action_space.sample()
		observation, reward, done, info = \
							env.step(action)
		If done:
			print("Episode finished after %i \
					timesteps" % (t+1))
			break
```

在本例中，运行环境是CartPole-V0。在CartPole-V0游戏中，需要处理一个控制问题，游戏场景是轨道上有一辆装有立杆的小车，立杆和小车由一个节点相接，玩家对小车施加向前或向后的力以保持杆尽可能长时间的直立。你可以通过在youtube上观看这个视频来观看游戏的详细情况：https://www.youtube.com/watch?v=qMlc43-sc-lg。 这是IIT Madras的动力学和控制实验室的实际实验的一部分，该实验使用了基于类神经元的自适应元素，此方法能解决许多困难的控制问题。

Cartpole问题在以下论文中被提出：能解决困难的学习控制问题的类神经元自适应元素（http://ieeexplore.ieee.org/Document/6313077/），此论文由BARTO，Andrew G.；SUTTON，Richard S.；Anderson，Charles W发表于《IEEE transactions on systems, man, and Cybernetics》。 

下面是在示例中使用的\`env\`的简要说明：  

* `reset()`：将环境的状态重置为初始默认条件。它将返回初始条件的观察结果。
* `step(action)`：做出动作，将环境向后推进一个时间片。它返回一个含有四个变量的向量：观察、奖励、完成状态和额外信息。观察表示当前的环境状态。在游戏中，环境状态是一个向量，不同的价值向量表示不同的环境。例如，在一个物理游戏CartPole-V0中，环境向量由购物车的位置、购物车的速度、杆的角度和杆的速度组成。奖励是前一个动作所取得的分数（若想得到当前总分数，您需要手动对所有奖励进行加和）。完成状态是一个布尔变量，它告诉您在游戏中是否处于结束状态(游戏结束)。额外信息将提供额外的诊断信息，这些信息不用于您的算法，而是用于调试环境。  
* `render( mode='human', close=False)`：渲染一帧环境。默认模式为用户友好模式，该模式下它会进行一些用户友好的操作，比如可能弹出一个窗口。传递\`close\`标志将指示渲染引擎不生成此类窗口。  

这些命令产生的效果如下：  

* 设置CartPole-V0的初始环境
* 运行1000个步骤
* 随机选择对购物车施加正向或反向力
* 将结果可视化

这种方法的有趣之处在于您可以轻松地更换游戏，只需将一个不同的字符串提供给`gym.make`方法（例如，尝试MsPacman-V0或Breakout-V0，或者从可用列表中选择任意一个，可用列表可以通过`gym.print(envs.registry.all())`得到），此方法可以在不改变代码的情况下测试不同的环境。OpenAI Gym通过在所有环境中使用公共接口，使得测试用户的算法对不同问题的泛化能力变得简单。此外，它还为您根据该模式推理、理解和解决智能体环境问题提供了一个框架：在时间t-1时，一个状态和奖励被推送给一个智能体，该智能体以一个动作发生反应，产生一个新的状态，并在t时产生一个新的奖励：

【这里应该有一张图】

图1：环境和智能体如何通过状态、行为和奖励进行交互

在OpenAI Gym中的每一个不同的游戏中，动作空间(智能体响应的命令)和观察空间(表示状态)都会发生变化。您可以在您设置了一个环境之后，通过使用一些打印命令来查看它们是如何变动的：

```python
print(env.action_space)
print(env.observation_space)
print(env.view_space.High)
print(env.observation_space.low)
```

## 在Linux上安装OpenAI(Ubuntu14.04或16.04)

建议在Ubuntu系统上安装OpenAI环境。OpenGym AI是在Linux系统创建的，而对Windows也有些许支持。根据您的系统配置，可能需要先安装一些附加软件：  

```
apt-get install -y python3-dev python-dev python-numpy libcupti-dev libjpeg-turbo8-dev make golang tmux htop chromium-browser git cmake zlib1gdev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
```

 我们建议使用Anaconda。您可以在https://www.anaconda.com/download/上找到此Python发行版的下载地址。  

在设置系统需求之后，安装OpenGym AI及其所有模块非常简单： 

```shell
git clone https://github.com/openai/gym 
cd gym 
pip install -e .[all]
```

对于这个项目中，如何使用Box2D是一个重要问题。Box2D是一个2D物理引擎，在2D环境中提供类似仿真游戏的真实物理的渲染。可以通过在Python中运行以下命令来测试Box2D模块是否工作：  

```python
import gym 
env = gym.make('LunarLander-v2') 
env.reset() 
env.render()
```

如果所提供的代码运行没有问题，则可以继续。在一些情况下，Box2D可能会出现问题，比如可能会出现 https://github.com/cbfinn/gps/issues/34 中报告的，或者其他问题。在基于Python3.4的conda环境中安装Gym可能有所帮助：  
```shell
conda create --name gym python=3.4 anaconda gcc=4.8.5 
source activate gym 
conda install pip six libgcc swig 
conda install -c conda-forge opencv 
pip install --upgrade tensorflow-gpu 
git clone https://github.com/openai/gym 
cd gym 
pip install -e . 
conda install -c https://conda.anaconda.org/kne pybox2d
```
这个安装序列应该允许您创建一个Conda环境，该环境适合我们将在本章中介绍的项目。  
## 在OpenAI Gym中使用登月飞行器  
LunarLander-v2是由OpenAI的工程师奥列格·克里莫夫(Oleg Klimov)开发的场景，灵感来自最初的雅达利游戏机上的月球着陆器游戏(https://github.com/olegklimov)。在此场景中，您需要将您的着陆舱带到始终位于坐标x=0和y=0的着陆台。此外，着陆舱的实际x和y坐标是已知的，它们的值存储在状态向量的前两个元素中，状态向量包含用于强化学习算法的所有信息，以决定在某个时刻采取的最佳行动。  
状态向量使得问题变得容易许多。因为您不必处理不确定的自身位置或目标位置，而这在机器人中是一个常见问题。
【这里应该有一张图】
在每个时刻，着陆舱有四种可能的行动可供选择：  
* 什么都不做
* 左转
* 右转
* 反冲

这个问题有趣的地方在于复杂的奖励系统： 

* 从屏幕顶部移动到着陆台附近后以零速度着陆会获得100到140点的奖励(可以在着陆台外着陆)
* 如果着陆舱选择离开着陆台而不是留下，它将失去先前的一些奖励
* 每章（指一次游戏过程）在着陆舱撞毁或安全着陆时，分别提供额外的-100或 +100点奖励。  
* 与地面接触的每条腿+10奖励
* 点燃主引擎需要每帧消耗0.3点奖励(但燃料是无限的)
* 过关给予200点奖励

这个游戏与离散命令(实际上它们是二值的：全推力或无推力)配合得很好，正如模拟的作者所说，根据庞蒂亚金的最大原理（Pontryagin's maximum principle），最好是全速启动或完全关闭引擎。

该游戏也可以使用一些简单的启发式方法来解决，比如基于目标距离并使用比例积分导数（proportional integral derivative，PID）控制器来控制下降的速度和角度。PID是一种用于有反馈条件下控制系统的工程解决方案。您可以从以下地址获得更详细的说明：https://www.csimn.com/CSI_pages/PIDforDummies.html 

## 用深度学习来探索强化学习  

在此项目中，我们不深入展开如何构建启发式算法或构造PID（但它仍然能解决人工智能中的许多问题）。相反，我们打算利用深度学习为智能体提供足够的智能来顺利的操作月球登陆器。

强化学习理论为解决这类问题提供了几个框架： 

* 基于价值的学习：通过评估处于某种状态的回报，比较不同可能状态下的奖赏，选择导向最佳状态的行为。Q学习（Q-learning，质量函数学习）就是这种方法的一个例子。  
* 基于策略的学习：根据来自环境的奖励来评估不同的控制策略后选择最终达到最佳效果的策略。  
* 基于模型的学习：在智能体内建立环境模型，从而允许智能体模拟不同的行为及其相应的奖励。

在我们的项目中，我们将使用基于价值的学习框架；具体来说，我们将使用基于Q学习的经典强化学习方法，有证据表明这种方法在以下情况是有效的：游戏中智能体需要决定一系列将导致游戏后期延迟奖励的动作。该方法由C.J.C.H.Watkins于1989年在他的博士论文中设计，也称为Q学习。Q学习基于以下理念：一个智能体在环境中，考量当前状态的情况并定义一系列将导致最终回报的行为： 

【这里应该有一张图】

上面的公式描述了一个状态s在一个动作a之后是如何导致一个奖励r和一个新的状态s’的。从游戏的初始状态开始，该公式应用一系列动作，这些动作一个接一个地转换每个后续状态，直到游戏结束。然后，您可以将游戏想象为一系列动作所链接的状态。然后，您还可以描述上述公式如何通过一系列动作a将初始状态s转换为最终状态s’和最终奖励r。

在强化学习中，策略是行动a的最序列。策略可以用一个称为Q的函数逼近，给定当前状态s和可能的动作a，作为输入，它将提供从该动作得到的最大报酬r的估计： 

【这里应该有一张图】

 这种方法显然是贪婪的，这意味着我们只是在精确的状态下选择最佳的操作，因为我们期望总是在每一步中选择最佳的操作将导致最好的结果。因此，在贪婪的方法中，我们不考虑可能导致奖励的行为链，而只考虑下一个行为，a。不过，很容易证明，如果符合以下条件，我们可以有信心地采取贪婪的做法，并利用这种策略获得最大回报：  

* 我们能够找到一个完美的策略先知，Q\*
* 我们在一个信息完美的环境中工作(这意味着我们可以了解关于环境的一切)
* 环境遵循马尔可夫原理(详情见提示框)

马尔可夫原理指出，未来(状态，回报)只取决于现在，而不是过去，因此，我们可以简单地通过观察现状而忽略以前发生的事情来获得最好的结果。 

事实上，如果我们将Q函数构建为一个递归函数，我们只需要探索(使用广度优先搜索方法)对要测试的操作的当前状态的影响，并且递归函数将返回最大可能的回报。  

这种方法在计算机模拟中非常有效，但在现实世界中却毫无意义：  

* 环境大多是基于概率的。即使你采取了行动，你也不一定能得到确切的回报
* 环境与过去联系在一起，单靠现在无法描述未来，因为过去可能会产生隐藏的或长期的后果
* 环境不是完全可以预测的，所以你不能事先知道一个行动的回报，但是你可以在事后知道它们(这被称为后验条件)
* 环境非常复杂。你不能在一个合理的时间内计算出一个行动的所有可能的后果，因此你不能确定一个行动产生的最大回报

解决方案是采用一个近似Q函数，它可以考虑概率结果，并且不需要通过预测来探索所有的未来状态。显然，它应该是一个真正的近似函数，因为在复杂的环境中构建值搜索表是不切实际的(有些状态空间可能需要连续的值，从而使可能的组合变得无限多)。此外，该函数可以离线学习，这意味着智能体需要利用以往的经验(这表明记忆能力非常重要)。 

以前也有人尝试过用神经网络来逼近q函数，但唯一成功的应用是TD_Gammon，这是一个仅由多层感知器驱动的强化学习的双子棋程序。TD_Gammon达到了超越人类水平的游戏，但在当时，它的成功不能复制在不同的游戏，如国际象棋或围棋上。

这在当时导致了一种信念，即神经网络并不真正适合于计算出一个Q函数，除非游戏是某种随机的(你必须在双子棋中掷骰子)。直到2013年，一篇关于深度强化学习的论文《使用深度强化学习玩雅达利游戏机》（Playing Atari with deep reinforcement learning，https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf），描述了Volodymyr Minh等人将Q学习应用于旧雅达利游戏，从而证伪了这个观点。

这篇论文演示了如何使用神经网络学习Q函数来玩一系列的雅达利游戏(例如架束导弹、突围、拉力赛、乒乓球、Q伯特、海洋行动和空间入侵者)，只需输入视频(60 Hz采样率，210×160分辨率，RGB视频)并输出操纵杆和射击按钮命令。本文将这种方法命名为深度Q网络(Deep Q-Network， DQN)，并介绍了经验回放和探索与利用的概念，我们将在下一节中讨论这些概念。当尝试将深度学习应用于强化学习时，这些概念有助于克服一些关键问题： 

* 没有足够的例子可供学习——这是强化学习所必需的，使用深度学习时更是不可或缺的。  
* 在行动和有效奖励之间有较长的延迟，这需要在获得奖励之前处理一系列可变长度的行动。  
* 一系列高度相关的动作序列(因为动作通常会影响到后续的动作)，这可能导致任何随机梯度下降算法过度适应最新的示例，或者只是非最优收敛(随机梯度下降期望随机示例，而不是相关示例)。  

Mnih和其他研究人员的论文《通过深度强化学习实现人类级别的控制》(http://www.davidqiu.com:8888/research/nature14236.pdf)只是证实了DQN的有效性，即利用DQN进行了更多的游戏，并将DQN的性能与人类玩家和经典强化学习算法进行比较。在许多游戏中，DQN被证明比人类的表现得更好，尽管长期策略仍然是算法的一个问题。在某些游戏中，比如“突围”，智能体发现了一些狡猾的策略，比如挖一条隧道穿过墙壁，以轻松的方式把球送进并摧毁墙壁。在其他的比赛中，比如蒙特祖马的复仇，智能体仍然表现得较为迷茫。  

在论文中，作者们详细地讨论了智能体如何理解赢得突围游戏的细节，并给出了DQN函数的响应表，展示了如何将高回报分数分配给先在墙上挖一个洞然后让球通过的行为。 

## Q学习的技巧
神经网络实现的Q学习是不稳定的，直到一些技巧使之成为可能并可行。在深度Q学习中有两个重要支柱，我们将随后讨论。尽管最近开发了其他的算法变体，以解决原解的性能和收敛性问题，但我们不展开讨论这些变体：双重Q学习，延迟Q学习，贪婪GQ，快速Q学习等。我们将要探索的两个主要支柱是经验回放以及探索与利用之间不断减少的权衡取舍。

通过经验回放，我们简单地将所观察到的游戏状态存储在一个预先确定大小的队列中，当队列满时，将丢弃旧的序列。在存储的数据中有一些元组，包括当前状态、应用操作、结果得到的状态和获得的奖励。在此基础上我们需要考虑一个由当前状态和操作组成的元组。我们可以通过观察智能体在环境中行动导致的结果考虑导致结果的根本原因，进一步的，我们可以将元组(当前状态和动作)作为奖励的预测器(x向量)。在此基础上我们可以估计当前环境下与行动直接相关的奖励，以及在游戏结束时将获得的奖励来进行行为决策。

给定这样的存储数据(作为智能体的记忆)，我们对其中的一些数据进行采样，以便创建一个批，并使用获得的批处理来训练我们的神经网络。但是，在将数据传递给网络之前，我们需要定义我们的目标变量（y向量）。由于抽样状态大多不是最终状态，所以我们可能会有一个零奖励或只是部分奖励来匹配已知的输入(当前状态和选择的动作)。部分奖励存在缺陷的地方是它只描述了我们需要知道的信息的一部分，而我们的目标是知道在我们目前的状态采取行动(我们的x值)后，游戏结束时我们将得到的全部奖励。

在这种情况下，由于我们没有获得最终奖励，我们尝试使用我们现有的Q函数来近似这个值，以便估计剩余回报。这将是我们正在考虑的(状态，动作)元组的结果的最大值。得到它之后，我们将用Bellman方程对它的值进行衰减。

在谷歌的软件工程师萨尔·坎迪多博士的这篇优秀的教程中，可以读到关于这种现在经典的强化学习方法的解释： http://robotics.ai.uiuc.edu/~scandido/?Developing_Reinforcement_Learning_from_the_Bellman_Equation ，其中将未来的奖励衰减后添加到了当前的奖励中。  

使用小值(接近于零)的衰减使Q函数更倾向于短期奖励，而使用高衰减值(接近1)使Q函数更倾向于未来收益。

第二个非常有效的技巧是利用系数来进行探索与利用之间的权衡。在探索过程中，期望智能体尝试不同的行动，以找到给定特定状态下的最佳行动方案。在利用过程中，智能体利用它在以前的探索中学到的知识，并简单地决定在这种情况下应该采取什么最好的行动。  

在探索与利用之间找到一个好的平衡，与我们前面讨论的经验回放密切相关。在开始对DQN算法进行优化时，我们只需使用一组随机的网络参数，就像我们在本章的简单介绍性示例中所做的那样，简单的随机抽样。在这种情况下，智能体将探索不同的状态和操作，并形成初始Q函数。对于复杂的游戏，例如登月飞行器，使用随机选择不会使智能体走得太远，而且从长远来看，甚至可能会十分低效，因为只有在智能体之前做了一系列正确的事情时才能访问一些具有奖励的状态和操作，而随机操作很难达到这些状态。事实上，在这种情况下，DQN算法将很难找到适当地分配正确的奖励给一个行动，因为它将永远不会看到一次完整的游戏。由于游戏足够复杂，它不太可能被行动的随机序列解决。  

正确的方法是在学习的机会和使用已学会的东西之间寻求平衡，以正确的动作序列让智能体深入到游戏较深的状态后学习尚未解决的问题。这类似于通过一系列连续的近似先找到一个解决方案，每次让智能体更接近于安全和成功着陆的正确动作序列。因此，智能体应该先随机学习，找出在特定情况下要做的最合适的事情，然后应用所学到的东西，进入新的情况，通过随机选择，这些新情况也将被依次解决、学习和应用。  
这是通过使用递减值作为阈值来实现的。让智能体决定在游戏中的某个点上，是否采取随机选择，观察发生的事情，或者利用到目前为止学到的知识，并利用它在这一点上做出最佳的操作。从均匀分布[0，1]中选取一个随机数，智能体将其与递减值`epsilon`进行比较，如果随机数大于`epsilon`，则使用其学习的神经Q函数。否则，它将从可用的选项中选择一个随机操作。在此之后，`epsilon`将会略微减小。`epsilon`被初始化为最大值1.0，根据衰减因子的设置，它将随着时间的推移而或多或少地减小，得到一个不是零(不可能永不随机移动)的最小值，以便总是有可能通过意外的意外(最小的开放因素)学习新的和意想不到的东西。 

## 理解DQN的局限性  
即使是深度Q学习也存在一些限制，无论您是通过视觉图像或其他对于环境的观察来近似q函数：
* 这种近似需要很长的时间才能收敛，有时它并不能很顺利地实现：你甚至可能看到神经网络的学习指标在很多时期都在恶化而不是变得更好。  
* 作为一个基于贪心策略的方法，Q学习提供的方法与启发式方法没有什么不同：它指出了最好的方向，但不能提供详细的规划。当处理长期目标或需要分解为次级目标的目标时，Q学习表现得很糟糕。 
* Q学习的机制导致的另一个结果是，它并不是从通用的角度来理解游戏的进行，而是从一个特定的角度来理解(它重复了它在训练中所体验到的有效的东西)。因此，任何引入游戏的新颖事物(在训练过程中从未体验过)都会破坏算法，使其完全无效。在算法中引入新游戏时也是如此，它根本无法执行。  

## 开始编码  
在经过长时间的强化学习和DQN方法之后，我们终于准备好开始编码了，对如何操作OpenAI GEM环境和如何设置Q函数的DQN近似有了所有的基本理解。我们只是开始导入所有必要的包：  
```python
import gym 
from gym import wrappers 
import numpy as np 
import random, tempfile, os 
from collections import deque 
import tensorflow as tf
```
`tempfile`模块用于生成临时文件和目录，这些文件和目录可以用作数据文件的临时存储区域。`collections`模块中的`deque`命令创建一个双端队列（实际上它是一个列表），您可以在列表的开头或结尾添加删除元素。需要注意的是，它可以设置为给定的容量。当容量为满时，旧元素将被丢弃，以便为新元素创建位置。  

我们将使用一系列表示智能体、智能体的大脑(这里是DQN)、智能体的记忆和环境的类来构造这个项目。环境类可以在OpenAI Gym中找到，但它需要我们编写代码以正确地连接到智能体。 

## 定义智能体的大脑  

项目的第一步是创建一个包含所有神经网络代码的类，`Brain`，这个类将用来计算Q函数的近似值。这个类将包含必要的初始化、用于创建合适的TensorFlow计算图的代码、一个简单的神经网络(不是一个复杂的深度学习结构，而是一个用于我们项目的简单的、实用的网络，您可以稍后用更复杂的网络结构替换它)，以及学习和预测操作的函数。 

让我们从初始化开始。首先，我们需要知道与我们从游戏中得到的信息相对应的状态输入(nS)的大小，以及我们可以在游戏中执行的操作相对应的动作输出(nA)的大小。我们强烈建议为网络设置域(Scope，非必须)，它是一个字符串，帮助我们隔离为不同目的创建的网络。建议使用它的原因是在我们的项目中需要使用两个网络，一个用于处理下次的奖励，一个用于猜测最终奖励。  

然后，我们需要定义优化器（这里使用Adam优化器）的学习率。 

你可以从以下论文了解Adam优化器： https://arxiv.org/abs/1412.6980 。它是一种非常有效的基于梯度的优化方法，只需很少的调整即可正常工作。Adam优化算法是一种随机梯度下降算法，类似于带有动量的RMSProp算法。我们可以从加州大学伯克利分校计算机视觉评论快报上的这篇文章了解更多信息： https://theberkeleyview.wordpress.com/2015/11/19/berkeleyview-for-adam-a-method-forstochastic-optimization/ 。根据我们的经验，将样本分批训练深度学习算法是最有效的解决方案之一，它需要用户对学习速率进行一些调整。
最后，我们还需要提供：
* 神经网络架构(如过想替换示例中提供的基础网络)。
* global_step，这是一个全局变量，它将提示迄今为止提供给DQN网络的训练批次的数量。
* 用于存储TensorBoard日志的目录，TensorBoard是TensorFlow的标准可视化工具。
```python
class Brain:    
	"""    
	A Q-Value approximation obtained using a neural network.    
	This network is used for both the Q-Network and the Target Network.
	"""    
	def __init__(self, nS, nA, scope="estimator",
				learning_rate=0.0001,
				neural_architecture=None,
				global_step=None, 
				summaries_dir=None):
		self.nS = nS        
		self.nA = nA        
		self.global_step = global_step        
		self.scope = scope        
		self.learning_rate = learning_rate        
		if not neural_architecture:            
			neural_architecture = self.two_layers_network        
		# Writes Tensorboard summaries to disk        
		with tf.variable_scope(scope):            
			# Build the graph
			self.create_network(network=neural_architecture,
			learning_rate=self.learning_rate)
		if summaries_dir:                
			summary_dir = os.path.join(summaries_dir,
										"summaries_%s" % scope)
		if not os.path.exists(summary_dir):
			os.makedirs(summary_dir)                
			self.summary_writer = \
				tf.summary.FileWriter(summary_dir)  
			else:                
				self.summary_writer = None
```
` tf.summary.FileWriter `命令将事件文件在目标目录下(`summary_dir`)初始化，我们将在该目录中存储学习过程的度量。相关调用句柄保存在` self.summary_writer`中，稍后我们将使用它来存储我们关注的训练前后的相关指标，以监视和调试学习过程。  
下一个要定义的方法是我们将用于这个项目的默认神经网络。它接受输入层和我们将要使用的隐层的各自节点数量作为输入。输入层是由我们正在使用的状态长度定义的，比如我们的例子中使用的向量，也可以是原始DQN文件中的图像。
这些层可以简单地使用TensorFlow中Layers模块中的高层抽象接口来定义(https://www.tensorflow.org/api_guides/python/contrib.layers)。我们选择原生的全连接层，使用ReLU(整流线性激活函数)作为两个隐层的激活函数，输出层使用线性激活函数。
32的隐层大小非常适合我们的任务，当然如果你愿意，可以增加它。此外，我们不在网络中使用Dropout。显然，这里的问题不是过拟合，而是学习样本的质量。只有通过不断提供有用的不相关状态下的序列，和相应的对最终回报的良好估计，才能提高学习内容的质量。有用的状态序列，特别是在探索与利用之间权衡的情况下，是不使网络过拟合的关键。在强化学习问题中，如果您陷入以下两种情况之一，您就已经过拟合了：
* 次优性：算法提出次优解，即我们的着陆器学会了一种粗略的着陆方法后持续使用此方法，因为此方法至少着陆了。
* 无助：算法陷入了一种学习的无助状态，也就是说，它没有找到正确着陆的方法，所以它只是接受了以尽可能不坏的方式撞毁。
这两种情况对于强化学习算法(如DQN)来说确实很难克服，除非该算法有机会在游戏中探索其他解决方案。不时地采取随机行动并不是像是读者一开始认为的一种简单的把事情搞砸的方法，而是一种避免陷阱的策略。
另一方面，在更大的网络中，失效的神经元可能会产生一些问题，我们需要使用`tf.nn.leaky_relu`来让它重新起效。失效的ReLU总是输出相同的值，通常是零，并且它会阻止反向传播更新。  
从TensorFlow 1.4开始，就有了`leaky_relu`激活函数。如果您正在使用以前版本的TensorFlow，您可以创建一个： 
```python
def leaky_relu(x，alpha=0.2)：
	return tf.nn.relu(x) - alpha * tf.nn.relu(-x)  
```
我们现在开始对我们的`Brain`类进行编码，为它添加一些更多的函数： 
```python
def two_layers_network(self, x, layer_1_nodes=32,
							layer_2_nodes=32):
	layer_1 = tf.contrib.layers.fully_connected(x, layer_1_nodes,
										activation_fn=tf.nn.relu)    
	layer_2 = tf.contrib.layers.fully_connected(layer_1,
										layer_2_nodes, 
										activation_fn=tf.nn.relu) 
	return tf.contrib.layers.fully_connected(layer_2, self.nA, 
										activation_fn=None) 
```
 方法`create_network`结合了输入、神经网络、损失和优化器。损失只是将原始奖励与估计结果之间的差取平方，并将所学习的批次中的所有样本的损失取平均值。我们使用Adam优化器将损失最小化。  

此外，还使用TensorBoard记录了一些值：  

* 批的平均损失，以便在训练期间跟踪训练状况。  
* 批最大预测奖励，以跟踪指示最成功的行动的预测。  
* 批平均预测奖励，为了跟踪预测一般动作的趋势。

下面是`create_network`的代码：

```python
 def create_network(self, network, learning_rate=0.0001):
        # Placeholders for states input        
        self.X = tf.placeholder(shape=[None, self.nS],                                 
        						dtype=tf.float32, name="X")        
        # The r target value        
        self.y = tf.placeholder(shape=[None, self.nA],                                 
        						dtype=tf.float32, name="y")        
        # Applying the choosen network        
        self.predictions = network(self.X)        
        # Calculating the loss        
        sq_diff = tf.squared_difference(self.y, self.predictions)        
        self.loss = tf.reduce_mean(sq_diff)        
        # Optimizing parameters using the Adam optimizer        
        self.train_op = tf.contrib.layers.optimize_loss(self.loss,                        
        				global_step=tf.train.get_global_step(),                        
        				learning_rate=learning_rate,                        
        				optimizer='Adam')        
        # Recording summaries for Tensorboard        
        self.summaries = tf.summary.merge([            
        	tf.summary.scalar("loss", self.loss),            
        	tf.summary.scalar("max_q_value",                             
        		tf.reduce_max(self.predictions)),            
        	tf.summary.scalar("mean_q_value",                             
        		tf.reduce_mean(self.predictions))])
```

 这个类包含了方法`predict`和`fit`。方法`fit`以状态矩阵s作为输入，以奖励r的向量作为输出。它还包括希望训练迭代的次数(在原论文中，建议每批只使用一个迭代，以避免对当前批过拟合)。然后在当前会话中，我们对输入与输出进行拟合，并输出用于追踪训练的记录值(我们创建网络时定义的)。 

```python
def predict(self, sess, s):
	"""
	Predicting q values for actions
	"""
	return sess.run(self.predictions, {self.X: s})

def fit(self, sess, s, r, epochs=1):
	"""
	Updating the Q* function estimator
	"""
	feed_dict = {self.X: s, self.y: r}
	for epoch in range(epochs):
		res = sess.run([self.summaries, self.train_op,
						self.loss,
						self.predictions,
						tf.train.get_global_step()],
						feed_dict)
		summaries, train_op, loss, predictions,
									self.global_step = res
    if self.summary_writer:
    	self.summary_writer.add_summary(summaries,
    									self.global_step)
```

`global_step`作为结果之一被返回。它是一个计数器，可以跟踪到目前为止在训练中使用的批数量，并且记录下来供以后使用。  

## 为经验回放创建记忆  

在定义了大脑(TensorFlow神经网络)之后，我们的下一步是定义记忆，即数据存储，这将为DQN的学习过程提供动力。在每一个用于训练的游戏会话中，一步由一个状态和一个动作组成，与随后的状态和该会话的最终奖励一起被记录下来(最终奖励只有在该会话结束时才会知道)。  

添加一个标志告诉观察是否是终止状态就完成了记录信息的集合。其想法是将某些动作不仅与即时奖励(可能为空或非常少)关联，而且与结束奖励相关联，从而将该会话中的每一步与其关联起来。  

`memory`类含有一个固定大小的队列，该队列中充满了以往经历的信息，我们可以很容易的对其进行采样和提取。由于它的大小固定，我们需要将较老的样本从队列中弹出，从而使可用的样本始终位于最后。  

这个类包括了`__init__`方法，用于初始化双端队列并固定其大小，也包括`__len__`方法(以用来获取内存是否为满，等待数据足够丰富时才进行训练，此时样本具有更好的随机性和多样性)，`add_memory`用于在队列中添加样本，以及从记忆中恢复所有数据的`recall_memory`：

```python
class Memory:
	"""
	A memory class based on deque, a list-like container with
	fast appends and pops on either end (from the collections
	package)
	"""
	def __init__(self, memory_size=5000):
		self.memory = deque(maxlen=memory_size)
	def __len__(self):
		return len(self.memory)

    def add_memory(self, s, a, r, s_, status):
    	"""
    	Memorizing the tuple (s a r s_) plus the Boolean flag status,
    	reminding if we are at a terminal move or not
    	"""
    	self.memory.append((s, a, r, s_, status))

    def recall_memories(self):
    	"""
    	Returning all the memorized data at once
    	"""
    	return list(self.memory)
```

## 创建智能体  

我们要编码的下一个类是智能体。它具有初始化和维护大脑(提供近似Q函数)和记忆的作用。更重要的是，它是与环境互动的主要对象。它的初始化设置了一系列的参数，这些参数大多是固定的（根据我们以往在这个游戏中训练智能体的经验）。当然你可以在在初始化智能体时显式更改它们：  

* `epsilon = 1.0`是探索—利用参数的初始值。1.0迫使智能体完全依赖于探索，即随机移动。  
* `epsilon_min = 0.01`设置探索—利用参数的最小值：值0.01表示着陆舱随机移动的可能性为1%，而不总是基于Q函数反馈。这意味着我们一直有一个非常小的机会找到另一个更优的方式完成游戏。  
* `epsilon_decay = 0.9994`是调节`epsilon`减小到最小值的速度。在这个设置中，在大约5，000个游戏会话之后，它将被调整到最小值。一般来说，这个参数应该为算法提供至少200万个可供学习的样本。  
* `gamma = 0.99`是奖励折现因子，我们近似的Q函数将根据它来衡量当前奖励和未来奖励，它控制了算法玩游戏的方式是短视或远视(在月球着陆器这个游戏中，最好是远视，因为奖励只在登月舱着陆时才会得到)。
* `learing_rate = 0.0001`是Adam优化器学习样本的学习速率。  
* `epochs = 1`是神经网络拟合批样本而使用的训练轮数。
* `batch_size = 32`是批大小。
* `memory = Memory(memory_size=250000)`是内存队列的大小。  

使用预置参数可以确保智能体在当前项目正常工作。对于不同的OpenAI环境，您可能需要经过尝试找到不同的最佳参数。  

初始化还将定义TensorBoard日志的放置位置(默认情况下是`experiment`目录)、用于估计下一个即时奖励的模型，以及估计最终奖励的另一个模型。此外，初始化中还将定义一个保存器(`tf.train.Saver`)，它可以将整个会话序列化后保存到磁盘，以便以后恢复它并将其用于玩真正的游戏。  

上述两个模型在同一个会话中初始化，使用不同的域名(一个是`q`，由TensorBoard监视的估计下一个奖励的模型；另一个是`target_q`)。使用两个不同的域名可以方便地处理神经元的系数，从而可以用类中的另一个方法交换它们： 

```python
class Agent:
	def __init__(self, nS, nA, experiment_dir):
		# Initializing
		self.nS = nS
		self.nA = nA
		self.epsilon = 1.0  # exploration-exploitation ratio
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.9994
		self.gamma = 0.99  # reward decay
		self.learning_rate = 0.0001
		self.epochs = 1  # training epochs
		self.batch_size = 32
		self.memory = Memory(memory_size=250000)

        # Creating estimators
        self.experiment_dir =os.path.abspath\
	        			("./experiments/{}".format(experiment_dir))
	    self.global_step = tf.Variable(0, name='global_step',
	    									trainable=False)
	    self.model = Brain(nS=self.nS, nA=self.nA, scope="q",
	    					learning_rate=self.learning_rate,
	    					global_step=self.global_step,
	    					summaries_dir=self.experiment_dir)
	    self.target_model = Brain(nS=self.nS, nA=self.nA,
	    							scope="target_q",
	    							learning_rate=self.learning_rate,
	    							global_step=self.global_step)
	
        # Adding an op to initialize the variables.
        init_op = tf.global_variables_initializer()
        # Adding ops to save and restore all the variables.
        self.saver = tf.train.Saver()

        # Setting up the session
        self.sess = tf.Session()
        self.sess.run(init_op)
```

`epsilon`表示探索与利用之间的权衡，它在`epsilon_update`方法中被更新，该方法只是通过将实际`epsilon`乘以`epsilon_decay`来修改实际的`epsilon`，除非它已经达到了允许的最小值：

```python
def epsilon_update(self, t):
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay
```

使用`save_weights`与`load_weights`方法来保存以及恢复会话：

```python
def save_weights(self, filename):
	"""
	Saving the weights of a model
	"""
	save_path = self.saver.save(self.sess,
								"%s.ckpt" % filename)
	print("Model saved in file: %s" % save_path)

def load_weights(self, filename):
	"""
	Restoring the weights of a model
	"""
	self.saver.restore(self.sess, "%s.ckpt" % filename)
	print("Model restored from file")
```

`set_weight`和`target_model_update`一起用Q网络的权重来更新目标Q网络(`set_weights`是一个通用的、可重用的函数，您也可以在其他解决方案中使用)。由于这两个作用域的命名不同，因此很容易从可训练变量列表中枚举每个网络的变量。通过枚举，当前会话将对变量执行赋值： 

```python
 def set_weights(self, model_1, model_2):
 	"""
 	Replicates the model parameters of one
 	estimator to another.
 	model_1: Estimator to copy the parameters from
 	model_2: Estimator to copy the parameters to
 	"""
 	# Enumerating and sorting the parameters
 	# of the two models
 	model_1_params = [t for t in tf.trainable_variables() \
 						if t.name.startswith(model_1.scope)]
 	model_2_params = [t for t in tf.trainable_variables() \
 						if t.name.startswith(model_2.scope)]
 	model_1_params = sorted(model_1_params,
 							key=lambda x: x.name)
 	model_2_params = sorted(model_2_params,
 							key=lambda x: x.name)
 	# Enumerating the operations to be done
 	operations = [coef_2.assign(coef_1) for coef_1, coef_2 \
 					in zip(model_1_params, model_2_params)]
 	# Executing the operations to be done
 	self.sess.run(operations)

 def target_model_update(self):
 	"""
 	Setting the model weights to the target model's ones
 	"""
 	self.set_weights(self.model, self.target_model)
```

`act`方法是执行策略的核心，因为它将基于`epsilon`来决定当前移动是随机还是采取尽可能好的操作。如果它要采取尽可能好的操作，它将要求经过训练的Q网络为每个可能的下一步动作提供一个奖励估计(通过在月球登陆器游戏中按四个按钮的二进制编码)，它将返回将会获得最大预测奖励(解决方案的贪婪方法)的操作： 

```python
def act(self, s):
	"""
	Having the agent act based on learned Q* function
	or by random choice (based on epsilon)
	"""
	# Based on epsilon predicting or randomly
	# choosing the next action
	if np.random.rand() <= self.epsilon:
		return np.random.choice(self.nA)
	else:
		# Estimating q for all possible actions
		q = self.model.predict(self.sess, s)[0]
		# Returning the best action
		best_action = np.argmax(q)
		return best_action
```

本类的最后一个方法是`replay`。这个方法很关键，因为它使DQN算法的学习过程变为可能。因此我们在此深入讨论它的运作原理。重播方法所做的第一件事是从先前游戏场景的记忆中取一批样(我们在初始化时定义了批的大小，这些内存只是包含状态、动作、奖励、下一个状态的变量，以及是否是最终状态的标志变量)。随机抽样使模型能够通过对网络权值的逐步调整，一批又一批地找出最优的系数，从而学到最终的Q函数近似。

然后该方法观察样本的最终标志，非最终奖励需要更新，以表示您在游戏结束时得到的奖励。这是通过使用目标网络来完成的，目标网络表示上一次学习结束时固定的Q函数网络的快照。向目标网络提供以上状态，在用伽玛因子折现后，将得到的报酬与当前报酬相加。  

使用当前的Q函数可能会导致学习过程的不稳定性，导致无法得到一个令人满意的Q函数网络。 

```python
 def replay(self):
 	# Picking up a random batch from memory
 	batch = np.array(random.sample(\
 			self.memory.recall_memories(), self.batch_size))
 	# Retrieving the sequence of present states
 	s = np.vstack(batch[:, 0])
 	# Recalling the sequence of actions
 	a = np.array(batch[:, 1], dtype=int)
 	# Recalling the rewards
 	r = np.copy(batch[:, 2])
 	# Recalling the sequence of resulting states
 	s_p = np.vstack(batch[:, 3])
 	# Checking if the reward is relative to
 	# a not terminal state
 	status = np.where(batch[:, 4] == False)
 	# We use the model to predict the rewards by
 	# our model and the target model
 	next_reward = self.model.predict(self.sess, s_p)
 	final_reward = self.target_model.predict(self.sess, s_p)
    
    if len(status[0]) > 0:
    	# Non-terminal update rule using the target model
    	# If a reward is not from a terminal state,
    	# the reward is just a partial one (r0)
        # We should add the remaining and obtain a
        # final reward using target predictions
        best_next_action = np.argmax(\
        				next_reward[status, :][0], axis=1)
        # adding the discounted final reward
        r[status] += np.multiply(self.gamma,
        		final_reward[status, best_next_action][0])

        # We replace the expected rewards for actions
        # when dealing with observed actions and rewards
        expected_reward = self.model.predict(self.sess, s)
        expected_reward[range(self.batch_size), a] = r
        # We re-fit status against predicted/observed rewards
        self.model.fit(self.sess, s, expected_reward,
        				epochs=self.epochs)
```

当非最终状态的奖励被更新时，批样本被输入到神经网络中进行训练。  

## 指定环境

要实现的最后一个类是`Environment`。实际上，环境是由`gym`命令提供的，尽管您需要一个好的封装来使它与前面的`agent`类一起工作。在初始化时，它启动月球着陆器游戏，并设置关键变量，如`nS`、`nA`(状态和动作的维度)、`agent`和累积奖励(通过提供最后100步的平均值来进行测试)： 

```python
class Environment:    
    def __init__(self, game="LunarLander-v2"):        
        # Initializing        
        np.set_printoptions(precision=2)        
        self.env = gym.make(game)        
        self.env = wrappers.Monitor(self.env, tempfile.mkdtemp(),                               
                                force=True, video_callable=False)        
        self.nS = self.env.observation_space.shape[0]        
        self.nA = self.env.action_space.n        
        self.agent = Agent(self.nS, self.nA, self.env.spec.id)
        
        # Cumulative reward        
        self.reward_avg = deque(maxlen=100)
```

 然后，我们编写了`test`、`train`和`incremental`(增量训练)方法的代码，这些方法是`learn`方法的封装。  

使用增量训练需要一些技巧，以免破坏迄今为止训练已经取得的结果。导致这个问题的原因是当我们重新启动时，`Brain`有预先训练过的系数，但实际上记忆是空的(我们称之为冷重启)。由于智能体的记忆是空的，过少且具有局限性而不能良好的支持学习。因此，所提供的示例的质量对于学习来说确实不怎么样(这些示例大部分是相互关联的，并且非常局限于少数几个新体验的场景)。通过使用非常低的`epsilon`(我们建议将其设置为最低，0.01)，可以减少训练被破坏的风险：通过这种方式，网络将在大多数情况下简单地重新学习自己的权重，因为它将为每个状态建议它已经知道的操作，它的性能不应该恶化，而是在内存中有足够的示例之前以稳定的方式振荡，并且它将在具有足量数据后再次开始改进。  

下面是正确的训练和测试方法的代码： 

```python
def test(self):
    self.learn(epsilon=0.0, episodes=100,
                trainable=False, incremental=False)

def train(self, epsilon=1.0, episodes=1000):
    self.learn(epsilon=epsilon, episodes=episodes,
                trainable=True, incremental=False)

def incremental(self, epsilon=0.01, episodes=100):
    self.learn(epsilon=epsilon, episodes=episodes,
                trainable=True, incremental=True)
```

最后一种方法是`learn`，它用于将智能体与环境交互学习所需的所有东西准备好，并进行学习。该方法接受`epsilon`(从而覆盖智能体拥有的任何先前的`epsilon`)、在环境中运行游戏的次数、是否对其进行训练(布尔标志)，以及训练是否从以前的模型(另一个布尔标志)的训练中继续进行。  

在第一个代码块中，我们可以为Q值近似函数加载先前训练的网络权重：  

1. 测试网络，看看它是如何工作的； 
2. 利用更多样本继续进行之前的训练。 

然后，该方法深入到一个嵌套迭代中。外部迭代循环运行游戏的次数(每一次运行月球着陆器游戏都将进行到结束)。而内部迭代则是经过最多1000步组成的一次游戏运行。 

在迭代的每一步中，神经网络将给出下一步的动作。如果它在测试中，它总是简单地提供下一个最佳动作的答案。如果它正在训练中，根据epsilon的值，它可能不会建议最好的动作，而是随机移动。 

```python
def learn(self, epsilon=None, episodes=1000,
           trainable=True, incremental=False):
   """
   Representing the interaction between the enviroment
   and the learning agent
   """
   # Restoring weights if required
   if not trainable or (trainable and incremental):
       try:
           print("Loading weights")
           self.agent.load_weights('./weights.h5')
       except:
           print("Exception")
           trainable = True
           incremental = False
           epsilon = 1.0

   # Setting epsilon
   self.agent.epsilon = epsilon
   # Iterating through episodes
   for episode in range(episodes):
       # Initializing a new episode
       episode_reward = 0
       s = self.env.reset()
       # s is put at default values
       s = np.reshape(s, [1, self.nS])

       # Iterating through time frames
       for time_frame in range(1000):
           if not trainable:
               # If not learning, representing
               # the agent on video
               self.env.render()
           # Deciding on the next action to take
           a = self.agent.act(s)
           # Performing the action and getting feedback
           s_p, r, status, info = self.env.step(a)
           s_p = np.reshape(s_p, [1, self.nS])

           # Adding the reward to the cumulative reward
           episode_reward += r

           # Adding the overall experience to memory
           if trainable:
               self.agent.memory.add_memory(s, a, r, s_p,
                                           status)
           # Setting the new state as the current one
           s = s_p

           # Performing experience replay if memory length
           # is greater than the batch length
           if trainable:
               if len(self.agent.memory) > \
                       self.agent.batch_size:
                   self.agent.replay()
                   
           # When the episode is completed,
           # exiting this loop
           if status:
               if trainable:
                   self.agent.target_model_update()
               break

       # Exploration vs exploitation
       self.agent.epsilon_update(episode)
       # Running an average of the past 100 episodes
       self.reward_avg.append(episode_reward)
       print("episode: %i score: %.2f avg_score: %.2f"
               "actions %i epsilon %.2f" % (episode,
                                           episode_reward,
                                           np.average(self.reward_avg),
                                           time_frame,
                                           epsilon)

   self.env.close()
   if trainable:
       # Saving the weights for the future
       self.agent.save_weights('./weights.h5')
```

在行动之后，我们收集所有信息(初始状态、选择的动作、获得的奖励和随后的状态)并保存到记忆中。在这个时间帧中，如果记忆足够大，可以为逼近Q函数的神经网络创建样本批，则运行训练。当该次游戏的所有时间帧都被消耗完时，当前DQN的权重被存储到另一个网络中，在下一次游戏中学习时，作为一个稳定的参考。 

## 运行强化学习

最后，在扯一大堆关于强化学习和DQN的内容，并编写了项目的完整代码之后，您可以使用脚本或Jupyter Notebook来运行它。利用将所有代码功能放在一起的`Environment`：

```python
lunar_lander = Environment(game="LunarLander-v2") 
```

在实例化之后，您需要运行`train`，从`epsilon=1.0`开始，并将目标设置为5000次游戏(这相当于大约220万个状态、行为和奖励的链式变量样本)。我们提供的实际代码参数被设置为训练完整的DQN模型，考虑到GPU的可用性及其计算能力，它可能需要一些时间：

```python
lunar_lander.train(epsilon=1.0, episodes=5000)
```

最后，该类将完成所需的训练，将模型保存在磁盘上(可以随时运行)。您可以使用一个简单的命令来运行TensorBoard，从shell运行：

```shell
tensorboard --logdir=./experiments --port 6006 
```
这些图表将出现在您的浏览器上，并且可以在`localhost:6006`上查看它们： 

<这里应该有一张图>

图4：训练中损失图，峰值代表学习中的突破，例如在80万时，它开始安全地降落在地面上。  

损失图与其他项目不同的是，优化的过程仍然是损失逐渐减少，但在这一过程中出现了许多峰值和问题：  

这里表示的图表是运行该项目一次的结果。由于训练过程有随机因素，因此在您自己的计算机上运行项目时，您可能会获得稍微不同的绘图。  

<这里应该有一张图>

图5：单批训练过程中最大Q值图  

最大预测Q值和平均预测Q值说明了同样的情况。该网络在最后得到了改进，尽管它可以稍微回溯一点，并在更优值上停留很长一段时间：  

<这里应该有一张图>

图6：单批训练过程中平均Q值图  

只有在最后100个最终奖励的平均值中，您才能看到一条逐渐增加的路径，这表示了DQN网络的持续和稳步改进：  

<这里应该有一张图>

图7：每个学习阶段结束时实际获得分数图，它更清楚地描述了DQN不断增长的过程。  

使用来自训练输出的信息，而不是来自TensorBoard的信息，您还会发现平均操作数根据`epsilon`值的值变化。开始时，完成一局游戏所需的动作次数少于200次。当`epsilon`为`0.5`时，平均动作数趋于稳定增长，在750左右达到峰值(着陆舱已经学会用火箭来抵消重力)。  

最后，网络发现这是一个次优策略，当`epsilon`低于`0.3`时，完成一局游戏的平均操作数也会下降。在这一阶段，DQN正在探索如何以更有效的方式成功地着陆：  

图8：`epsilon`(探索/利用率)与DQN网络效率之间的关系，网络效率为完成一局游戏所使用操作数。  

如果出于任何原因，您认为网络需要更多的示例和学习，您可以使用`incremental`方法重复学习，记住在这种情况下`epsilon`应该非常低：  

```python
lunar_lander.incremental(episodes=25, epsilon=0.01)  
```

训练结束后，如果您需要查看结果并了解平均每100局游戏DQN可以得分多少(理想的目标是分数>=200)，您可以运行以下命令：  

```python
lunar_lander.test()
```

## 鸣谢
在这个项目结束时，我们非常感谢Peter Skvarenina，他的项目“登月飞行器II”(https://www.youtu.com/Watch?v=yiAmrZuBaYU)是我们项目的主要灵感来源，他也在我们制作自己的DQN的过程中给出了很多提示和建议。 

## 总结 
在这个项目中，我们探索了增强算法在OpenAI环境中可以实现什么，并且我们编写了一个TensorFlow程序，它能够学习如何在一个以智能体、状态、动作和随后的奖励为特征的环境中估计最终的奖励。这种方法称为DQN，目的是用神经网络方法逼近Bellman方程。本章的产出是一个程序，该程序可以在训练结束时通过读取游戏状态并在任何时候通过决策来采取正确的行动，以成功地玩“登月飞行器”游戏。 