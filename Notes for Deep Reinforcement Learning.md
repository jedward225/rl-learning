# Notes for Deep Reinforcement Learning

---

## Pre-lessons

- MDP(Markov Decision Process) 和 MRP(Markov Reward Process)

- 动作价值函数 $Q^\pi(s,a)$

- 状态价值函数 $V^\pi(s)$

- 贝尔曼方程
  $$
  V^\pi(s) =\mathbb{E}_\pi[G_t|S_t = s] =  \mathbb{E}_\pi[R_t + \gamma V^\pi(S_{t+1}) | S_t = s] = \sum_{a\in \mathcal A} \pi(a | s)\left(r(s, a) + \gamma\sum_{s'\in \mathcal S}p(s'|s, a)V^\pi(s') \right) \\
  Q^\pi(s, a) = \mathbb{E}_\pi[R_t + \gamma Q^\pi(S_{t+1}, A_{t +1}) | S_t = s, A_t = a] = r(s, a) + \gamma \sum_{s' \in \mathcal S} p(s' | s, a) \sum_{a' \in \mathcal A}\pi(a'|s')Q^\pi(s', a')
  $$
  去掉中间的等式，简写：
  $$
  V^\pi(s) = \sum_{a\in \mathcal A} \pi(a | s)\left(r(s, a) + \gamma\sum_{s'\in \mathcal S}p(s'|s, a)V^\pi(s') \right) \\
  Q^\pi(s, a) = r(s, a) + \gamma \sum_{s' \in \mathcal S} p(s' | s, a) \sum_{a' \in \mathcal A}\pi(a'|s')Q^\pi(s', a')
  $$
  
- 贝尔曼方程的矩阵形式
  $$
  \mathcal{V} = \mathcal R + \gamma \mathcal{PV}
  $$
  ![image-20250307162411711](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250307162411711.png)

- 策略提升定理
  对于两个策略 $\pi, \pi'$，如果满足以下性质，$\pi'$ 就是 $\pi$ 的策略提升：
  对任意状态 $s$ ，$Q^\pi(s, \pi'(s)) \ge V^\pi(s)$
  进而，$\pi$ 和 $\pi'$ 满足：
  对任意状态 $s$，$V^{\pi'}(s) \ge V^\pi(s)$ 

- $\epsilon$ - 贪婪
  $$
  \pi(a | s) = \begin{cases} \epsilon / |\mathcal A| + 1 - \epsilon &\text{, if } a = \arg\max_{a'}Q(s, a') \\ \epsilon / |\mathcal A| & \text{, otherwise} \end{cases}
  $$
  
- 占用度量

  - 状态访问分布 $\nu^\pi(s)$，state visitation distribution
    用 $P_t^\pi(s)$ 表示采取策略 $\pi$ 使智能体在 $t$ 时刻状态为 $s$ 的概率
    $$
    \nu^\pi(s) = (1 - \gamma)\sum_{t = 0}^{\infty} \gamma^t P_t^\pi(s) \\
    \nu^\pi(s') = (1 - \gamma)\nu_0(s') + \gamma \int P(s' | s, a)\pi(a | s)\nu^\pi(s) \text{d}s\text{d}a
    $$

  - 占用度量 $\rho^\pi(s, a)$，occupancy measure
    $$
    \rho^\pi(s, a) = \nu^\pi(s)\pi(a | s)
    $$

  - [定理1]  MDP中，$\rho^{\pi_1} = \rho^{\pi_2} \Leftrightarrow  \pi_1 = \pi_2$

  - [定理2]  给定一个合法占用度量 $\rho$，可生成该占用度量的唯一策略 $\pi_\rho$：
    $$
    \pi_\rho = \frac{\rho(s, a)}{\sum_{a'}\rho(s, a')}
    $$

- 贝尔曼回溯算子
  $$
  \mathcal B V(s) = \max_a \left\{r(s, a) + \gamma \sum_{s' \in \mathcal S}p(s' | s, a)V(s')\right\}
  $$
  性质：

  - 作用于价值函数
  - 返回一个价值函数
  - 是一个压缩映射，存在唯一不动点
  - 如果可能的话，优化价值 $V^\pi = \mathcal B^\pi\mathcal B^\pi\cdots\mathcal B^\pi V$

---

## Model-Based Value Methods

- 基于**策略迭代**的动态规划算法
  $$
  \pi^0 \overset{策略评估}{\longrightarrow} V^{\pi^0} \overset{策略提升}{\longrightarrow} \pi^1 \overset{策略评估}{\longrightarrow} V^{\pi^1} \overset{策略提升}{\longrightarrow} \cdots \overset{策略提升}{\longrightarrow} \pi^* \\
  策略评估: V^{k + 1}(s) = \sum_{a\in \mathcal A}\pi(a|s)\left(r(s, a) + \gamma\sum_{s'\in \mathcal S}P(s'|s, a)V^k(s')\right)\\
  策略提升: \pi'(s) = \arg \max_a Q^\pi(s, a) = \arg\max_{a}\{r(s, a) + \gamma\sum_{s'}P(s'|s, a)V^\pi(s')\}
  $$

  ![image-20250307204129974](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250307204129974.png)
  
- 基于**价值迭代**的动态规划算法
  $$
  V^{k + 1}(s) = \max_{a\in \mathcal A}\{r(s, a) + \gamma\sum_{s'\in \mathcal S}P(s' | s, a)V^k(s')\}
  $$

  ![image-20250307204535895](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250307204535895.png)

---

## Model-Free Value Methods

- 蒙特卡洛方法（高方差，无偏）
  $$
  N(s) \gets N(s) + 1 \\
  V(s) \gets V(s) + \frac{1}{N(s)}(G_t - V(s))
  $$

- 时序差分算法（低方差，有偏）
  $$
  V(s_t) \gets V(s_t) + \alpha[r_{t} + \gamma V(s_{t + 1}) - V(s_t)]
  $$

- Sarsa 算法
  $$
  Q(s_t, a_t) \gets Q(s_t, a_t) + \alpha[r_t + \gamma Q(s_{t+1}, a_{t + 1}) - Q(S_t, a_t)]
  $$
  ![image-20250307205438041](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250307205438041.png)
  
- 多步Sarsa算法
  
  <img src="C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250404100731754.png" alt="对于 n 步 Sarsa 方法的回溯树。每一个黑色的圆圈都代表了一个状态，每一个白色的圆圈都代表了一个动作。在这个无穷多步的 Sarsa 里，最后一个状态就是它的终止状态" style="zoom:67%;" />
  $$
  Q(s_t, a_t) \gets Q(s_t, a_t) + \alpha[r_t + \gamma r_{t + 1} + \cdots +\gamma^{n - 1}r_{t + n -1 }+ \gamma^n Q(s_{t + n}, a_{t + n}) - Q(s_t, a_t)]
  $$
  ![image-20250404101841396](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250404101841396.png)
  
  拓展：[**使用重要性采样的多步离线学习法**和**多步树回溯算法**(N-step Tree Backup Algorithm)](https://wnzhang.net/teaching/sjtu-rl-2024/slides/4-model-free-control.pdf)
  
- Q-learning 算法
  $$
  Q(s_t, a_t) \gets Q(s_t, a_t) + \alpha[r_t + \gamma \max_{a}Q(s_{t +1}, a) - Q(s_t, a_t)]
  $$
  ![image-20250404101901224](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250404101901224.png)
  
- On-policy 和 Off-policy
  ![img](https://hrl.boyuai.com/static/400.78f393db.png)

- 资格迹，Eligibility Trace
  
  资格迹结合了MC方法和TD方法的优点，能够在时间和状态之间分配奖励或惩罚，提升学习效率。
  
  - **定义**
    资格迹为**每个状态或状态-动作对**分配一个值，表示在学习过程中该状态或状态-动作对的**重要性**。在每个时间步，资格迹会衰减并在访问相应状态或执行相应动作时增加。
  
    <img src="C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250404102304452.png" alt="image-20250404102304452" style="zoom: 50%;" />
    
  - 前向视角（Forward View）：
  
    通过引入 $\lambda$ ，将不同步长的回报进行加权平均：
    $$
    G_t^\lambda = (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} G_t^{(n)}
    $$
    其中，$\lambda \in [0, 1]$ 为衰减因子。当 $\lambda=0$ 时，$G_t^0$ 等价于一步 TD 方法；当 $\lambda=1$ 时，$G_t^1$ 等价于蒙特卡洛方法。
  
  - 后向视角（Backward View）：
  
    后向视角通过引入资格迹向量 $\mathbf{e}$，用于记录每个状态自上次访问以来的衰减程度。在每个时间步 $t$，对于每个状态 $s$：
    $$
    \mathbf{e}_t(s) = \gamma \lambda \mathbf{e}_{t-1}(s) + \mathbb{I}(S_t = s)
    $$
  
  - TD($λ$) 算法
  
    结合资格迹的时序差分方法称为 TD($λ$) 算法，具体来说：
    $$
    V(s) \leftarrow V(s) + \alpha \delta_t \mathbf{e}_t(s)
    $$
    其中，$\alpha$ 为学习率，$\delta_t$ 为时序差分误差，TD-error：
    $$
    \delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
    $$

---





**DQN（Deep Q-Network）**

**DQN**将深度神经网络与Q-Learning相结合，使用卷积神经网络近似Q函数，在Atari游戏上取得革命性成果。

关键技巧：

\1. **Experience Replay**：将交互经验存储在回放池中，随机小批量采样来打破数据相关性；

\2. **Target Network**：固定目标Q网络一段时间再更新，避免网络剧烈震荡。

------

**[Actor-Critic架构](https://zhida.zhihu.com/search?content_id=242259687&content_type=Article&match_order=1&q=Actor-Critic架构&zhida_source=entity)**

为平衡高方差和低偏差，在**Actor-Critic**中将策略函数（Actor）和价值函数（Critic）同时学习：

• **Actor**：输出策略 \pi_\theta(a|s) ；

• **Critic**：估计价值函数 V_w(s) 或 Q_w(s,a) ；

• 每一次采样时，利用 Critic 来估计动作优势（Advantage），更新 Actor 的梯度，使得训练更稳定。

\1. **on-policy**

- **PPO**、**A2C/A3C** 等。
- 在 PPO 中，为了保证策略稳定性，需要用最新的（或近似最新的）策略去采样交互数据，然后紧接着对这批数据进行更新，更新后又要丢弃旧数据，重新采样。
- 好处是可以严格保证“策略分布”和“数据分布”一致，**收敛性**更易分析。
- 坏处是**数据利用率低**，因为一旦策略更新，这批数据就算“过期”了，很难重复使用来训练新策略。

\2. **off-policy**

- **Q-Learning**、**DQN**、**SAC**、**TD3** 等。
- DQN 中的经验回放池就是典型的 off-policy：采集数据时使用的是 \epsilon-greedy 策略（带随机探索），但在更新 Q 函数时，我们朝着 “greedy” 或某个更优的目标策略方向改进。
- 好处是可以**重复使用**历史数据，样本效率更高；还可以从人类示教数据或其他智能体的轨迹中学习。
- 坏处是策略与数据分布不一致带来的复杂性，可能更难保证收敛，更新过程也更易出现分布偏移（Distribution Shift）问题。

### **on/off-policy和online/offline的区别**

\1. **不同维度**

- **on/off-policy** 解决的问题：**“采样策略”与“目标策略”的一致程度**。
- **online/offline** 解决的问题：**“是否可以持续交互收集新数据”**。

\2. **常见组合**

**离线强化学习基本一定是 off-policy，但在线强化学习可以既有 on-policy，也可以有 off-policy。**

- **在线 + on-policy**：比如 PPO、A2C 这些算法，需要跟环境交互，采集数据时就使用当前策略，采完就更新，旧数据不再使用。
- **在线 + off-policy**：比如 DQN，虽然也是在线与环境交互，但 DQN 会把交互数据放到 replay buffer，后面训练时用到的旧数据不一定来自当前的策略，所以是 off-policy。
- **离线 + off-policy**：这最常见。离线 RL 必然不能和“当前目标策略”一致地采样，因为数据集已经固定了，通常是其他策略或历史操作生成的数据，所以几乎都是 off-policy。
- **离线 + on-policy**：理论上很难，因为离线数据本身就是固定收集的，跟当前想学的策略无法保持一致——所以离线强化学习通常都被视为 off-policy 的特例。

---





## DQN

- Q-Learning 回顾
  
  <img src="C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250404103531261.png" alt="image-20250404103531261" style="zoom: 67%;" />
  
- Q-learning 中的误差（下面的 $\omega$ 和上面的 $\theta$ 对应，只是符号差异）
  $$
  \omega^* = \arg\min_\omega \frac{1}{2N} \sum_{i = 1}^N \left[Q_\omega(s_i, a_i) - \left(r_i + \gamma \max_{a'}Q_\omega(s_i', a')\right)\right]^2
  $$

- **经验放回**
  
  维护一个**回放缓冲区**（replay buffer），将每次从环境中采样得到的四元组数据（状态、动作、奖励、下一状态）存储到回放缓冲区中，训练 Q 网络的时候再从回放缓冲区中随机采样若干数据来进行训练。
  
  **作用**：

  - 可以打破样本之间的相关性，使样本满足独立假设。
  - 提高样本效率，十分适合深度神经网络的梯度学习。
  
- **目标网络**
  
  DQN算法目标在于用神经网络 $Q_\omega(s, a)$ 逼近 $r + \gamma\max_{a'}Q_\omega(s', a')$，而 TD 误差目标本身就包含神经网络的输出（$\max_{a'}Q_\omega(s', a')$），因此神经网络训练非常不稳定。为此想先固定住TD 误差目标的 Q 网络，我们设计两套神经网络：
  
  - 原来的训练网络 $Q_\omega(s, a)$，用于计算原本的损失函数中的 $Q_\omega(s, a)$ 项。
  - 目标网络 $Q_{\omega^-}(s, a)$，用于计算原先损失函数中的 $(r + \gamma \max_{a'}Q_{\omega^-}(s', a'))$ 项。

![image-20250404120401300](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250404120401300.png)

## DQN 优化技巧

### Double DQN

在实现 DQN 上，Q 值往往是被高估的。问题在于目标值的计算：
$$
\text{Target:} \quad y_j = r_j + \gamma\underline{ \max_{\mathbf{a}'_j} Q_{\phi'} \left( \mathbf{s}'_j, \mathbf{a}'_j \right)}
$$
划线部分是问题所在，这是因为 $E[\max(X_1, X_2)] \geq \max(E[X_1], E[X_2])$（Jensen不等式），当我们取最大值时，实际上选择了那些具有更大噪声的值。

传统DQN的 TD 误差目标 $r + \gamma \max_{a'}Q_{\omega^-}(s', a')$ 可以写成如下形式：
$$
r + \gamma Q_{\omega^-}\left(s', \arg\max_{a'}Q_{\omega^{-}}(s', a')\right)
$$

而DDQN的改进之处就在于将 TD 误差目标改写成了：
$$
r + \gamma Q_{\omega^-}\left(s', \arg\max_{a'}\underline{Q_{\omega}(s', a')}\right)
$$

### Dueling DQN

利用优势函数 $A(s,a) = Q(s,a) - V(s)$，在同一个状态下，所有动作的优势值之和为 0。更改后的 Q网络：
$$
Q_{\theta,\alpha,\beta}(s, a) = V_{\theta,\alpha}(s) + A_{\theta, \beta}(s, a) - \max_{a'}A_{\theta, \beta}(s, a')
$$
此时 $V(s) = \max_{a} Q(s, a)$​，可以确保 $V$​ 值建模的唯一性。实现过程中还可以用平均代替最大化操作，以获得更好

的稳定性：
$$
Q_{\theta,\alpha,\beta}(s, a) = V_{\theta,\alpha}(s) + A_{\theta, \beta}(s, a) - \frac{1}{|\mathcal A|}\sum_{a'}A_{\theta, \beta}(s, a')
$$
此时 $V(s) = \frac{1}{|\mathcal A|}\sum_{a'}A_{\theta, \beta}(s, a')$。

<img src="https://datawhalechina.github.io/easy-rl/img/ch7/7.6.png" alt="img" style="zoom: 200%;" />

### 优先级经验回放（prioritized experience replay，PER）

采样时可以由一个重要性指标函数 $f((s_t, a_t, r_t, s_{t+1}))$ 来衡量样本的重要性。可以将 TD 误差和优先级 $p$ 联系起来——$p_i = |\delta_i| + \epsilon$ 或者是 $p_i = \frac{1}{\text{rank}(i)}$，其中 $\text{rank}$ 是基于 $|\delta_i|$ 的等级评定，于是重要性采样下，（式子中 $\alpha$ 是一个超参数，$\alpha = 0$ 时对应回均匀采样）
$$
\omega_i = (NP(i))^{-\beta}, \text{ where } P(i) = \frac{p_i^\alpha}{\sum_{k}p_k^{\alpha}}
$$
$\beta$ 是训练过程中将会模拟退火（Anneal）到1的超参数，这么设置是由于随着训练增加，更新会趋近于无偏。

> [!NOTE]
>
> 模拟退火法中的退火是一种简单的实现是线性退火，例如，若设置初始值 0.6，终止值 1.0，最大迭代步数为 100，则第 $0 \leq t < 100$ 步取 $\beta = 0.6 + t(1.0 − 0.6)/99$

<img src="https://datawhalechina.github.io/easy-rl/img/ch7/7.7.png" alt="img" style="zoom: 67%;" />

### 其他改进内容：多步学习、噪声网络和值分布强化学习等

- Q-Learning 多步学习变体的目标:
  $$
  r_t^{(k)} + \gamma^k \max_{a'}Q(s_{t + k}, a') = \sum_{k = 0}^{n - 1} \gamma^k r_{t + k} + \gamma^k \max_{a'}Q(s_{t + k}, a')
  $$

- 噪声网络（使用一个额外的噪声流将噪声加入线性层 $y = (W x + b)$ 中）
  $$
  y = (Wx+b) + ((W_{\text{noisy}}\odot\epsilon_w)x + b_{\text{noisy}}\odot\epsilon_b)
  $$
  $W_\text{noisy}$ 和 $b_\text{noisy}$ 都是可训练的参数，而 $\epsilon_w$ 和 $\epsilon_b$ 是将退火到 0 的随机标量。

---

## Policy-Gradient (Policy-Based Method)

轨迹 $\tau = \{s_1, a_1, s_2, a_2, \cdots, s_t, a_t\}$，继而

$$
p_{\theta}(\tau) = p(s_1)p_\theta(a_1|s_1)p(s_2|s_1, a_1)\cdots = p(s_1)\prod_{t = 1}^T p_\theta(a_t|s_t)p(s_{t+1}|s_t, a_t)
$$
其中，奖励函数 $R(\tau)$ 代表一个轨迹 $\tau$ 的奖励。
$$
J(\theta) = \mathbb E_{\tau\sim p_\theta(\tau)} \left[\sum_{t}r(s_t, a_t)\right] = \bar R_\theta = \sum_{\tau}R(\tau)p_{\theta}(\tau)
$$


$$
\nabla \bar R_\theta & = & \sum_{\tau}R(\tau)p_{\theta}(\tau)\nabla \log p_\theta(\tau)\\
&= &\mathbb E_{\tau \sim p_\theta(\tau)}[R(\tau)\nabla \log p_\theta(\tau)]\\
&\approx& \frac{1}{N}\sum_{k = 1}^N R(\tau^k)\nabla\log p_\theta(\tau^k) \\
& = & \frac{1}{N}\sum_{k = 1}^N\sum_{t = 1}^{T_n}R(\tau^k)\nabla \log p_\theta(a_t^k | s_t^k)
$$

<img src="https://datawhalechina.github.io/easy-rl/img/ch4/4.8.png" alt="img" style="zoom: 50%;" />

- **降低高方差实现技巧:**

  - Add a Baseline:
    $$
    \nabla\bar R_\theta \approx \frac{1}{N}\sum_{k = 1}^N\sum_{t = 1}^{T_n}(R(\tau^k) - b)\nabla \log p_\theta(a_t^k | s_t^k)\text{, 其中 b 可以是 }\mathbb E[R(\tau)]
    $$
  - Assign Suitable Credit:
    $$
    \nabla\bar R_\theta \approx \frac{1}{N}\sum_{k = 1}^N\sum_{t = 1}^{T_n}(\sum_{t' = t} ^ {T_n} \gamma^{t' - t}r_{t'}^k - b)\nabla \log p_\theta(a_t^k | s_t^k)=\frac{1}{N}\sum_{k = 1}^N\sum_{t = 1}^{T_n}(G_t^n - b)\nabla \log p_\theta(a_t^k | s_t^k)
    $$

- **REINFORCE 算法**
  $$
  \nabla \bar R_\theta = \mathbb E_{\pi_\theta}\left[\sum_{t = 1}^T\left(\sum_{t' = t}^T\gamma^{t' - t}r_{t'}\right)\nabla_{\theta}\log \pi_\theta(a_t|s_t)\right] \approx \frac{1}{N}\sum_{n = 1}^N\left[\sum_{t = 1}^{T}\nabla \log p_\theta(a_t^n | s_t^n)\left(\sum_{t' = t}^T r(s_{t'}^n, a_{t'}^n) \right)\right]
  $$

  **Causality**原则要求在 $t'$ 时刻的策略不应该影响 $t(t \leq t')$ 时刻的奖励，上面的 $\sum_{t' = t}^T r(s_{t'}^n, a_{t'}^n)$ 也即 $\sum_{t' = t}^T\gamma^{t' - t}r_{t'}$，可以用符号未来价值函数 $G_t$ 来表示，显然有 $G_t = r_{t} + \gamma G_{t + 1}$。
  
  ![](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250404120413815.png)

<img src="https://datawhalechina.github.io/easy-rl/img/ch4/4.28.png" alt="img" style="zoom:50%;" />

## Actor-Critic

> 之前的讲了**基于值函数的方法（DQN）**和**基于策略的方法（REINFORCE）**，其中基于值函数的方法只学习一个价值函数，而基于策略的方法只学习一个策略函数。那么，一个很自然的问题是，**有没有什么方法既学习价值函数，又学习策略函数**呢？答案就是 **Actor-Critic**。

Policy-gradient 更一般的形式：
$$
g = \mathbb E\left[\sum_{t = 0}^T \psi_t \nabla_{\theta}\log \pi_\theta(a_t|s_t)\right]
$$
其中，$\psi_t$ 可以有很多形式：

1. $$\sum_{t' = 0} ^ T \gamma^{t'}r_{t'}:轨迹的总回报$$；
2. $$\sum_{t' = t} ^ T \gamma^{t' - t}r_{t'}:动作a_t之后的回报$$；
3. $$\sum_{t' = t} ^ T \gamma^{t' - t}r_{t'} - b(s_t):基准线版本的改进$$；
4. $$Q^{\pi_\theta}(s_t, a_t): 动作价值函数$$；
5. $$A^{\pi_\theta}(s_t, a_t): 优势函数$$；
6. $$r_t + \gamma V^{\pi_\theta}(s_{t + 1}) - V^{\pi_\theta}(s_t): 时序差分残差$$。

REINFORCE 通过蒙特卡洛采样的方法对策略梯度的估计是无偏的，但是方差非常大。之前提到过，我们可以用形式(3)引入**基线函数**（baseline function）来减小方差。此外，我们也可以采用 Actor-Critic 算法估计一个动作价值函数，代替蒙特卡洛采样得到的回报，这便是形式(4)。

Actor（策略网络）和 Critic（价值网络）

- Actor 的更新采用策略梯度的原则。

- Critic 的价值函数 $V_{\omega}$ 的损失函数：
  $$
  \mathcal L(\omega) = \frac{1}{2}(r +\gamma V_{\omega}(s_{t + 1}) - V_\omega(s_t))^2
  $$
  对应的梯度为：
  $$
  \nabla_\omega\mathcal L(\omega) = -(r + \gamma V_\omega(s_{t + 1}) - V_\omega(s_t))\nabla_\omega V_\omega(s_t)
  $$

  ![](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250404132636286.png)

- 生成对抗网络和Actor-Critic

  ![image-20250404133100973](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250404133100973.png)

### 同步优势 Actor-Critic（A2C） 

A2C 在上面的 Actor-Critic 算法的基础上增加了并行计算的设计，全局行动者和全局批判者在 Master 节点维护。每个 Worker 节点的增强学习智能体通过协调器和全局行动者、全局批判者对话。

<img src="C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250404133127293.png" alt="image-20250404133127293" style="zoom:50%;" />

### 异步优势 Actor-Critic（A3C）

在 A3C 的设计中，协调器被移除。每个 Worker 节点直接和全局行动者和全局批判者进行对话。Master 节点则不再需要等待各个 Worker 节点提供的梯度信息，而是在每次有 Worker 节点结束梯度计算的时候直接更新全局 Actor-Critic。

由于不再需要等待，A3C 有比A2C 更高的计算效率。但是同样也由于没有协调器协调各个 Worker 节点，Worker 节点提供梯度信息和全局 Actor-Critic 的一致性不再成立，即每次 Master 节点从 Worker 节点得到的梯度信息很可能不再是当前全局 Actor-Critic 的梯度信息。

## 信赖域策略优化（TRPO） & 近端策略优化（PPO）

> 之前介绍的 Actor-Critic 算法有一个明显的缺点：当策略网络是深度模型时，沿着策略梯度更新参数，相同的步长可能使策略在策略空间中有完全不一样幅度的更新，这使得本就难选择的步长 $\eta_\theta$ 在实际应用中变得更加难以选择。举个例子，考虑当前的策略 $\pi = (\sigma(\theta), 1 − \sigma(\theta))$ 的两种不同情况。这里 $\sigma(\theta)$ 是 Sigmod 函数。假设在第一种情况下， $\theta$ 从 $\theta = 6$ 更新到了 $\theta = 3$。而在另一种情况中， $\theta$ 从 $\theta = 1.5$ 更新到了 $\theta = -1.5$。两种情况 $\pi_\theta$ 在参数空间中的更新幅度都是 $3$。然而，在第一种情况下，$\pi_\theta$ 在策略空间中从几乎是 $\pi \approx (1.00, 0.00)$ 变成了 $\pi \approx (0.95, 0.05)$，而在另一种情况下， $\pi = (0.82, 0.18)$ 被更新到了 $\pi = (0.18, 0.82)$。虽然两者在参数空间中的更新幅度相同，但是在策略空间中的更新幅度却完全不同。
>
> 针对以上问题，我们考虑开发能更好处理步长的策略梯度算法，在更新时找到一块**信任区域**（trust region），在这个区域上更新策略时能够得到某种策略性能的安全性保证，这就得到了**信任区域策略优化**（trust region policy optimization，TRPO）算法。

![img](https://pic2.zhimg.com/v2-e519a12e0617dd0eb66de29db96af429_1440w.jpg)

>  **引理**
>  $$
>  J(\theta') = J(\theta) + \mathbb E_{\tau \sim \pi_\theta'}\left[ \sum_{t\ge 0} \gamma^tA^{\pi_\theta}(S_t, A_t) \right]
>  $$
>  这里 $J(\theta) = \mathbb E_{\tau \sim \pi_\theta}[\sum_{t\ge 0} \gamma^tR(S_t, A_t)]$，$\tau$ 是由 $\pi_\theta'$ 产生的同状态的动作轨迹。

所以，学习最优的策略 $\pi_\theta$ 等价于最大化 $$ \mathbb{E}_{\tau \sim \pi'_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t A^{\pi_\theta}(S_t, A_t) \right]. $$ 而上述式子其实难以直接优化，因为式子中的期望是在 $\pi'_\theta$ 上，所以 TRPO 优化该式子做了一个近似，我们用 $\mathcal{L}_{\pi_\theta}(\pi'_\theta)$ 表示，见如下推导（这里的 $\rho$ 表示概率分布，具体而言，$\rho_{\pi_\theta}(s)$ 表示遵循策略 $\pi_\theta$ 时系统处于状态 $s$ 的概率分布）：

$$
\mathbb{E}_{\tau \sim \pi'_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t A^{\pi_\theta}(S_t, A_t) \right]  &=& \mathbb{E}_{s \sim \rho_{\pi'_\theta}(s)} \left[ \mathbb{E}_{a \sim \pi'_\theta(a|s)} \left[ A^{\pi_\theta}(s, a) | s \right] \right] \\ &\approx& \mathbb{E}_{s \sim \rho_{\pi_\theta}(s)} \left[ \mathbb{E}_{a \sim \pi'_\theta(a|s)} \left[ A^{\pi_\theta}(s, a) | s \right] \right] \\ &=& \mathbb{E}_{s \sim \rho_{\pi_\theta}(s)} \left[ \mathbb{E}_{a \sim \pi_\theta(a|s)} \left[ \frac{\pi'_\theta(a|s)}{\pi_\theta(a|s)} A^{\pi_\theta}(s, a) | s \right] \right] \\ &=& \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t \frac{\pi'_\theta(A_t|S_t)}{\pi_\theta(A_t|S_t)} A^{\pi_\theta}(S_t, A_t) \right].
$$
记 $$ \mathcal{L}_{\pi_\theta}(\pi'_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t \frac{\pi'_\theta(A_t|S_t)}{\pi_\theta(A_t|S_t)} A^{\pi_\theta}(S_t, A_t) \right].$$

虽然直接用 $\rho_{\pi_\theta}(s)$ 来近似 $\rho_{\pi'_\theta}(s)$ 的行为看似粗糙，但下述定理在理论上证明了，当 $\pi_\theta$ 和 $\pi'_\theta$ 相似的时候，这个近似并不差。

>**定理** 让 $D^{\max}_{\mathrm{KL}}(\pi_\theta \| \pi'_\theta) = \max_s D_{\mathrm{KL}}(\pi_\theta(\cdot | s) \| \pi'_\theta(\cdot | s))$，那么
>$$
>|J(\theta') - J(\theta) - \mathcal{L}_{\pi_\theta}(\pi'_\theta)| \leq C D^{\max}_{\mathrm{KL}}(\pi_\theta \| \pi'_\theta).
>$$
>
>这里 $C$ 是和 $\pi'_\theta$ 无关的常数。

因此，如果 $D^{\max}_{\mathrm{KL}}(\pi_\theta \| \pi'_\theta)$ 很小，那么 $\mathcal{L}_{\pi_\theta}(\pi'_\theta)$ 可以合理地被作为一个优化目标。因此得：

$$
\begin{aligned}
&\max_{\pi'_\theta} \quad \mathcal{L}_{\pi_\theta}(\pi'_\theta)  = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t \frac{\pi'_\theta(A_t|S_t)}{\pi_\theta(A_t|S_t)} A^{\pi_\theta}(S_t, A_t) \right] \\
&\text{s.t.} \quad \mathbb{E}_{s \sim \rho_{\pi_\theta}} \left[ D_{\mathrm{KL}}(\pi_\theta \| \pi'_\theta) \right] \leq \delta.
\end{aligned}
$$

---

而 PPO 将上面的带约束的优化问题直接整理成了一个正则化的版本：
$$
\max_{\pi_{\theta'}} & \mathcal L_{\pi_\theta}(\pi_\theta') - \lambda\mathbb E_{s\sim\rho_{\pi_\theta}}[D_{\text{KL}}(\pi_\theta \| \pi_\theta') ]\\
$$
这里 $\lambda$ 是正则化系数，值依赖于 $\pi_\theta$。在 PPO 中，我们通过检验 KL 散度的值来决定 $\lambda$ 的值应该增大还是减小。这个版本的 PPO 算法称为 **PPO-Penalty**。

令 $ d_k = D_{\text{KL}}(\pi_\theta \| \pi_\theta')$，$\beta$ 的更新规则可以是（$\delta$ 对应上述 TRPO 中优化问题的 $\delta$ 值）：

1. 如果 $ d_k < \delta / 1.5 $，那么 $\beta_{k+1} = \beta_k / 2$
1. 如果 $ d_k > \delta \times 1.5 $，那么 $\beta_{k+1} = \beta_k \times 2$
1. 否则 $\beta_{k+1} = \beta_k$

另一个方法是直接剪断用于策略梯度的目标函数：

$$
\mathcal{L}^{\text{PPO-Clip}}(\pi'_\theta) = \mathbb{E}_{\pi_\theta} \left[ \min \left( \frac{\pi'_\theta(A_t | S_t)}{\pi_\theta(A_t | S_t)} A^{\pi_\theta}(S_t, A_t), \text{clip}\left(\frac{\pi'_\theta(A_t | S_t)}{\pi_\theta(A_t | S_t)}, 1 - \epsilon, 1 + \epsilon\right) A^{\pi_\theta}(S_t, A_t) \right) \right].
$$

这里 $\text{clip}(x, 1 - \epsilon, 1 + \epsilon)$ 将 $x$ 截断在 $[1 - \epsilon, 1 + \epsilon]$ 中。这个版本的算法被称为 **PPO-Clip**，它能将 $\frac{\pi'_\theta(A_t | S_t)}{\pi_\theta(A_t | S_t)}$ 截断在 $[1 - \epsilon, 1 + \epsilon]$ 中来保证 $\pi'_\theta$ 和 $\pi_\theta$ 之间的变化不会过大。

<img src="C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250416170122855.png" alt="image-20250416170122855" style="zoom: 67%;" />

![img](https://datawhalechina.github.io/easy-rl/img/ch5/5.3.png)

![https://spinningup.openai.com/en/latest/algorithms/ppo.html](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250416181704908.png)

## 群体相对策略优化（GRPO）

> 强化学习已成为微调LLM以符合人类偏好的基石。在强化学习算法中，PPO 因其稳定性和效率而被广泛采用。然而，随着模型规模不断扩大、任务日益复杂，PPO 的局限性（例如内存开销和计算成本）促使人们开发更先进的方法。

![img](https://paper-assets.alphaxiv.org/figures/2402.03300/x2.png)



- **RL 在 LLM 培训中扮演什么角色？**https://community.aws/content/2rJrpj6m2eh591fjMcRZ3ushpB7/deep-dive-into-group-relative-policy-optimization-grpo

  1. 监督微调（SFT）

     ```
        ┌───────────────┐
        │ High-Quality  │
        │ Human-Labeled │    > (使用高质量数据冷启动训练)
        │     Data      │
        └───────────────┘
                │
                ▼
        ┌───────────────┐
        │ Fine-Tuned LLM│
        │   (π_SFT)     │
        └───────────────┘
     ```

  2. 奖励模型

     ```
        ┌───────────────┐
        │   Pairwise    │
        │  Comparison   │
        │     Data      │
        └───────────────┘
                │
                ▼
        ┌───────────────┐
        │ Reward Model  │
        │     (R)       │
        └───────────────┘
     ```

  3. RL 优化

     ```
        ┌───────────────┐
        │   Generate    │
        │   Responses   │
        └───────────────┘
                │
                ▼
        ┌───────────────┐
        │   Compute     │
        │   Rewards     │
        └───────────────┘
                │
                ▼
        ┌───────────────┐
        │   Optimize    │
        │ Policy (π_θ)  │
        └───────────────┘
     ```

     

- 

![image-20250416181956281](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250416181956281.png)

![image-20250416182019246](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250416182019246.png)

![image-20250416182011305](C:\Users\22597\AppData\Roaming\Typora\typora-user-images\image-20250416182011305.png)

1. **奖励重塑（Reward Reshaping）**

PPO直接使用环境提供的原始奖励函数 $r(s, a)$ 来优化策略。然而，原始奖励往往稀疏或难以引导策略快速学习最优行为。针对于此，GRPO使用一个参数化的函数 $f_\phi(s, a)$ 来动态调整奖励，其中 $\phi$ 是可学习的参数。

重塑后的奖励：原始奖励被改造为$r'(s, a) = r(s, a) + f_\phi(s, a)$，其中$f_\phi(s, a)$根据状态$s$和动作$a$提供额外的奖励信号。

学习过程：GRPO通过最大化策略的累积奖励来优化$f_\phi(s, a)$的参数$\phi$。具体来说，$\phi$可以通过梯度下降更新，确保重塑后的奖励能更好地引导策略趋向最优解。

2. **策略更新的优化**

GRPO在策略更新上引入了更灵活和精确的控制机制：

- 信任区域优化：GRPO延续了TRPO的信任区域思想，通过限制策略更新的步长来确保稳定性。与PPO-Clip的硬性剪切不同，GRPO更注重策略在概率分布上的平滑过渡。

- KL散度约束：GRPO在目标函数中显式加入KL散度惩罚项，优化目标变为：
  $$
  \mathcal L^{\text{GRPO}}(\pi_\theta') = \mathcal L^{\text{PPO-Clip}}(\pi_\theta')-\beta\mathbb E_{\pi\sim\rho_{\pi_\theta}}[D_{\text{KL}}(\pi_\theta \| \pi_\theta')]
  $$
  其中，$\beta$是控制惩罚强度的超参数。

- 自适应调整：GRPO通过动态调整$\beta$来平衡探索和稳定性。例如：

  - 如果$D_{\text{KL}}(\pi_\theta | \pi'_\theta)$过大（策略变化过剧烈），则增大$\beta$以加强约束。
  - 如果$D_{\text{KL}}$过小（更新过于保守），则减小$\beta$以允许更大更新。

3. **优势估计的改进**

PPO使用基于值函数的优势估计$A^{\pi_\theta}(s, a) = Q^{\pi_\theta}(s, a) - V^{\pi_\theta}(s)$，通常通过单步TD误差或简单的多步回报计算。然而，这种方法可能存在较大的方差或偏差，影响策略优化的精度。

GRPO在优势估计上采用了更精确的方法：

- 广义优势估计（GAE）：GRPO使用GAE来计算优势函数：
  $$
  A_{t}^{\text{GAE}(\gamma, \lambda)} = \sum_{l\ge 0} (\gamma\lambda)^l\delta_{t + l}
  $$
  其中，$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$是TD误差，$\lambda$是平衡偏差和方差的参数（通常在0到1之间）。

- 多步TD目标：GRPO进一步结合多步回报估计，通过考虑更长的未来奖励序列来提高优势估计的准确性。

---

## DDPG



---

## SAC



[强化学习进阶 第七讲 TRPO - 知乎](https://zhuanlan.zhihu.com/p/26308073)

https://medium.com/@jonathan_hui/rl-the-math-behind-trpo-ppo-d12f6c745f33







![img](https://pic4.zhimg.com/v2-eba10ba1c76228967df531c5ac287f3f_1440w.jpg)

```

```

