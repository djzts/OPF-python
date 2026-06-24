# 3-Bus QHD/QUBO 规模说明

## 结论

当前 3-bus QCE 配置使用：

- 连续优化变量数：31
- Unary encoding resolution：25
- Ising spins / logical binary variables：

$$
N_{\mathrm{binary}}=31\times25=775
$$

因此，QHD/SimBi 实际求解的是一个 **775-spin Ising 模型**。它所对应的 QUBO 具有 **775 个二进制变量**，等价 QUBO 矩阵的维度为：

$$
\boxed{775\times775}
$$

## 计算公式

该 ACOPF 模型包含以下连续变量：

| 变量 | 数量 |
|---|---:|
| $P_G$ | $N_G$ |
| $Q_G$ | $N_G$ |
| $V_R$ | $N_B$ |
| $V_I$ | $N_B$ |
| $V_{sq}$ | $N_B$ |
| $P_{ij}$ | $N_A$ |
| $Q_{ij}$ | $N_A$ |
| $S_{ij,\mathrm{tot}}^2$ | $N_A$ |

每条物理线路对应两个有向 arc，因此：

$$
N_A=2N_L
$$

连续变量总数为：

$$
\begin{aligned}
d
&=2N_G+3N_B+3N_A \\
&=2N_G+3N_B+6N_L
\end{aligned}
$$

使用 unary encoding，每个连续变量由 $R$ 个 spin/binary variables 表示，所以：

$$
\boxed{N_{\mathrm{binary}}=R\left(2N_G+3N_B+6N_L\right)}
$$

对于当前 3-bus 系统：

$$
N_G=2,\qquad N_B=3,\qquad N_L=3,\qquad R=25
$$

于是：

$$
d=2(2)+3(3)+6(3)=31
$$

$$
N_{\mathrm{binary}}=25(31)=775
$$

## QHD 实际构造的模型

在当前程序中，QHD 接收完整的 31 维 `variable_list`，并设置：

```text
dimension = len(variable_list) = 31
resolution = 25
number of spins = dimension × resolution = 775
```

SimBi 后端首先将问题编译为 Ising 形式：

$$
H(s)=\sum_i h_i s_i+\sum_{i<j}J_{ij}s_i s_j,
\qquad s_i\in\{-1,+1\}
$$

其中：

- `h` 的理论长度为 775；
- `J` 使用稀疏字典保存非零耦合项；
- 求解前，程序将 `J` 转换成 $775\times775$ 的稠密对称矩阵 `Jdense`。

通过 $s_i=2x_i-1$（或相反的符号约定），该 Ising 模型可以转换为等价 QUBO：

$$
E(x)=x^TQx+\text{constant},
\qquad x_i\in\{0,1\}
$$

所以，“775”是当前问题的 **logical spin/binary-variable count**；“$775\times775$”是其等价 QUBO 矩阵或稠密 Ising coupling matrix 的维度。

## 系数数量

一个对称的 $775\times775$ QUBO 矩阵最多具有以下数量的独立系数（包含对角线）：

$$
\frac{775(775+1)}{2}=300700
$$

这只是理论上限，并不表示当前模型实际存在 300700 个非零系数。实际非零 coupler 数量由展开后的 LALM 二次多项式及 unary penalty 的稀疏结构决定，通常少于该上限。

## Beam search 对 QUBO 大小的影响

当前 `N=10` 的多点 beam search 不会把单个 QUBO 扩大为 7750 个变量。每个 linearization point 产生一个独立的 775-variable Ising/QUBO 子问题。

因此：

- 单个子问题大小仍为 775 个 binary variables；
- 每轮可能求解 10 个同等规模的子问题；
- 计算工作量随子问题数量增加，但单个矩阵维度不变。

## Logical 与 physical qubits

775 表示逻辑变量或 logical spins 的数量：

- 使用本地 SimBi 时，不涉及量子硬件的 minor embedding；
- 如果改用 D-Wave，受硬件连接拓扑和 chain embedding 影响，实际需要的 physical qubits 可能大于 775；
- physical qubit 数量必须在完成具体硬件 embedding 后才能确定。

## 对应代码位置

- QCE resolution 设置：`Sympy_OPF_LALM_mu_final_3bus_QCE.py:49`
- ACOPF 变量构造：`Sympy_OPF_LALM_class.py:476`
- QHD 问题维数：`qhdopt/qhd_base.py:36`
- spin 数量构造：`qhdopt/backend/backend.py:20`
- SimBi 稠密 coupling matrix 构造：`qhdopt/backend/simbi_backend.py:216`
