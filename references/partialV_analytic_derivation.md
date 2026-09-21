# HMC 有效势的解析梯度：推导、记号与代码对应

> 代码来源：`Class.py` 中 `MCMCSampler.chi_square`、`MCMCSampler.sample`（`approx_fprime` 调用处）、`Interpolator.predict`
> 论文来源：arXiv:2512.11536（Phys. Rev. C 114, 024912 (2026)），式 (207)–(221)、(446)–(484)、(601)–(608)
>
> 本文档的目的是把 HMC 中 Leapfrog 所需的 \(\partial V/\partial x_i\) 由数值差分改为解析表达式。文档只给出推导、记号与实现对应关系，不包含数值验证。

## 0. 目标

Leapfrog 积分（论文式 601–608）每一步需要势能梯度 \(\partial V/\partial\hat{x}_i\)：

$$
\hat{p}_i\left(t+\frac{\delta_t}{2}\right)=\hat{p}_i(t)-\frac{\delta_t}{2}\frac{\partial V}{\partial \hat{x}_i}(t),
\qquad
\hat{x}_i(t+\delta_t)=\hat{x}_i(t)+\delta_t\frac{\hat{p}_i\left(t+\frac{\delta_t}{2}\right)}{m},
$$

$$
\hat{p}_i(t+\delta_t)=\hat{p}_i\left(t+\frac{\delta_t}{2}\right)-\frac{\delta_t}{2}\frac{\partial V}{\partial \hat{x}_i}(t+\delta_t).
$$

`Class.py` 目前用 `scipy.optimize.approx_fprime`（两点前向差分，步长 \(\varepsilon_i=\) `config.gradient_epsilon`）计算这两处的梯度。若改用解析梯度，得到的必须**与代码中实际使用的 \(V(\boldsymbol{x})\) 严格一致**，否则 Leapfrog 不再保能量，Metropolis 接受率会塌掉。

## 1. 记号总表

### 1.1 参数、数据与似然

| 符号 | 代码对象 | 含义 | 维数 |
| --- | --- | --- | --- |
| \(\boldsymbol{x}\) | `x`（`current_x`、`x_t`） | 被 HMC 采样的参数向量（大坐标） | \(d=\) `N_parameter` |
| \(\boldsymbol{x}_{\mathrm{full}}\) | `x_full = np.append(x, fixed_params)` | GP/PCA 的实际输入（采样参数 + 固定参数） | \(d+d_{\mathrm{fix}}\) |
| \(\boldsymbol{y}_{\mathrm{exp}}\) | `data_loader.experimental_data` | 实验中心值 | \(N=\) `N_data` |
| \(\overline{\boldsymbol{y}}(\boldsymbol{x})\) | `mu_x` | 模拟器（emulator）给出的理论预测 | \(N\) |
| \(\Delta\boldsymbol{y}_{\boldsymbol{x}}\) | `delta_y = mu_x - experimental_data` | 理论与实验之差 | \(N\) |
| \(\Sigma_{\mathrm{exp}}\) | `data_loader.Cov_exp` | 实验协方差（统计 + 系统），与 \(\boldsymbol{x}\) 无关 | \(N\times N\) |
| \(\Sigma_{\mathrm{th}}(\boldsymbol{x})\) | `Cov_x` | 理论（GP 模拟器）协方差，随 \(\boldsymbol{x}\) 变化 | \(N\times N\) |
| \(\Sigma(\boldsymbol{x})\) | `Cov = Cov_x + Cov_exp` | 总协方差 | \(N\times N\) |
| \(T\) | 取 1（隐含） | 有效温度，式 (550)、(558) | 标量 |
| \(m\) | 取 1（隐含） | 大动量对应的质量，式 (554)、(570) | 标量 |
| \(V(\boldsymbol{x})\) | `0.5 * chi_square(x)` | 有效势，式 (550) | 标量 |
| \(f\) | `config.variance_inflation`（`self.variance_inflation`） | 方差膨胀因子，代码特有的似然约定 | 标量 |
| \(\delta_t,\ N_t\) | `config.delta_t`、`config.leapfrog_steps` | Leapfrog 步长、每条轨迹的步数 | 标量 |

### 1.2 PCA 与 GP（模拟器内部）

| 符号 | 代码对象 | 含义 | 维数 |
| --- | --- | --- | --- |
| \(\boldsymbol{U}\) | `pca.components_`（`Unitary`） | PCA 编码矩阵，第 \(m\) 行是第 \(m\) 个主成分方向 | \(n\times N\) |
| \(\boldsymbol{U}^{\dagger}\) | `Unitary.T` | PCA 解码矩阵，第 \(m\) 列是第 \(m\) 个主成分方向 | \(N\times n\) |
| \(\overline{\boldsymbol{y}}\) | `pca.mean_`（`mu`） | 训练集输出的均值 | \(N\) |
| \(P_m(\boldsymbol{x})\) | 第 \(m\) 个 `gp.predict` 的均值 | 第 \(m\) 个主成分系数 | 标量 |
| \(\boldsymbol{P}(\boldsymbol{x})\) | `Y_pred_pca` | 全部主成分系数构成的向量 | \(n=\) `n_pca_components` |
| \(P_q\) | 训练点的第 \(m\) 个主成分系数 | GP 的训练目标（已去均值） | 标量 |
| \(\sigma^2_{P,m}(\boldsymbol{x})\) | 第 \(m\) 个 `gp.predict(..., return_std=True)` 的 std 平方 | 第 \(m\) 个主成分的 GP 预测方差 | 标量 |
| \(\kappa_m(\boldsymbol{x},\boldsymbol{x}')\) | 第 \(m\) 个 GP 的核函数 | 径向基（RBF）核 | 标量 |
| \(K_m\) | `gp.kernel_(gp.X_train_)` | 含白噪声的训练核矩阵，论文式 (459)–(463) | \(n_{\mathrm{train}}\times n_{\mathrm{train}}\) |
| \(l_m\) | `gp.kernel_.k1.k2.length_scale` | 第 \(m\) 个 GP 的长度尺度（特征长度） | 标量 |
| \(C_m\) | `gp.kernel_.k1.k1.constant_value` | 第 \(m\) 个 GP 的核幅度（自相关系数平方） | 标量 |
| \(\sigma^2_{\mathrm{wn},m}\) | `gp.kernel_.k2.noise_level` | 第 \(m\) 个 GP 的白噪声方差 | 标量 |
| \(\boldsymbol{x}_p\) | `gp.X_train_[p]` | 第 \(p\) 个训练参数点（拉丁超立方设计点） | \(d+d_{\mathrm{fix}}\) |

GP 均值与方差采用论文式 (472)–(476) 的形式：

$$
\overline{P}_m(\boldsymbol{x})=\sum_{p,q=1}^{n}\kappa_m(\boldsymbol{x},\boldsymbol{x}_p)\left(K_m^{-1}\right)_{pq}P_q,
$$

$$
\sigma^2_{P,m}(\boldsymbol{x})=\kappa_m(\boldsymbol{x},\boldsymbol{x})-\sum_{p,q=1}^{n}\kappa_m(\boldsymbol{x},\boldsymbol{x}_p)\left(K_m^{-1}\right)_{pq}\kappa_m(\boldsymbol{x}_q,\boldsymbol{x}),
$$

核函数取论文式 (481) 的 RBF 形式（每个主成分独立训练一套超参数）：

$$
\kappa_m(\boldsymbol{x},\boldsymbol{x}')=C_m^2\exp\left(-\frac{\left\|\boldsymbol{x}-\boldsymbol{x}'\right\|_2^2}{2l_m^2}\right),
\qquad
\frac{\partial\kappa_m(\boldsymbol{x},\boldsymbol{x}')}{\partial x_i}=-\frac{(x-x')_i}{l_m^2}\,\kappa_m(\boldsymbol{x},\boldsymbol{x}').
$$

## 2. 推导

### 2.1 似然与有效势

多元正态似然（论文式 442、207）：

$$
\mathcal{P}\left(\boldsymbol{y}_{\mathrm{exp}}\middle|\boldsymbol{y}(\boldsymbol{x})\right)=\frac{\exp\left[-\frac{1}{2}\Delta\boldsymbol{y}_{\boldsymbol{x}}^{T}\Sigma^{-1}(\boldsymbol{x})\Delta\boldsymbol{y}_{\boldsymbol{x}}\right]}{\sqrt{(2\pi)^{d}\det\left[\Sigma(\boldsymbol{x})\right]}},
$$

取对数（手稿中把 \(\ln\det\Sigma(\boldsymbol{x})\) 简写为 \(\ln\Sigma(x)\)）：

$$
\ln\mathcal{P}=-\frac{1}{2}\Delta\boldsymbol{y}_{\boldsymbol{x}}^{T}\Sigma^{-1}(\boldsymbol{x})\Delta\boldsymbol{y}_{\boldsymbol{x}}-\frac{d}{2}\ln(2\pi)-\frac{1}{2}\ln\det\Sigma(\boldsymbol{x}),
$$

有效势（论文式 550）：

$$
V(\boldsymbol{x})=-T\ln\mathcal{P},\qquad \frac{\partial V}{\partial x_i}=-T\frac{\partial\ln\mathcal{P}}{\partial x_i}.
$$

### 2.2 对参数求导

对 \(x_i\) 求导，得到三项，分别来自"残差随 \(\boldsymbol{x}\) 变化"、"\(\Sigma^{-1}\) 随 \(\boldsymbol{x}\) 变化"和"归一化因子随 \(\boldsymbol{x}\) 变化"：

$$
\frac{\partial\ln\mathcal{P}}{\partial x_i}
=-\Delta\boldsymbol{y}^{T}\Sigma^{-1}\frac{\partial\Delta\boldsymbol{y}}{\partial x_i}
+\frac{1}{2}\Delta\boldsymbol{y}^{T}\Sigma^{-1}\frac{\partial\Sigma}{\partial x_i}\Sigma^{-1}\Delta\boldsymbol{y}
-\frac{1}{2}\mathrm{tr}\left[\Sigma^{-1}\frac{\partial\Sigma}{\partial x_i}\right].
$$

（手稿中第二个二次型的写法 \(\Delta\boldsymbol{y}\,\Sigma^{-1}\frac{\partial\Sigma}{\partial x_i}\Sigma^{-1}\Delta\boldsymbol{y}^{T}\) 只是转置记号不严谨，应写成上式；第一项同理，是行向量 \(\Delta\boldsymbol{y}^{T}\) 乘矩阵再乘列向量。常数项 \(-\frac{d}{2}\ln(2\pi)\) 对 \(\boldsymbol{x}\) 的导数为零，已略去。）

于是

$$
\frac{\partial V}{\partial x_i}
=T\left[
\Delta\boldsymbol{y}^{T}\Sigma^{-1}\frac{\partial\Delta\boldsymbol{y}}{\partial x_i}
-\frac{1}{2}\Delta\boldsymbol{y}^{T}\Sigma^{-1}\frac{\partial\Sigma}{\partial x_i}\Sigma^{-1}\Delta\boldsymbol{y}
+\frac{1}{2}\mathrm{tr}\left(\Sigma^{-1}\frac{\partial\Sigma}{\partial x_i}\right)
\right].
$$

### 2.3 残差对参数的导数（PCA 解码）

模型预测由主成分解码得到（论文式 437）：

$$
\boldsymbol{y}(\boldsymbol{x})=\boldsymbol{U}^{\dagger}\boldsymbol{P}(\boldsymbol{x})+\overline{\boldsymbol{y}},
\qquad
\Delta\boldsymbol{y}=\boldsymbol{y}(\boldsymbol{x})-\boldsymbol{y}_{\mathrm{exp}}
=\boldsymbol{U}^{\dagger}\boldsymbol{P}(\boldsymbol{x})+\left(\overline{\boldsymbol{y}}-\boldsymbol{y}_{\mathrm{exp}}\right).
$$

\(\boldsymbol{U}^{\dagger}\) 与 \(\overline{\boldsymbol{y}}\) 都与 \(\boldsymbol{x}\) 无关，故

$$
\frac{\partial\Delta\boldsymbol{y}}{\partial x_i}
=\frac{\partial\left(\boldsymbol{U}^{\dagger}\boldsymbol{P}(\boldsymbol{x})\right)}{\partial x_i}
=\boldsymbol{U}^{\dagger}\frac{\partial\boldsymbol{P}(\boldsymbol{x})}{\partial x_i}.
$$

### 2.4 GP 均值对参数的导数

$$
\frac{\partial \overline{P}_m(\boldsymbol{x})}{\partial x_i}
=\frac{\partial}{\partial x_i}\left[\sum_{p,q=1}^{n_{\mathrm{train}}}\kappa_m(\boldsymbol{x},\boldsymbol{x}_p)\left(K_m^{-1}\right)_{pq}P_q\right]
=-\frac{1}{l_m^{2}}\sum_{p,q=1}^{n_{\mathrm{train}}}\kappa_m(\boldsymbol{x},\boldsymbol{x}_p)\,(x-x_p)_i\left(K_m^{-1}\right)_{pq}P_q .
$$

这里用到 RBF 核的对数导数 \((x-x_p)_i/l_m^2\)，以及 \(K_m^{-1}\)、\(P_q\) 都与 \(\boldsymbol{x}\) 无关。求和中的下标 \(i\) 走遍全部输入分量，但固定参数对应 \(x_i-x_{p,i}\equiv 0\)，因此只有被采样的 \(d\) 个分量贡献非零。

### 2.5 GP 方差对参数的导数

对 \(\sigma^2_{P,m}(\boldsymbol{x})\) 求导。因为 RBF 核对角项 \(\kappa_m(\boldsymbol{x},\boldsymbol{x})=C_m^2\) 与 \(\boldsymbol{x}\) 无关（白噪声项同样是与 \(\boldsymbol{x}\) 无关的常数），求导只剩下 \(\kappa_m^{T}K_m^{-1}\kappa_m\) 这一项，利用 \(K_m^{-1}\) 的对称性得

$$
\frac{\partial\sigma^2_{P,m}(\boldsymbol{x})}{\partial x_i}
=\frac{2}{l_m^{2}}\sum_{p,q=1}^{n_{\mathrm{train}}}(x-x_p)_i\,\kappa_m(\boldsymbol{x},\boldsymbol{x}_p)\left(K_m^{-1}\right)_{pq}\kappa_m(\boldsymbol{x}_q,\boldsymbol{x}).
$$

**还原因子 \(s_m^2\)。** 上式是**核空间**（标准化目标）中的方差导数。`fit` 里 `GaussianProcessRegressor(normalize_y=True)` 会把目标线性变换为 \((P-\overline{P})/s_m\) 后再拟合，`predict(..., return_std=True)` 返回的方差是**还原到物理单位**后的量（sklearn `_gpr.py` 第 488 行 `y_var *= _y_train_std**2`），即

$$
\sigma^2_{P,m}\big|_{\texttt{predict}}=s_m^2\Big[\kappa_m(\boldsymbol{x},\boldsymbol{x})-\kappa_m^{T}K_m^{-1}\kappa_m\Big],\qquad s_m=\texttt{gp.\_y\_train\_std}.
$$

`Interpolator.predict` 的 \(\Sigma_{\mathrm{th}}\) 正是由 `predict` 的 std 构造的，所以解析梯度必须带上同一个 \(s_m^2\)（代码：`predict_with_grad` 中的 `dvar_dx`）。这不是额外的物理缩放，只是与所采样势能保持逐点一致；若把 `fit` 改成 `normalize_y=False`，\(s_m\equiv1\)，该因子自动为 1。

### 2.6 协方差对参数的导数

理论协方差来自 PCA 空间的对角协方差再解码（论文式 450）：

$$
\Sigma_{\mathrm{Gauss}}(\boldsymbol{x})=\boldsymbol{U}^{\dagger}\,\mathrm{diag}\left(\sigma^2_{P,1}(\boldsymbol{x}),\sigma^2_{P,2}(\boldsymbol{x}),\cdots\right)\boldsymbol{U},
\qquad
\Sigma_{\mathrm{th}}(\boldsymbol{x})=2\,\Sigma_{\mathrm{Gauss}}(\boldsymbol{x}).
$$

由于 \(\boldsymbol{U}\)、\(\boldsymbol{U}^{\dagger}\) 与 \(\boldsymbol{x}\) 无关，而 \(\Sigma_{\mathrm{exp}}\) 也与 \(\boldsymbol{x}\) 无关，

$$
\frac{\partial\Sigma(\boldsymbol{x})}{\partial x_i}
=\frac{\partial\Sigma_{\mathrm{th}}(\boldsymbol{x})}{\partial x_i}
=2\,\boldsymbol{U}^{\dagger}\,\mathrm{diag}\left(\frac{\partial\sigma^2_{P,1}}{\partial x_i}(\boldsymbol{x}),\frac{\partial\sigma^2_{P,2}}{\partial x_i}(\boldsymbol{x}),\cdots\right)\boldsymbol{U}.
$$

### 2.7 合并结果

把 2.3–2.6 代回 2.2，即得 HMC 需要的解析梯度：

$$
\frac{\partial V}{\partial x_i}
=T\left[
\Delta\boldsymbol{y}^{T}\Sigma^{-1}\boldsymbol{U}^{\dagger}\frac{\partial\boldsymbol{P}}{\partial x_i}
-\frac{1}{2}\Delta\boldsymbol{y}^{T}\Sigma^{-1}\frac{\partial\Sigma}{\partial x_i}\Sigma^{-1}\Delta\boldsymbol{y}
+\frac{1}{2}\mathrm{tr}\left(\Sigma^{-1}\frac{\partial\Sigma}{\partial x_i}\right)
\right],
$$

其中 \(\partial\boldsymbol{P}/\partial x_i\)、\(\partial\sigma^2_{P,m}/\partial x_i\)、\(\partial\Sigma/\partial x_i\) 分别由 2.4、2.5、2.6 给出。代码中 \(T=1\)。

## 3. 与 `Class.py` 的对应（实现要点）

### 3.1 代码里的似然带有方差膨胀因子 \(f\)

`chi_square` 返回的量是

$$
\chi^2(\boldsymbol{x})=\frac{1}{f}\Delta\boldsymbol{y}^{T}\Sigma^{-1}\Delta\boldsymbol{y}+\ln\det\Sigma+d\ln f,
$$

而 Leapfrog 使用的势能是

$$
V(\boldsymbol{x})=\frac{1}{2}\chi^2(\boldsymbol{x})=\frac{1}{2}\left[\frac{1}{f}\Delta\boldsymbol{y}^{T}\Sigma^{-1}\Delta\boldsymbol{y}+\ln\det\Sigma+d\ln f\right].
$$

注意 \(1/f\) **只乘在残差二次型上**，不乘 \(\ln\det\Sigma\)（这正是代码注释强调的"精确膨胀"：\(\Sigma\to f\Sigma\) 给出的就是 \(\chi^2_1/f+\ln\det\Sigma+d\ln f\)）。因此解析梯度应为

$$
\frac{\partial V}{\partial x_i}
=\frac{1}{f}\left[
\Delta\boldsymbol{y}^{T}\Sigma^{-1}\frac{\partial\Delta\boldsymbol{y}}{\partial x_i}
-\frac{1}{2}\Delta\boldsymbol{y}^{T}\Sigma^{-1}\frac{\partial\Sigma}{\partial x_i}\Sigma^{-1}\Delta\boldsymbol{y}
\right]
+\frac{1}{2}\mathrm{tr}\left(\Sigma^{-1}\frac{\partial\Sigma}{\partial x_i}\right).
$$

若把上式整体除以 \(f\)，就会与代码中的 \(\chi^2\) 不一致（等价于错误地压平了 \(\ln\det\Sigma\) 的 \(\boldsymbol{x}\) 依赖）。

### 3.2 两种协方差模式

* **`analytical=True`（当前的 `Input.py`）**：`Interpolator.predict` 返回 \(\Sigma_{\mathrm{th}}=\mathbf{0}\)，故 \(\Sigma=\Sigma_{\mathrm{exp}}\) 与 \(\boldsymbol{x}\) 无关，\(\partial\Sigma/\partial x_i=0\)，后两项消失：

$$
\frac{\partial V}{\partial x_i}=\frac{1}{f}\Delta\boldsymbol{y}^{T}\Sigma_{\mathrm{exp}}^{-1}\frac{\partial\boldsymbol{y}(\boldsymbol{x})}{\partial x_i},
$$

此时 \(\partial\boldsymbol{y}/\partial x_i\) 就是解析模型 `config.analytical_model` 的雅可比。

* **`analytical=False`（PCA + GP）**：用 2.7 / 3.1 的完整表达式。

### 3.3 建议预计算

核函数与训练点固定，因此以下量与 \(\boldsymbol{x}\) 无关，可在训练后或采样前算一次：

* 各主成分的 \(l_m,\ C_m,\ \sigma^2_{\mathrm{wn},m}\)：`gp.kernel_.k1.k2.length_scale`、`gp.kernel_.k1.k1.constant_value`、`gp.kernel_.k2.noise_level`；
* \(K_m\) 或它的 Cholesky 分解（代码用 `gp.kernel_(gp.X_train_) + gp.alpha*I` 的逆，与 sklearn 内部 `cho_solve(cho_factor(L_))` 一致），以及 §2.4 内层对 \(q\) 的求和
  $$
  \alpha_m \equiv K_m^{-1}\left(P_q-\overline{P}_m\right),
  $$
  它把 \(n_{\mathrm{train}}\) 个训练点压缩成一个权重向量：预测值相对训练均值的偏差正好是 \(\overline{P}_m(\boldsymbol{x})-\overline{P}_m=\kappa_m(\boldsymbol{x})^{T}\alpha_m\)，即"以训练点为基、以 \(\alpha_m\) 为权重"的插值；
* 训练点矩阵 `gp.X_train_` 与主成分系数 `Y_pca`（`pca.transform(y_data)`）。

关于 `normalize_y=True`：**平移与缩放要分开看。** 平移 \(\overline{P}_m\) 必须显式减掉（否则 \(K_m^{-1}\) 会作用在一个非零常数向量上，梯度多出一项）；缩放 \(s_m\) 对**均值**项自动抵消（因为 \(P_q\) 只出现在 \(\alpha_m\) 里，而求导只作用在 \(\kappa_m\) 上），但对**方差**项不抵消——见 §2.5 的 \(s_m^2\)。
代码实现上等价的两条路：直接用 sklearn 现成的对偶系数 `gp.alpha_ * gp._y_train_std`（已知 $=K_m^{-1}(P_q-\overline{P}_m)$），或按本文档显式构造 `K_inv @ (P_train - y_mean)`（`predict_with_grad` 采用后者，便于与 §2.4 逐行对照）。

此外，两个二次型可以只通过 \(n\) 维向量计算，避免显式构造 \(N\times N\) 的 \(\partial\Sigma/\partial x_i\)（令 \(\boldsymbol{w}=\boldsymbol{U}\Sigma^{-1}\Delta\boldsymbol{y}\)、\(\boldsymbol{G}=\boldsymbol{U}\Sigma^{-1}\boldsymbol{U}^{\dagger}\)，两者都是 \(n\) 维/\(n\times n\) 量）：

$$
\Delta\boldsymbol{y}^{T}\Sigma^{-1}\frac{\partial\Sigma}{\partial x_i}\Sigma^{-1}\Delta\boldsymbol{y}
=\sum_{m=1}^{n}\frac{\partial\sigma^2_{P,m}}{\partial x_i}\,w_m^2,
\qquad
\mathrm{tr}\left(\Sigma^{-1}\frac{\partial\Sigma}{\partial x_i}\right)
=\sum_{m=1}^{n}\frac{\partial\sigma^2_{P,m}}{\partial x_i}\,G_{mm}.
$$

配合 2.4、2.5 中"\(\alpha_m=K_m^{-1}P\)、\(\beta_m=K_m^{-1}\kappa_m(\boldsymbol{x})\)"的预分解形式，每个梯度分量只需对训练点做一次线性求和，取代原先每个梯度方向一次（乃至 \(d+1\) 次）完整模拟器求值。



## 4. 校验结论

手稿的推导在结构上是正确的：由多元正态似然 \(\to\) 有效势 \(\to\) 三项求导 \(\to\) PCA 解码 \(\to\) GP 均值/方差导数 \(\to\) 协方差导数，与论文式 (207)–(221)、(446)–(484)、(601)–(608) 以及 `Class.py` 的实现框架一致。需要修正/留意的有三点：

1. **理论协方差的因子 2 已按论文约定写入代码（2026-09-20 修改）。** 手稿中 \(\partial\Sigma/\partial x_i\) 前面的因子 2 来自论文式 (448) 的约定 \(\Sigma_{\mathrm{th}}=2\Sigma_{\mathrm{Gauss}}\)；此前 `Class.py` 的 `Interpolator.predict` 返回的是 `Unitary.T @ diag(std**2) @ Unitary`，即 \(\Sigma_{\mathrm{th}}=\Sigma_{\mathrm{Gauss}}\)，**没有这个因子 2**（参考实现同样如此），两者不一致。现已在 `Interpolator.predict` 中改为 `2.0 * (Unitary.T @ Y_cov_pca @ Unitary)`，因此 2.6 / 2.7 / 3.1 节中的因子 2 与代码一致，解析梯度实现时必须保留它。
2. **转置记号。** 论文中 \(\boldsymbol{U}\) 为 \(n\times N\) 的编码矩阵（`pca.components_`），\(\boldsymbol{U}^{\dagger}=\boldsymbol{U}^{T}\) 才是 \(N\times n\) 的解码矩阵。手稿中 \(\partial\Sigma/\partial x_i=2\boldsymbol{U}\,\mathrm{diag}(\cdots)\,\boldsymbol{U}\) 的写法在维数上不成立，应写成 \(2\boldsymbol{U}^{\dagger}\mathrm{diag}(\cdots)\boldsymbol{U}\)（代码里就是 `Unitary.T @ diag(...) @ Unitary`）。残差那一处的 \(\boldsymbol{U}^{+}\) 即 \(\boldsymbol{U}^{\dagger}\)，与论文式 (437) 一致，没有问题。另外手稿中第二个二次型与第一项的转置记号不严谨，正确形式见 2.2 节。
3. **常数项不影响梯度。** \(\overline{\boldsymbol{y}}-\boldsymbol{y}_{\mathrm{exp}}\)、\(\sigma^2_{P,m}\) 中的白噪声底、\(\kappa_m(\boldsymbol{x},\boldsymbol{x})=C_m^2\)、归一化因子 \(-\frac{d}{2}\ln(2\pi)\) 都与 \(\boldsymbol{x}\) 无关（或对 \(x_i\) 的导数为零），手稿略去它们是合理的；实现时不需要额外补回。

除上述记号与因子约定外，未发现推导性错误。

## 5. 代码落地（2026-09-21）

只有 `analytical=False`（PCA+GP）分支使用本文的解析梯度；`analytical=True` 时模拟器本身就是显式函数、没有 PCA+GP 中间量，Leapfrog 仍调用 `scipy.optimize.approx_fprime`（`MCMCSampler._potential_gradient` 负责分流）。

| 环节 | 位置 |
| --- | --- |
| 与 $\boldsymbol{x}$ 无关的量（$K_m^{-1}$、$\alpha_m$、$l_m$、$s_m$、`X_train_`） | `Interpolator._ensure_gradient_cache`，首次调用时构建并缓存 |
| §2.3–2.6 的 $\partial\boldsymbol{P}/\partial x_i$、$\partial\sigma^2_{P,m}/\partial x_i$、$\partial\boldsymbol{y}/\partial x_i$ | `Interpolator.predict_with_grad` |
| §2.6 / 式 (450) 的 $\Sigma_{\mathrm{th}}=2U^{\dagger}\mathrm{diag}(\sigma^2)U$ | `Interpolator.sigma_th_from_var`（`predict` 与解析梯度共用，因子 2 只有这一处） |
| §2.7 / §3.1 的三项合并、$1/f$ 的位置 | `MCMCSampler.gradient` |
| Leapfrog 的两次梯度调用 | `MCMCSampler.sample`（`grad_xt`、`grad_xt_plus_delta`），反射与动量反号逻辑不变 |

实现约定：

* `predict_with_grad` 的**数值**取自 `gp.predict`，与 `chi_square` 里的势能逐位一致；只有导数是新算的，因此势能与梯度不会互相打架。
* 导数对 $x$ 的**全部**分量（采样参数 + 固定参数）给出，调用方按 `range(N_parameter)` 取前 $d$ 个分量，因此不依赖"训练集中固定参数恒定"这一假设（本数据集恰好满足：末列恒为 0.18）。
* 若把 $n_{\mathrm{train}}$ 做大、$N$ 变大，§3.3 的 $\boldsymbol{w}=\boldsymbol{U}\Sigma^{-1}\Delta\boldsymbol{y}$、$\boldsymbol{G}=\boldsymbol{U}\Sigma^{-1}\boldsymbol{U}^{\dagger}$ 形式可以把 $\partial\Sigma/\partial x_i$ 的 $N\times N$ 构造完全省掉；当前实现按 §2.6/§2.7 显式构造，便于逐行核对（$N=57$、$d=5$ 时开销远小于 GP 前向）。
