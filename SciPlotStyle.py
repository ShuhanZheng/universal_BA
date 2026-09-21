import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# 全局样式设置（仅基础样式，不控制边距，避免冲突）
plt.rcParams.update({
    # 绘图大小
    'figure.figsize': (6.2, 6),

    # 字体大小
    'font.size': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 18,
    
    # 刻度样式（朝内）
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    
    # 新增：调整刻度长度（单位为点，值越大长度越长）
    'xtick.major.size': 12,  # x轴主刻度长度
    'ytick.major.size': 12,  # y轴主刻度长度
    'xtick.minor.size': 4,  # x轴次刻度长度
    'ytick.minor.size': 4,  # y轴次刻度长度
    
    # 线条与网格
    'lines.linewidth': 4,
    'axes.grid': False,
    
    # 新增字体设置
    'font.family': 'serif',  # 声明字体家族为衬线
    'font.serif': ['Times New Roman'],  # 优先使用Times New Roman
    
    # 关键修改：关闭LaTeX渲染，改用内置mathtext
    'text.usetex': False,  # 关闭LaTeX依赖
    # 配置mathtext的字体（与全局Times New Roman匹配）
    'mathtext.fontset': 'stix',  # stix字体集与Times兼容
    'mathtext.rm': 'Times New Roman',  # 数学文本中的罗马体用Times
})

def set_plot_style(ax=None, make_square=True, reduce_margins=True):
    """
    确保：
    1. 刻度线包围的区域是严格物理正方形（像素级对齐）
    2. 减小上下留白，且兼容 plt.tight_layout()
    3. 不干扰用户设置的 xlim/ylim
    """
    if ax is None:
        ax = plt.gca()
    fig = ax.figure  # 获取当前画布

    # 1. 基础刻度设置（上下左右+次刻度）
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # 2. 处理边距（兼容 tight_layout）
    if reduce_margins:
        # 先让 tight_layout 处理标签不重叠，再压缩边距
        plt.tight_layout(pad=0.3)  # pad 调至最小，减少基础留白
        # 手动压缩上下边距（在 tight_layout 之后生效）
        plt.subplots_adjust(
            top=0.98,    # 顶部极窄留白
            bottom=0.06, # 底部极窄留白
            left=0.15,    # 左侧保留标签空间
            right=0.95    # 右侧保留标签空间
        )

    # 3. 核心：强制绘图区域为严格正方形（物理尺寸）
    if make_square:
        # 获取当前画布尺寸（英寸）
        fig_w, fig_h = fig.get_size_inches()
        # 获取当前绘图区位置（相对于画布的比例）
        bbox = ax.get_position()
        ax_left, ax_bottom, ax_w_ratio, ax_h_ratio = bbox.bounds

        # 计算绘图区实际物理尺寸（英寸）
        ax_w_actual = ax_w_ratio * fig_w  # 当前宽度
        ax_h_actual = ax_h_ratio * fig_h  # 当前高度

        # 取最小尺寸作为正方形边长，确保严格等宽高
        square_size = min(ax_w_actual, ax_h_actual)
        # 计算新的比例（相对于画布）
        new_w_ratio = square_size / fig_w
        new_h_ratio = square_size / fig_h

        # 居中对齐，重新设置绘图区位置（确保严格正方形）
        ax.set_position([
            ax_left + (ax_w_ratio - new_w_ratio)/2,  # 水平居中
            ax_bottom + (ax_h_ratio - new_h_ratio)/2, # 垂直居中
            new_w_ratio,  # 宽度=正方形边长
            new_h_ratio   # 高度=正方形边长
        ])

    return ax