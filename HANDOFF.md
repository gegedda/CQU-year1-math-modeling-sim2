# 项目交接手册

> 最后更新：2026-07-21 Day 1 晚间（追加任务已完成）
> 当前状态：✅ 全部代码可运行 / 全部图表已生成 / 模型文档已完稿 / 论文素材包就绪

---

## 一、项目地图（先看这张表）

| 你要找什么 | 在哪里 |
|------------|--------|
| 三个问题的数学模型（符号表+假设+定理+伪代码） | `src/problem1/model.md`、`problem2/`、`problem3/` |
| 所有结果数据表（CSV） | `output/tables/`（19 个文件，含 3-opt/HK/双向取料新增） |
| 所有分析图（PNG） | `output/figures/`（33+ 个文件） |
| 可运行的求解代码 | `src/problem1/solve_all.py`、`src/problem2/hierarchical_solver.py`、`src/problem3/pick_place_single.py` 等 |
| 论文工作区 | `论文/`（含框架模板、参考文献格式、精读笔记） |
| 关键数字速查表（论文手直接引用） | `output/key_numbers.md` |
| 论文框架与写作指导 | `论文/论文框架模板.md` |
| AI 使用记录（论文必附） | `AI工具使用记录/AI工具使用详情.md` |

---

## 二、如何运行（如果队友不在了，按这个来）

### 环境要求
- Python：`D:\Anaconda\python.exe`
- 依赖：`numpy`, `scipy`, `matplotlib`, `pandas`（Anaconda 自带，无需额外安装）

### 一键验证所有脚本

在终端中依次运行（工作目录 `D:\数模\B题`）：

```powershell
# 问题1 基础求解（NN + 2-opt + SA，约 2 分钟）
& "D:\Anaconda\python.exe" "D:\数模\B题\src\problem1\solve_all.py"

# 问题1 SA 参数敏感性（约 30 秒）
& "D:\Anaconda\python.exe" "D:\数模\B题\src\problem1\sensitivity_analysis.py"

# 问题1 四算法对比（约 3 分钟）
& "D:\Anaconda\python.exe" "D:\数模\B题\src\problem1\algorithm_comparison.py"

# 问题2 分层 TSP（约 30 秒）
& "D:\Anaconda\python.exe" "D:\数模\B题\src\problem2\hierarchical_solver.py"

# 问题2 换刀敏感性（约 30 秒）
& "D:\Anaconda\python.exe" "D:\数模\B题\src\problem2\sensitivity_analysis.py"

# 问题3 槽位分配（约 5 秒）
& "D:\Anaconda\python.exe" "D:\数模\B题\src\problem3\slot_assignment.py"

# 问题3 单件取放（约 5 秒）
& "D:\Anaconda\python.exe" "D:\数模\B题\src\problem3\pick_place_single.py"

# 问题3 预拾取（约 5 秒）
& "D:\Anaconda\python.exe" "D:\数模\B题\src\problem3\pick_place_double.py"

# 问题3 配对策略对比（约 5 秒）
& "D:\Anaconda\python.exe" "D:\数模\B题\src\problem3\pairing_comparison.py"
```

**如果某脚本报错：** 把完整错误信息复制到群里，AI 会立即修。

---

## 三、给郭同学 — Day 2 精确操作清单

> 你今天独立推进。**高优先级技术改进已全部在 Day 1 晚间完成**，你的任务主要是验证、补图、打包素材。

### Step 1：环境确认 + 新增脚本验证（09:00–10:00）

- [ ] 打开终端，`cd D:\数模\B题`
- [ ] 运行 `src\problem1\advanced_optimizations.py`，确认生成 3 个新 CSV
- [ ] 运行 `src\problem3\bidirectional_pick.py`，确认双向取料结果 22,842 mm
- [ ] 打开 `output\tables\p1_results.csv`，确认 12 行数据（4 规模 × 3 算法）
- [ ] **检查标准：** n=1173 的 SA 结果 ≈ 13278，2-opt ≈ 12664

### Step 2：问题1 深度验证（10:00–11:30）

- [ ] 打开 `output\tables\p1_3opt_check.csv`，确认 3-opt 改进 0.00%
- [ ] 打开 `output\tables\p1_hk_bound.csv`，确认 Held-Karp LB = 1798.86
- [ ] 打开 `output\tables\p1_mvodm_gi.csv`，确认 GI 结果
- [ ] **思考题**（论文加分点）：3-opt 零改进意味着什么？如何在论文中正面表述？

### Step 3：问题2 深度验证（11:30–12:30）

- [ ] 打开 `output\tables\p2_results.csv`，确认 4 行数据
- [ ] 检查时间的合理性：n=50 → 62.5s，n=198 → 133.4s，n=442 → 309.1s，n=1173 → 491.2s
- [ ] 打开 `output\figures\p2_time_breakdown.png`，确认柱状图合理
- [ ] 打开 `output\tables\p2_sensitivity_changetime.csv`
- [ ] **检查标准：** 换刀时间=5s 的那一行的 Total_Time_s 与 p2_results.csv 一致

### Step 4：问题3 双向取料验证（13:30–14:30）

- [ ] 打开 `output\tables\p3_bidirectional_comparison.csv`
- [ ] 确认三种策略对比：单件 36,324 / 预拾取 23,688 / 双向 22,842
- [ ] **思考题**（论文加分点）：双向取料为什么能消灭所有单件任务？跨槽配对的物理可行性如何论证？

### Step 5：全量可视化补全（14:30–16:00）

以下图如果已有则跳过，如果没有则让 AI 生成：

- [ ] 问题1：四个规模的最优路径图并排对比（4 in 1 子图）
- [ ] 问题2：换刀时间占比随 n 的变化曲线
- [ ] 问题3：三种策略对比图（单件 / 预拾取 / 双向取料）
- [ ] 问题3：预拾取 vs 单件取放的路径长度对比（细分到每个 feeder 的贡献）

### Step 6：AI 记录 + 论文素材包（16:00–17:00）

- [ ] 如果有新的 AI 交互，追加到 `AI工具使用记录\AI工具使用详情.md`
- [ ] **打包论文素材**（17:00 前发到群里给论文手）：
  - [ ] `论文/论文框架模板.md`（论文手写作指导）
  - [ ] `output/key_numbers.md`（关键数字速查表）
  - [ ] `src/*/model.md` 三份（数学模型）
  - [ ] `output\tables\` 全部 CSV
  - [ ] `output\figures\` 全部 PNG
  - [ ] `论文/参考文献格式.md`

---

## 四、给论文手 — 素材索引

### 你要写的核心章节，素材在这里：

| 论文章节 | 素材来源 |
|----------|----------|
| 符号说明 | 三个 `src/*/model.md` 的「符号说明」节（三份合起来即完整符号表） |
| 模型假设 | 三个 `src/*/model.md` 的「模型假设与论证」节（每条有论证，直接改写） |
| 问题1 模型 | `src/problem1/model.md` §三（决策变量+目标函数+DFJ约束） |
| 问题1 求解 | 同上 §四（NN/2-opt/SA 三种算法伪代码+参数选择表） |
| 问题1 结果 | `output/tables/p1_results.csv` + `p1_full_comparison.csv` + `p1_optimality_gap.csv` |
| 问题2 模型 | `src/problem2/model.md` §三（时间三分量+定理1顺序无关性） |
| 问题2 结果 | `output/tables/p2_results.csv` + `p2_sensitivity_changetime.csv` |
| 问题3 模型 | `src/problem3/model.md` §三-四（X排序分配定理+预拾取配对模型） |
| 问题3 结果 | `output/tables/p3_comparison.csv` + `p3_pairing_comparison.csv` |
| 敏感性分析 | `p1_sensitivity_*.csv`（3个）+ `p2_sensitivity_changetime.csv` |
| 模型评价 | 各 model.md 文末，以及本文「已知局限」 |
| AI 使用声明 | `AI工具使用记录/AI工具使用详情.md`（直接作为附录） |

### 关键数字速查（可以直接引用）

| 问题 | 关键结果 |
|------|----------|
| 1 | n=50 最优性差距仅 5.1%（vs LKH 精确解 1,834 mm） |
| 1 | **Held-Karp 下界** 1,798.86 mm，与 LKH 差仅 1.90% — 双重下界更严密 |
| 1 | **3-opt 验证**：改进 0.00% — 2-opt 已收敛至 3-optimal（正面证据） |
| 1 | MVODM 预处理：NN 改进 13.1%（n=50）、5.5%（n=1173） |
| 1 | Kirkpatrick 原 SA 参数（α=0.9）零改进，本文优化 SA 稳定 ~14% |
| 1 | n=1173 最优路径 12,664 mm（2-opt / SA-2opt），耗时 126.6 s |
| 2 | n=1173 总完工时间 491.2 s（钻孔 51% + 移动 46% + 换刀 3%） |
| 2 | 空间聚类失败：路径增加 30.7%，但计算时间↓98%（13.3s→0.23s） |
| 3(1) | 单件取放总距离 36,324 mm |
| 3(2) | 预拾取总距离 23,688 mm（**↓ 34.8%**） |
| 3(3) | **双向取料** 22,842 mm（比预拾取再省 **3.57%**，消灭所有单件任务） |
| 槽位 | X-sort 优化后 avg｜Δx｜从 89.5→17.4 mm（↓80.6%） |
| SA参数 | α 必须 ≥ 0.999，T₀ 影响很小 |
| 换刀敏感性 | 换刀时间从 5s→50s，n=1173 时总时间仅增加 27% |

---

## 五、已知局限（诚实写进论文）

1. SA-from-NN 在大规模（n>500）上不如确定性 2-opt——随机邻域探索效率随 n 增长急剧下降。但**SA-2opt 混合策略**（从 2-opt 解出发退火）可消除退化，保持 2-opt 质量。SA 无法从 2-opt 的强局部最优进一步逃离，**3-opt 验证（改进 0.00%）证实该解已收敛至 3-optimal**，可能接近全局最优。
2. 问题3 的槽位分配和路径规划是两阶段解耦的，未做联合优化（Ho & Ji[9]的联合优化框架可作为改进方向）
3. 拾取/贴装的机械动作时间被忽略（但量级远小于空移时间）
4. ~~预拾取配对假设仅限同槽位内配对~~ → **已解决**：双向取料策略实现了跨槽位配对，总距离再降 3.57%
5. 所有距离计算假设无障碍物和干涉，实际 PCB 可能有禁区
6. 空间聚类策略（Yang[7]）在本文数据集上失败（路径 +30.7%），因 PCB 点位分布均匀、不存在模块化阵列特征——说明该方法非普适
7. Kirkpatrick 1983 原始 SA 参数在本文实例上零改进，揭示了默认参数的危险性
8. ~~未计算 Held-Karp 下界~~ → **已解决**：Held-Karp LB = 1,798.86 mm，与 LKH 最优(1,833.77)差仅 1.90%，双重下界已收窄 gap 区间至 [5.1%, 7.1%]

## 六、论文讨论建议（来自 11 篇精读文献）

| 论文位置 | 讨论点 | 参考文献 |
|----------|--------|----------|
| 问题1 2-opt 段落 | 2-opt 几何本质：消除路径交叉，加 before/after 对比图 | Croes 1958 |
| 问题1 SA 段落 | Metropolis 接受准则形式化 + 统计力学类比 | Kirkpatrick 1983 |
| 模型评价 | 禁忌搜索 vs SA 取舍——SA 更简单但缺记忆机制 | Kolahan 1996 |
| 算法选择说明 | 布谷鸟搜索在 3 实例上匹配/优于 GA/SA/ACO——引证 SA 是合理折中 | Lim 2014 |
| 问题3 模型评价 | 联合优化（Ho 同时分配 feeder + 排序）vs 顺序优化 trade-off | Ho & Ji 2004 |
| 问题3 最优性讨论 | 9.93% MIP gap 类比——我们的 gap 是多少？ | Lu 2023 |
| 引言/背景 | 8-10h→10min 引出计算效率的工业意义 | Lin & Lin 2021 |
| 问题3 备选策略 | 双向取料 vs 近侧取料的结构性差异 | 彭乾伟 2022 |

---

## 六、提交前检查清单

- [ ] 论文 PDF 文件名：`B<队号><队员姓名>.pdf`
- [ ] 论文含：标题、摘要、关键词、问题重述、模型假设、符号说明、建模求解、结果分析、模型评价、参考文献、附录
- [ ] 参考文献含 AI 工具引用（格式见 README.md）
- [ ] 正文中 AI 辅助部分已标注
- [ ] 支撑材料含：所有 `.py` 源文件 + AI工具使用详情.pdf
- [ ] 支撑材料压缩包名：`B<队号><队员姓名>+支撑材料.rar`
- [ ] 邮件主题：`B<队号><队员姓名>`
- [ ] 邮件附件：论文 PDF + 支撑材料压缩包（两个文件，无超大附件）
- [ ] 7 月 28 日 23:55 前发送至 `mgsxjm@163.com`
