# OPF-python 文件整理说明

更新日期：2026-08-26

本文档记录 OPF-python 仓库中的算例数据整理、Python 路径调整、30-bus 算例补充，以及 QrOPF 旧脚本归档情况。

## 1. 整理结果概览

- 所有现有 `case*.json` 和 `*answer*.txt` 已从仓库根目录移动到 `case_data/`。
- 所有 Python 代码中指向这些文件的路径已更新为 `case_data/...`。
- 名称包含 `QrOPF` 的 Python 文件已移动到已有的 `old_scripts/`。
- 文件名和文件内容保持不变，仅调整存放位置及引用路径。
- 这些改动目前只存在于本地工作区，尚未提交或推送到 GitHub。

## 2. `case_data` 文件夹

当前目录结构如下：

```text
case_data/
├── case5_custom.json
├── case5_custom_pretty.json
├── case9_custom.json
├── case9_custom_pretty.json
├── case14_custom.json
├── case14_custom_pretty.json
├── case30_custom.json
├── 5bus-answer.txt
├── 9bus-answer.txt
├── 14bus-answer.txt
└── 30 bus-answer.txt
```

文件分为三类：

| 类型 | 文件 |
|---|---|
| 紧凑算例 JSON | `case5_custom.json`、`case9_custom.json`、`case14_custom.json`、`case30_custom.json` |
| 易读格式 JSON | `case5_custom_pretty.json`、`case9_custom_pretty.json`、`case14_custom_pretty.json` |
| 标准答案 | `5bus-answer.txt`、`9bus-answer.txt`、`14bus-answer.txt`、`30 bus-answer.txt` |

2-bus 和 3-bus 数据仍由相关 Python 文件中的 `build_model()` 直接构建，仓库原先没有对应的 `case2_custom.json` 或 `case3_custom.json`，本次没有额外创建。

## 3. Python 路径修改

共修改了 38 个 Python 文件，完成 46 处路径替换。主要变化如下：

| 原路径写法 | 新路径写法 |
|---|---|
| `case{n_bus}_custom.json` | `case_data/case{n_bus}_custom.json` |
| `case{n_bus}_custom_pretty.json` | `case_data/case{n_bus}_custom_pretty.json` |
| `case{option}_custom.json` | `case_data/case{option}_custom.json` |
| `5bus-answer.txt` | `case_data/5bus-answer.txt` |
| `9bus-answer.txt` | `case_data/9bus-answer.txt` |
| `14bus-answer.txt` | `case_data/14bus-answer.txt` |
| `30 bus-answer.txt` | `case_data/30 bus-answer.txt` |

修改范围包括仓库根目录、`old/`、`old_scripts/`、`logs/QCE_result/` 和 `output/` 中实际引用算例或答案文件的 Python 代码。

多数求解脚本仍以当前工作目录为基础解析 `case_data/...`。建议始终从仓库根目录运行，例如：

```powershell
Set-Location E:\ZZQ_python_script\OPF_data\OPF-python
python Sympy_OPF_LALM_mu_final_14bus.py
```

运行已归档的 QrOPF 脚本时也应从仓库根目录启动：

```powershell
python old_scripts\Sympy_QrOPF_ALM_mu_final_14bus.py
```

## 4. QrOPF 脚本归档

以下 9 个文件已从仓库根目录移动到 `old_scripts/`：

```text
Sympy_QrOPF_ALM_class.py
Sympy_QrOPF_ALM_class_backup.py
Sympy_QrOPF_ALM_class_notebook_mu_deps.py
Sympy_QrOPF_ALM_mu_final_2bus.py
Sympy_QrOPF_ALM_mu_final_3bus.py
Sympy_QrOPF_ALM_mu_final_5bus.py
Sympy_QrOPF_ALM_mu_final_9bus.py
Sympy_QrOPF_ALM_mu_final_14bus.py
Sympy_QrOPF_simpALM_class.py
```

这些文件之间的模块依赖仍位于同一个目录。归档后已验证 `Sympy_QrOPF_ALM_class_notebook_mu_deps` 可以正常导入。

## 5. 30-bus 算例说明

新增的 `case30_custom.json` 包含：

- 30 个母线；
- 41 条支路；
- 6 台发电机；
- `Sbase = 100`；
- 支路电阻、电抗、充电电纳和变比沿用 `IEEE30.xlsx`；
- 40 条支路恢复 `IEEE30.xlsx` 中逐条给定的容量，仅 6--8 号支路因原始 `0.32 pu` 会导致不可行而调整为 `0.40 pu`；
- 6 台发电机的有功下限恢复为 `[0.50, 0.20, 0.15, 0.10, 0.10, 0.12] pu`；
- 参考母线发电机无功范围设为 `[-1.0, 1.0] pu`，避免源表中 `[0, 0]` 与该算例无功平衡需求矛盾；
- bus 的电压幅值和相角字段保存了下面已验证可行解，可作为模型初始电压点。

对应的 `30 bus-answer.txt` 主要结果为：

| 指标 | 数值 |
|---|---:|
| 目标函数 | `29.91532700` |
| 总有功发电 | `2.8872 pu` |
| 总有功负荷 | `2.8040 pu` |
| 有功损耗 | `0.0832 pu` |
| 总无功发电 | `1.3075 pu` |
| 电压幅值范围 | `[0.9500, 1.0500] pu` |
| 最小支路容量余量 | `0.006078 pu` |
| 负荷供应率 | `100%` |
| 全精度最大功率平衡残差 | `7.27e-14` |

该 JSON 已成功用于构建现有 `SympyACOPFModel`，模型包含 348 个变量。将全精度解代入项目原生等式、变量上下界和支路容量检查后，全部约束均通过；项目原生模型测得最大等式残差约为 `1.00e-12`。

替换前的统一 `rateA = 4.0 pu` 宽松版已备份到：

```text
case_data/archive/case30_custom_relaxed_2026-08-26.json
case_data/archive/30_bus-answer_relaxed_2026-08-26.txt
```

## 6. 已完成的验证

- `case_data/` 顶层共有 11 个算例文件：7 个 JSON 和 4 个答案 TXT；`archive/` 另存 2 个 30-bus 宽松版备份。
- 7 个 JSON 均通过 JSON 解析检查。
- Python 源码中未发现仍指向仓库根目录的旧算例路径。
- 38 个修改后的 Python 文件全部通过 AST 语法解析。
- 3 个原有标准答案路径和 4 个紧凑算例 JSON 路径均通过运行时存在性检查。
- 9 个 QrOPF 文件全部通过 AST 语法解析，模块相互导入测试通过。

## 7. Git 状态说明

当前改动尚未提交。移动已跟踪文件后，在执行 `git add` 之前，Git 可能显示为“根目录文件已删除”和“新目录未跟踪”；执行以下命令后，Git 通常会将其识别为移动或重命名：

```powershell
git status
git add -A
git status
```

仓库中原有的 `.codex_ppt_build` 未提交修改与本次整理无关，本次没有修改或清理这些内容。
