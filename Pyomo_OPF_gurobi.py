#!/usr/bin/env python
# coding: utf-8

TOTAL_STEPS = 15

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def log_step(step, message):
    print(f"Step {step} of {TOTAL_STEPS}: {message}")


from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


# <a href="https://colab.research.google.com/github/edumntg/OPF-python/blob/main/Pyomo_OPF.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:





# In[1]:


#!pip install -q pyomo


# In[2]:


#!wget -N -q "https://ampl.com/dl/open/ipopt/ipopt-linux64.zip"
#!unzip -o -q ipopt-linux64


# In[3]:


log_step(1, "正在导入 Pyomo 和基础数值库。")
from pyomo.environ import *
from pyomo.environ import (
    ConcreteModel, Set, Var, Reals, Objective, Constraint, minimize, SolverFactory, value
)
import numpy as np
from math import pi


# In[4]:


#!pip install gurobipy


# In[5]:


log_step(2, "正在定义 MATPOWER JSON 读取函数。")
import json
def load_matpower_json(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    Sbase = float(data["Sbase"])

    # Convert "k1", "k2", ... into integer keys 1, 2, ...
    buses = {int(k.replace("k", "")): v for k, v in data["buses"].items()}
    lines = {int(k.replace("k", "")): v for k, v in data["lines"].items()}
    gens  = {int(k.replace("k", "")): v for k, v in data["gens"].items()}

    return Sbase, buses, lines, gens


# # Data

# In[6]:


log_step(3, "正在载入 bus、line 和 generator 数据。")
Sbase = 10 # MW

option = 5
if option == 3:
    #[bus_id, bus_type, Vm, Va, Gs, Bs, Pd, Qd]
    buses = {
        1: [1, 0, 1.00, 0.0, 0.0, 0.0, 0.0, 0.0],
        2: [2, 1, 1.01, 0.0, 0.0, 0.0, 0.0, 0.0],
        3: [3, 2, 1.00, 0.0, 0.0, 0.0, 0.3, 0.1]
    }

    lines = {
        1: [1, 2, 0.0192, 0.0575, 0.0264, 1, 30/Sbase],
        2: [1, 3, 0.0452, 0.1852, 0.0204, 1, 30/Sbase],
        3: [2, 3, 0.0570, 0.1737, 0.0184, 1, 30/Sbase]
    }

    gens = {
        1: [1, 0/Sbase, 20/Sbase, -20/Sbase, 100/Sbase, 0.00375, 2, 0],
        2: [2, 0/Sbase, 20/Sbase, -20/Sbase, 100/Sbase, 0.0175, 1.75, 0]
    }
elif option == 2:
    # 2bus model
    Sbase = 10.0
    buses = {
        1: [1, 0, 1.00, 0.0, 0.0, 0.0, 0.0, 0.0],
        2: [2, 1, 1.01, 0.0, 0.0, 0.0, 0.3, 0.1],
    }
    lines = {
        1: [1, 2, 0.0452, 0.1852, 0.0204, 1.0, 30.0 / Sbase],
    }
    gens = {
        1: [1, 0.0 / Sbase, 20.0 / Sbase, -20.0 / Sbase, 100.0 / Sbase, 0.00375, 2.0, 0.0],
    }

else:
    # n-bus model
    Sbase, buses, lines, gens = load_matpower_json(f"case{option}_custom.json")
    #model = SympyACOPFModel(Sbase=Sbase, buses=buses, lines=lines, gens=gens)

print("Model initialized with", option, "buses", lines, "lines and", gens, "generators.")


# # Create Ybus

# In[7]:


log_step(4, "正在根据 line 数据构建 Ybus、G 和 B 矩阵。")
nb = len(buses)
nl = len(lines)
ng = len(gens)

Ybus = np.zeros((nb, nb), dtype=np.complex128)
g = np.zeros((nb, nb))
b = np.zeros((nb,nb))
# Loop through lines
for lineid, linedata in lines.items():
  i = linedata[0]-1
  k = linedata[1]-1
  Z = linedata[2] + 1j*linedata[3]
  Bs = 1j*linedata[4]
  a = linedata[5]

  Ybus[i][i] += (1/(Z*a**2))
  Ybus[k][k] += (1/(Z*a**2))

  Ybus[i][i] += Bs
  Ybus[k][k] += Bs

  Ybus[i][k] -= 1/(a*Z)
  Ybus[k][i] -= 1/(a*Z)

  b[i][k] = Bs.imag
  b[k][i] = Bs.imag

G = Ybus.real
B = Ybus.imag

#print(Ybus)
#print(G)
#print(B)


# # Objective Function

# In[8]:


log_step(5, "正在定义 OPF 目标函数。")
def ObjectiveFunction(model):
  Cost = 0.0
  for genid, gendata in gens.items():
    a = gendata[7]
    b = gendata[6]
    c = gendata[5]

    Cost += c*model.Pgen[genid]**2 + b*model.Pgen[genid] + a

  return Cost
  #return -sum(model.l[i] for i in model.bus)


# # Constraints

# In[9]:


log_step(6, "正在定义发电机、潮流和功率平衡约束函数。")
def MinGen_P(model, g):
    return model.Pgen[g] >= gens[g][1]

def MaxGen_P(model, g):
    return model.Pgen[g] <= gens[g][2]

def MinGen_Q(model, g):
    return model.Qgen[g] >= gens[g][3]

def MaxGen_Q(model, g):
    return model.Qgen[g] <= gens[g][4]

def MaxFlowLineik(model, linea):
    S = lines[linea][6]
    i = lines[linea][0]
    k = lines[linea][1]
    return model.Pflow[i, k]**2 + model.Qflow[i, k]**2 <= S**2

def MaxFlowLineki(model, linea):
    S = lines[linea][6]
    i = lines[linea][1]
    k = lines[linea][0]
    return model.Pflow[i, k]**2 + model.Qflow[i, k]**2 <= S**2

def KirchoffBusesP(model, bus):
    # 该母线上的总发电有功
    Pgbus = sum(model.Pgen[g] for g in model.gen if gens[g][0] == bus)

    # 该母线流出的/流入的线路有功
    Pik = 0
    for linea in model.line:
        i = lines[linea][0]
        j = lines[linea][1]
        if i == bus:
            Pik += model.Pflow[i, j]
        elif j == bus:
            Pik += model.Pflow[j, i]

    # buses[bus][6] = Pd
    return Pgbus == buses[bus][6] + Pik

def KirchoffBusesQ(model, bus):
    # 该母线上的总发电无功
    Qgbus = sum(model.Qgen[g] for g in model.gen if gens[g][0] == bus)

    # 该母线线路无功
    Qik = 0
    for linea in model.line:
        i = lines[linea][0]
        j = lines[linea][1]
        if i == bus:
            Qik += model.Qflow[i, j]
        elif j == bus:
            Qik += model.Qflow[j, i]

    Qshunt = 0

    # buses[bus][7] = Qd
    return Qgbus == buses[bus][7] + Qik + Qshunt

# Lines equations
def Pflow_square_like_original(model, linea):
    i = lines[linea][0]
    j = lines[linea][1]

    Vi2 = model.VR[i]**2 + model.VI[i]**2
    cos_term = model.VR[i]*model.VR[j] + model.VI[i]*model.VI[j]
    sin_term = model.VI[i]*model.VR[j] - model.VR[i]*model.VI[j]

    return model.Pflow[i, j] == (
        (-G[i-1][j-1] + g[i-1][j-1]) * Vi2
        + G[i-1][j-1] * cos_term
        + B[i-1][j-1] * sin_term
    )

def Qflow_square_like_original(model, linea):
    i = lines[linea][0]
    j = lines[linea][1]

    Vi2 = model.VR[i]**2 + model.VI[i]**2
    cos_term = model.VR[i]*model.VR[j] + model.VI[i]*model.VI[j]
    sin_term = model.VI[i]*model.VR[j] - model.VR[i]*model.VI[j]

    return model.Qflow[i, j] == (
        B[i-1][j-1] * Vi2
        - b[i-1][j-1] * Vi2
        + (-B[i-1][j-1]) * cos_term
        + G[i-1][j-1] * sin_term
    )

def Pflow_square_like_original_rev(model, linea):
    i = lines[linea][1]
    j = lines[linea][0]
    Vi2 = model.VR[i]**2 + model.VI[i]**2
    cos_term = model.VR[i]*model.VR[j] + model.VI[i]*model.VI[j]
    sin_term = model.VI[i]*model.VR[j] - model.VR[i]*model.VI[j]
    return model.Pflow[i, j] == (
        (-G[i-1][j-1] + g[i-1][j-1]) * Vi2
        + G[i-1][j-1] * cos_term
        + B[i-1][j-1] * sin_term
    )

def Qflow_square_like_original_rev(model, linea):
    i = lines[linea][1]
    j = lines[linea][0]
    Vi2 = model.VR[i]**2 + model.VI[i]**2
    cos_term = model.VR[i]*model.VR[j] + model.VI[i]*model.VI[j]
    sin_term = model.VI[i]*model.VR[j] - model.VR[i]*model.VI[j]
    return model.Qflow[i, j] == (
        B[i-1][j-1] * Vi2
        - b[i-1][j-1] * Vi2
        + (-B[i-1][j-1]) * cos_term
        + G[i-1][j-1] * sin_term
    )


def V_square_def(model, bus):

    return model.Vsq[bus] == model.VR[bus]**2 + model.VI[bus]**2


# # Finally, solve

# In[10]:


log_step(7, "正在创建 Pyomo 模型、变量、目标函数和约束。")
model = ConcreteModel()

arcs = []
for linea in lines:
    i = lines[linea][0]
    j = lines[linea][1]
    arcs.append((i, j))
    arcs.append((j, i))

model.bus = Set(initialize = buses.keys())
model.line = Set(initialize = lines.keys())
model.gen = Set(initialize = gens.keys())

# Create variables
model.Pgen = Var(model.gen, initialize = 0)
model.Qgen = Var(model.gen, initialize = 0)
model.VR = Var(model.bus, initialize = 1.0, bounds = (-1.1, 1.1))
model.VI = Var(model.bus, initialize = 1.0, bounds = (-1.1, 1.1))
model.Vsq = Var(model.bus, initialize = 1.0, bounds = (0.81, 1.21))

# Line flows
model.arc = Set(initialize=arcs, dimen=2)
model.Pflow = Var(model.arc, initialize=0)
model.Qflow = Var(model.arc, initialize=0)

model.obj = Objective(rule = ObjectiveFunction, sense = minimize)

model.c0 = Constraint(expr = model.VR[1] == 1)
model.c0_im = Constraint(expr = model.VI[1] == 0)

model.c1 = Constraint(model.bus, rule = KirchoffBusesP)
model.c2 = Constraint(model.bus, rule = KirchoffBusesQ)

model.c3 = Constraint(model.gen, rule = MaxGen_P)
model.c4 = Constraint(model.gen, rule = MinGen_P)
model.c5 = Constraint(model.gen, rule = MaxGen_Q)
model.c6 = Constraint(model.gen, rule = MinGen_Q)


model.c7 = Constraint(model.line, rule = MaxFlowLineik)
model.c8 = Constraint(model.line, rule = MaxFlowLineki)

model.c9 = Constraint(model.line, rule = Pflow_square_like_original)
model.c10 = Constraint(model.line, rule = Pflow_square_like_original_rev)
model.c11 = Constraint(model.line, rule = Qflow_square_like_original)
model.c12 = Constraint(model.line, rule = Qflow_square_like_original_rev)


# # Solve

# In[11]:


log_step(8, "正在调用 Gurobi 求解 Pyomo OPF 模型。")
import gurobipy
from pyomo.environ import *
print(gurobipy.gurobi.version())
#solver = SolverFactory('ipopt', executable='/content/ipopt')
solver = SolverFactory('gurobi_direct')
#solver = SolverFactory('cyipopt')
results = solver.solve(model)
print(results.solver.termination_condition)


# In[12]:


log_step(9, "正在定义 ACOPF 结果打印函数。")
def PrintOPFACResults(model, buses, lineas, gens, shunts):
    """
    打印 ACOPF 结果，尽量按所有 buses/lines 数据完整展示。

    Parameters
    ----------
    model : Pyomo model
        已求解完成的模型
    buses : dict
        母线数据，key 为 bus id
    lineas : dict
        线路数据，key 为 line id
    gens : dict
        发电机数据，key 为 gen id
        默认 gens[g][0] 是该机组所在 bus id
    shunts : dict
        并联电纳/无功补偿数据，若没有可传空字典 {}
    """

    nb = len(buses)
    nl = len(lineas)
    ng = len(gens)
    ns = len(shunts)

    # =========================
    # 1) 建立 bus -> gens 映射
    # =========================
    gens_at_bus = {bus_id: [] for bus_id in buses.keys()}
    for g in gens:
        bus_id = gens[g][0]
        if bus_id in gens_at_bus:
            gens_at_bus[bus_id].append(g)

    # =========================
    # 2) 建立 bus -> Qshunt 映射
    #    如果没有 shunt 或格式不确定，默认 0
    # =========================
    qshunt_at_bus = {bus_id: 0.0 for bus_id in buses.keys()}

    for s in shunts:
        try:
            # 假设 shunts[s][0] 是 bus_id, shunts[s][1] 是 Qshunt
            bus_id = shunts[s][0]
            qsh = shunts[s][1]
            if bus_id in qshunt_at_bus:
                qshunt_at_bus[bus_id] += float(qsh)
        except Exception:
            pass

    # =========================
    # 3) 目标函数值
    # =========================
    try:
        objective_value = value(model.obj)
        print("Objective Function Value: {0:.8f}".format(objective_value))
        print("\n")
    except Exception:
        pass

    # =========================
    # 4) 母线结果
    # =========================
    print("BusID\tVR\tVI\tPg\tQg\tl\tPl\tQl\tQshunt\n")

    Pgtotal = 0.0
    Qgtotal = 0.0
    Ploadtotal = 0.0
    Qloadtotal = 0.0

    # Only use the bus-index fallback when Pgen/Qgen are truly indexed by bus.
    # Otherwise overlapping ids such as bus 2 and generator 2 can make the report
    # print the same generator on the wrong bus and count it twice.
    pgen_index_list = list(model.Pgen.index_set())
    qgen_index_list = list(model.Qgen.index_set())
    bus_index_list = list(model.bus)
    use_bus_indexed_fallback = (
        sorted(pgen_index_list) == sorted(bus_index_list)
        and sorted(qgen_index_list) == sorted(bus_index_list)
    )

    # 用 buses.keys() 保证展示所有 bus
    for i in sorted(buses.keys()):
        Vr = model.VR[i]()
        Vi = model.VI[i]()

        # 该母线所有机组出力求和
        Pg = 0.0
        Qg = 0.0
        for g in gens_at_bus.get(i, []):
            try:
                Pg += model.Pgen[g]()
                Qg += model.Qgen[g]()
            except Exception:
                # 如果你的模型是按 bus 建 Pgen[i], Qgen[i]，则退回 bus 索引
                pass

        # 若上面按 gen id 取不到，尝试按 bus id 取
        if use_bus_indexed_fallback and abs(Pg) < 1e-12 and abs(Qg) < 1e-12:
            try:
                Pg = model.Pgen[i]()
                Qg = model.Qgen[i]()
            except Exception:
                Pg = 0.0
                Qg = 0.0

        # 负荷
        Pl = buses[i][6]
        Ql = buses[i][7]

        # shunt
        Qshunt = qshunt_at_bus.get(i, 0.0)

        # l 暂时仍然按 1.0 输出
        lval = 1.0

        print(
            "{0:.0f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}\t{7:.4f}\t{8:.4f}".format(
                i, Vr, Vi, Pg, Qg, lval, Pl, Ql, Qshunt
            )
        )

        Pgtotal += Pg
        Qgtotal += Qg
        Ploadtotal += Pl
        Qloadtotal += Ql

    print("\n")
    print("TOTAL\t\t\t{0:.4f}\t{1:.4f}\t\t{2:.4f}\t{3:.4f}".format(
        Pgtotal, Qgtotal, Ploadtotal, Qloadtotal
    ))
    print("\n\n")

    # =========================
    # 5) 支路潮流结果
    # =========================
    print("Busi\tBusk\tPik\tPki\tQik\tQki")

    Ploss = 0.0
    Qloss = 0.0

    for l in sorted(lineas.keys()):
        i = lineas[l][0]
        j = lineas[l][1]

        Pik = model.Pflow[i, j]()
        Pki = model.Pflow[j, i]()
        Qik = model.Qflow[i, j]()
        Qki = model.Qflow[j, i]()

        print(
            "{0:.0f}\t{1:.0f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}".format(
                i, j, Pik, Pki, Qik, Qki
            )
        )

        Ploss += (Pik + Pki)
        Qloss += (Qik + Qki)

    print("\n")
    print("Total Ploss: {0:.4f}".format(Ploss))
    print("Total Qloss: {0:.4f}".format(Qloss))

    # =========================
    # 6) 负荷供应比例
    # =========================
    if Ploadtotal > 1e-8:
        Pl_supplied = Pgtotal - Ploss
        perc_supplied = (Pl_supplied / Ploadtotal) * 100.0
    else:
        perc_supplied = 0.0

    print("Total Load Supplied: {0:.4f}%".format(perc_supplied))


# In[13]:


log_step(10, "正在输出 ACOPF 求解结果。")
PrintOPFACResults(model, buses, lines, gens, [])


# In[14]:


log_step(11, "正在读取模型维度并显示 arc 集合。")
nb = len(model.bus)
nl = len(model.line)
ng = len(model.gen)


# In[15]:


model.arc.display()


# In[16]:


log_step(12, "正在提取有序解向量并保存到 opf_result_ordered.npy。")
P_G = np.array([model.Pgen[i].value for i in model.gen])
Q_G = np.array([model.Qgen[i].value for i in model.gen])
V_R = np.array([model.VR[b].value for b in model.bus])
V_I = np.array([model.VI[b].value for b in model.bus])
if hasattr(model, "V_sq"):
    V_sq = np.array([model.V_sq[b].value for b in model.bus])
else:
    V_sq = V_R**2 + V_I**2
P_ij = np.array([model.Pflow[i,j].value for i, j in model.arc])
Q_ij = np.array([model.Qflow[i,j].value for i, j in model.arc])
if hasattr(model, "S_tot_sq"):
    S_tot_sq = np.array([model.S_tot_sq[l].value for l in model.line])
else:
    S_tot_sq = P_ij**2 + Q_ij**2
#print("P_G:", P_G)
#print("Q_G:", Q_G)
#print("V_R:", V_R)
#print("V_I:", V_I)  
#print("V_sq:", V_sq)
#print("P_ij:", P_ij)
#print("Q_ij:", Q_ij)
#print("S_tot_sq:", S_tot_sq)

result = np.concatenate([
    P_G,
    Q_G,
    V_R,
    V_I,
    V_sq,
    P_ij,
    Q_ij,
    S_tot_sq
])
#print("Result array:", result, "with shape:", result.shape)
np.save("opf_result_ordered.npy", result)


# In[17]:


log_step(13, "正在导入绘图库并定义网络潮流绘图辅助函数。")
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


# In[18]:


def shorten_segment(p0, p1, d0, d1):
    p0 = np.array(p0, float)
    p1 = np.array(p1, float)
    v = p1 - p0
    L = np.linalg.norm(v)
    if L < 1e-12:
        return p0, p1
    u = v / L
    return p0 + u*d0, p1 - u*d1


# In[19]:


def arc3_midpoint(p0, p1, rad=0.05, t=0.5, offset=0.03):
    """
    For Matplotlib connectionstyle='arc3,rad=...':
    return a label position near the curve at parameter t,
    offset along the curve normal by 'offset' (data units).
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    v = p1 - p0
    L = np.linalg.norm(v)
    if L < 1e-12:
        return p0

    # unit normal to chord
    n_chord = np.array([-v[1], v[0]]) / L
    m = (p0 + p1) / 2.0
    c = m + rad * L * n_chord   # Arc3 control point

    # Quadratic Bezier point
    pt = (1-t)**2 * p0 + 2*(1-t)*t * c + t**2 * p1

    # Quadratic Bezier derivative (tangent)
    dpt = 2*(1-t)*(c - p0) + 2*t*(p1 - c)
    dL = np.linalg.norm(dpt)
    if dL < 1e-12:
        return pt

    t_hat = dpt / dL
    n_hat = np.array([-t_hat[1], t_hat[0]])  # normal to tangent

    # push text off the curve
    return pt + offset * n_hat

def format_mag(x, nd=3):
    return f"{abs(x):.{nd}f}"


# In[20]:


def draw_flow(ax, model, lines, flow_attr, pos, title, color_pos, color_neg):
    ax.set_title(title, fontsize=14, fontweight='bold')

    # 1) 画底图边（灰色拓扑）
    for line in model.line:
        i, j = lines[line][0], lines[line][1]
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                linewidth=0.0, alpha=0.35)

    # 2) 画节点（建议也归一化半径，否则看不出差别/或过大）
    Vs = []
    for bus in model.bus:
        V = (model.VR[bus]()**2 + model.VI[bus]()**2)**0.5
        Vs.append(V)
    Vmin, Vmax = min(Vs), max(Vs)
    span = max(Vmax - Vmin, 1e-6)

    r_map = {}

    for bus, V in zip(model.bus, Vs):
        r = 0.04 + 0.02 * ((V - Vmin) / span)   # 半径控制在 [0.03, 0.08]
        r_map[bus] = r
        circle = plt.Circle(pos[bus], r, color='tab:blue', alpha=0.85, zorder=3)
        ax.add_patch(circle)
        ax.text(pos[bus][0], pos[bus][1], str(bus),
                ha='center', va='center', fontweight='bold', color='white', zorder=4)

        # 计算空白方向
        p = np.asarray(pos[bus], float)

        vec = np.zeros(2)
        for nb in G.neighbors(bus):
            vec += (np.asarray(pos[nb], float) - p)

        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            direction = np.array([1.0, 0.0])  # 孤立点兜底
        else:
            direction = -vec / norm  # 反方向 = 空白方向

        # 标签距离：节点半径 + 额外偏移
        offset = (r_map[bus] + 0.2) * direction
        pv = p + offset

        Vlabel = f"{V:.3f} p.u."  # 你想带单位也行

        ax.text(pv[0], pv[1], Vlabel,
                fontsize=10,
                ha='center', va='center',
                color='black',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
                zorder=6)

    # 3) 收集所有流用于归一化
    flows = []
    for line in model.line:
        i, j = lines[line][0], lines[line][1]
        f = getattr(model, flow_attr)[i, j]()
        flows.append(abs(f))
    fmax = max(flows) if flows else 1.0
    fmax = max(fmax, 1e-9)

    # 4) 只画一次方向箭头（按符号决定方向）
    for line in model.line:
        i, j = lines[line][0], lines[line][1]
        f = getattr(model, flow_attr)[i, j]()

        if f >= 0:
            u, v = i, j
            col = color_pos
        else:
            u, v = j, i
            col = color_neg

        mutation_scale = 12
        gap = 0.010 + 0.002 * mutation_scale   # 让头越大，缩得越多
        p_start, p_end = shorten_segment(pos[u], pos[v], r_map[u] + gap, r_map[v] + gap)



        lw = 1.5 + 3.5 * (abs(f) / fmax)   # 线宽限制在 ~[0.8, 5.8]
        arrow = FancyArrowPatch(
                p_start, p_end,
                arrowstyle='->',
                mutation_scale=mutation_scale,
                linewidth=lw,
                color=col,
                alpha=0.75,
                connectionstyle="arc3,rad=0.05",
                shrinkA=0, shrinkB=0,   # 注意：这里先置 0，因为我们已经用 p_start/p_end 缩过了
                zorder=2
            )
        ax.add_patch(arrow)

        rad = 0.05  # 你当前用的弧度
        mid = arc3_midpoint(p_start, p_end, rad=rad, t=0.5, offset=0.12)

        ax.text(mid[0], mid[1],
                f"{f:.3f} p.u.",
                fontsize=10,
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.", fc="white", ec="none", alpha=0.85),
                zorder=6)

    set_auto_limits(ax, pos, pad=0.25)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')


# In[21]:


def set_auto_limits(ax, pos, pad=0.15):
    xs = np.array([pos[n][0] for n in pos])
    ys = np.array([pos[n][1] for n in pos])
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    dx = xmax - xmin
    dy = ymax - ymin
    ax.set_xlim(xmin - pad*dx, xmax + pad*dx)
    ax.set_ylim(ymin - pad*dy, ymax + pad*dy)


# In[22]:


log_step(14, "正在构建网络图并绘制有功、无功潮流。")
# ---- build graph ----
G = nx.Graph()
for bus in model.bus:
    G.add_node(bus)
for line in model.line:
    i, j = lines[line][0], lines[line][1]
    G.add_edge(i, j)

pos = nx.kamada_kawai_layout(G)   # 比 spring_layout 更稳

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

draw_flow(ax1, model, lines, "Pflow", pos, "Active Power Flow", "tab:green", "tab:red")
draw_flow(ax2, model, lines, "Qflow", pos, "Reactive Power Flow", "tab:blue", "tab:brown")

plt.tight_layout()
output_png = SCRIPT_DIR / f"Pyomo_OPF_gurobi_case{option}_power_flow.png"
fig.savefig(output_png, dpi=180, bbox_inches="tight")
print(f"Saved power flow figure to {output_png.resolve()}")
plt.show()


# In[23]:


log_step(15, "正在逐条打印 line 的有功潮流 Pflow。")
for line in model.line:
    i, j = lines[line][0], lines[line][1]
    f = getattr(model, "Pflow")[i, j]()
    print(i,j,f)

