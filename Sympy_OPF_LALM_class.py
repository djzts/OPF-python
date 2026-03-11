import sympy as sp
import numpy as np
import gurobipy as gp
from gurobipy import GRB

class SympyACOPFModel:
    """
    Build a rectangular-coordinate ACOPF SymPy Lagrangian (without quadratic penalty)
    together with a flat list of variables and their box bounds.

    Parameters
    ----------
    Sbase : float
        Power base (MVA/MW). Used only to interpret default data.
    buses : dict
        Bus data dictionary keyed by bus id (1-based). Each value is a list:
            [bus_id, bus_type, Vm, Va, Gs, Bs, Pd, Qd]
        Here Pd, Qd are in per-unit on Sbase.
    lines : dict
        Line data dictionary keyed by line id. Each value:
            [from_bus, to_bus, r, x, bsh, tap, rateA]
        r, x, bsh in per-unit, tap is magnitude tap ratio, rateA is MVA limit / Sbase.
    gens : dict
        Generator data dictionary keyed by gen id. Each value:
            [bus_id, Pmin, Pmax, Qmin, Qmax, a, b, c]
        With quadratic cost C(P) = a P^2 + b P + c.
    """
    def __init__(self, Sbase=None, buses=None, lines=None, gens=None):
        # --- defaults ---
        if Sbase is None:
            Sbase = 10.0
        if buses is None:
            buses = {
                1: [1, 0, 1.00, 0.0, 0.0, 0.0, 0.0, 0.0],
                2: [2, 1, 1.01, 0.0, 0.0, 0.0, 0.0, 0.0],
                3: [3, 2, 1.00, 0.0, 0.0, 0.0, 0.3, 0.1],
            }
        if lines is None:
            lines = {
                1: [1, 2, 0.0192, 0.0575, 0.0264, 1.0, 30.0 / Sbase],
                2: [1, 3, 0.0452, 0.1852, 0.0204, 1.0, 30.0 / Sbase],
                3: [2, 3, 0.0570, 0.1737, 0.0184, 1.0, 30.0 / Sbase],
            }
        if gens is None:
            gens = {
                1: [1, 0.0 / Sbase, 20.0 / Sbase, -20.0 / Sbase, 100.0 / Sbase, 0.00375, 2.0, 0.0],
                2: [2, 0.0 / Sbase, 20.0 / Sbase, -20.0 / Sbase, 100.0 / Sbase, 0.0175, 1.75, 0.0],
            }

        self.Sbase = float(Sbase)
        self.buses = dict(buses)
        self.lines = dict(lines)
        self.gens = dict(gens)
        self.linear_jacobian = None  # will be built later
        self.n_arcs = 0  # will be set in _build_index_sets()

        # basic sets and mappings
        self._build_index_sets()
        # network parameters (Ybus, G,B, branch admittances)
        self._build_network_matrices()
        # build SymPy decision variables and box bounds
        self._build_variables()
        # initialize Lagrange multipliers (all zeros)
        self.reset_lambdas(0.0)
        # build initial Lagrangian with current lambdas
        self._build_lagrangian()

    
    # ------------------------------------------------------------------
    # index / data helpers
    # ------------------------------------------------------------------

    def _build_index_sets(self):
        """
        Build all index sets and mapping dictionaries used by the ACOPF model.

        Created attributes
        ------------------
        bus_ids : list[int]
            Sorted bus ids.
        line_ids : list[int]
            Sorted line ids.
        gen_ids : list[int]
            Sorted generator ids.

        n_buses, n_lines, n_gens, n_arcs : int
            Number of buses / lines / generators / directed arcs.

        bus_index : dict
            bus_id -> 0-based bus index
        gen_index : dict
            gen_id -> 0-based generator index

        gen_indices_by_bus : dict
            bus_id -> list of generator indices at that bus
        gen_index_by_bus : dict
            bus_id -> first generator index at that bus
            (kept only for backward compatibility)

        line_collection : list[tuple[int, int]]
            For each line in self.line_ids order, store (from_bus_idx, to_bus_idx)

        line_pos : dict
            line_id -> position in self.line_ids / self.line_collection

        arc_ids : list[tuple[int, int]]
            Directed arc identifiers:
            (line_id, +1) means from_bus -> to_bus
            (line_id, -1) means to_bus   -> from_bus

        arc_collection : list[tuple[int, int]]
            Directed arc bus-index pairs aligned with arc_ids

        arc_to_line : list[int]
            For each arc position, corresponding line_id

        arc_index : dict
            (line_id, direction) -> arc position
        """

        # --------------------------------------------------
        # 1) Sorted ids and basic sizes
        # --------------------------------------------------
        self.bus_ids = sorted(self.buses.keys())
        self.line_ids = sorted(self.lines.keys())
        self.gen_ids = sorted(self.gens.keys())

        self.n_buses = len(self.bus_ids)
        self.n_lines = len(self.line_ids)
        self.n_gens = len(self.gen_ids)

        # --------------------------------------------------
        # 2) Basic index maps
        # --------------------------------------------------
        self.bus_index = {bid: i for i, bid in enumerate(self.bus_ids)}
        self.gen_index = {gid: i for i, gid in enumerate(self.gen_ids)}

        # --------------------------------------------------
        # 3) Generator-to-bus maps
        # --------------------------------------------------
        # Main version: allow multiple generators on one bus
        self.gen_indices_by_bus = {bid: [] for bid in self.bus_ids}
        for gid in self.gen_ids:
            bus_id = self.gens[gid][0]
            self.gen_indices_by_bus[bus_id].append(self.gen_index[gid])

        # Backward-compatible version: keep only the first generator on each bus
        self.gen_index_by_bus = {}
        for bid, idx_list in self.gen_indices_by_bus.items():
            if idx_list:
                self.gen_index_by_bus[bid] = idx_list[0]

        # --------------------------------------------------
        # 4) Undirected line collection
        # --------------------------------------------------
        self.line_collection = []
        for lid in self.line_ids:
            fbus, tbus = self.lines[lid][0], self.lines[lid][1]
            i = self.bus_index[fbus]
            j = self.bus_index[tbus]
            self.line_collection.append((i, j))

        self.line_pos = {lid: k for k, lid in enumerate(self.line_ids)}

        # --------------------------------------------------
        # 5) Directed arc collection
        # --------------------------------------------------
        # (lid, +1): from_bus -> to_bus
        # (lid, -1): to_bus   -> from_bus
        self.arc_ids = []
        self.arc_collection = []
        self.arc_to_line = []

        for lid in self.line_ids:
            fbus, tbus = self.lines[lid][0], self.lines[lid][1]
            i = self.bus_index[fbus]
            j = self.bus_index[tbus]

            # forward arc
            self.arc_ids.append((lid, +1))
            self.arc_collection.append((i, j))
            self.arc_to_line.append(lid)

            # reverse arc
            self.arc_ids.append((lid, -1))
            self.arc_collection.append((j, i))
            self.arc_to_line.append(lid)

        self.n_arcs = len(self.arc_collection)
        self.arc_index = {arc_id: k for k, arc_id in enumerate(self.arc_ids)}


    def build_initial_x0(self):
        """
        构造一个简单、通用的初始点 x^0，顺序与 self.variable_list 一致：
        [P_G (ng), Q_G (ng), V_R (nb), V_I (nb), V_sq (nb),
         P_ij (na), Q_ij (na), S_tot_sq (na)].

        设计原则：
        - P_G: 按 Pmax 比例分担总有功负荷 sum(P_D)，再剪裁到 [Pmin, Pmax]
        - Q_G: 按 P_G 比例分担总无功负荷 sum(Q_D)，再剪裁到 [Qmin, Qmax]
        - V_R, V_I: 按 bus 中给的 Vm, Va 转矩形坐标
        - V_sq: = V_R^2 + V_I^2
        - P_ij, Q_ij, S_tot_sq: 全部初始化为 0
        """

        nb = self.n_buses
        nl = self.n_lines
        ng = self.n_gens
        na = self.n_arcs

        # -------- 1) Generators: P_G 初值 --------
        # 总有功负荷
        total_Pd = float(np.sum(self.P_D))
        P_G0 = np.zeros(ng)

        # 机组 Pmin, Pmax
        Pmin = np.array([self.gens[gid][1] for gid in self.gen_ids], dtype=float)
        Pmax = np.array([self.gens[gid][2] for gid in self.gen_ids], dtype=float)

        if total_Pd > 1e-8:
            # 按 Pmax 比例分担总负荷
            weights = Pmax / np.sum(Pmax)
            P_G0 = total_Pd * weights
            # 剪裁到 [Pmin, Pmax]
            P_G0 = np.minimum(np.maximum(P_G0, Pmin), Pmax)
        else:
            # 没有负荷时，干脆都取 Pmin（通常是 0）
            P_G0 = Pmin.copy()

        # -------- 2) Generators: Q_G 初值 --------
        total_Qd = float(np.sum(self.Q_D))
        Q_G0 = np.zeros(ng)

        Qmin = np.array([self.gens[gid][3] for gid in self.gen_ids], dtype=float)
        Qmax = np.array([self.gens[gid][4] for gid in self.gen_ids], dtype=float)

        if abs(total_Qd) > 1e-8:
            # 用 P_G0 的比例来分担 Q 负荷
            Pg_positive = P_G0.copy()
            if np.sum(Pg_positive) <= 1e-8:
                # 如果 P_G 均接近 0，就平均分
                weights_Q = np.ones(ng) / ng
            else:
                weights_Q = Pg_positive / np.sum(Pg_positive)

            Q_G0 = total_Qd * weights_Q
            Q_G0 = np.minimum(np.maximum(Q_G0, Qmin), Qmax)
        else:
            # 无无功负荷时，简单设为 0，若超出范围再剪裁
            Q_G0 = np.zeros(ng)
            Q_G0 = np.minimum(np.maximum(Q_G0, Qmin), Qmax)

        # -------- 3) Bus voltages: V_R, V_I, V_sq --------
        V_R0 = np.zeros(nb)
        V_I0 = np.zeros(nb)
        V_sq0 = np.zeros(nb)

        for bid in self.bus_ids:
            bdata = self.buses[bid]
            bus_idx = self.bus_index[bid]

            Vm = float(bdata[2])  # 电压幅值
            Va_deg = float(bdata[3])  # 相角（一般是度）
            Va_rad = Va_deg * np.pi / 180.0

            Vr = Vm * np.cos(Va_rad)
            Vi = Vm * np.sin(Va_rad)

            V_R0[bus_idx] = Vr
            V_I0[bus_idx] = Vi
            V_sq0[bus_idx] = Vr**2 + Vi**2

        # 剪裁到变量给定的 box bounds（防止数据比较奇怪时越界）
        V_R_lb = np.array([b[0] for b in [[-1.1, 1.1]] * nb])
        V_R_ub = np.array([b[1] for b in [[-1.1, 1.1]] * nb])
        V_I_lb = V_R_lb.copy()
        V_I_ub = V_R_ub.copy()
        V_sq_lb = np.array([0.9**2] * nb)
        V_sq_ub = np.array([1.1**2] * nb)

        V_R0 = np.minimum(np.maximum(V_R0, V_R_lb), V_R_ub)
        V_I0 = np.minimum(np.maximum(V_I0, V_I_lb), V_I_ub)
        V_sq0 = np.minimum(np.maximum(V_sq0, V_sq_lb), V_sq_ub)

        # -------- 4) Branch flows: P_ij, Q_ij, S_tot_sq --------
        P_ij0 = np.zeros(na)
        Q_ij0 = np.zeros(na)
        S_tot_sq0 = np.zeros(na)

        # S_tot_sq 上界来自每条线的 rate^2
        Ssq_lb = np.zeros(na)
        Ssq_ub = np.array([self.lines[lid][6]**2 for lid in self.arc_to_line], dtype=float)
        S_tot_sq0 = np.minimum(np.maximum(S_tot_sq0, Ssq_lb), Ssq_ub)

        # -------- 5) 按 variable_list 顺序拼接成 x0 --------
        x0 = np.concatenate([
            P_G0,       # ng
            Q_G0,       # ng
            V_R0,       # nb
            V_I0,       # nb
            V_sq0,      # nb
            P_ij0,      # na
            Q_ij0,      # na
            S_tot_sq0   # na
        ])

        assert x0.size == len(self.variable_list), \
            f"构造的 x0 长度 {x0.size} 与 variable_list 长度 {len(self.variable_list)} 不一致"

        return x0

        

    def _build_network_matrices(self):
        nb = self.n_buses
        nl = self.n_lines

        # build Ybus with simple MATPOWER-like π-model (no taps other than magnitude)
        Ybus = np.zeros((nb, nb), dtype=np.complex128)
        # series admittances for each line (for branch flow equations)
        g_series = np.zeros(nl)
        b_series = np.zeros(nl)

        for ell, lid in enumerate(self.line_ids):
            from_bus, to_bus, r, x, bsh, tap, rate = self.lines[lid]
            i = self.bus_index[from_bus]
            j = self.bus_index[to_bus]

            z = r + 1j * x
            if z == 0:
                # avoid division by zero; treat as very large admittance
                y = 1e6 + 0j
            else:
                y = 1.0 / z  # series admittance

            # shunt susceptance (total), split half to each end
            Bc = 1j * bsh

            a = tap if tap != 0 else 1.0

            # contributions to Ybus (standard π-model with tap on "from" side)
            Ybus[i, i] += (y / (a ** 2)) + Bc / 2.0
            Ybus[j, j] += y + Bc / 2.0
            Ybus[i, j] -= y / a
            Ybus[j, i] -= y / a

            g_series[ell] = y.real
            b_series[ell] = y.imag

        self.Ybus = Ybus
        self.G_mat = Ybus.real
        self.B_mat = Ybus.imag
        self.g_series = g_series
        self.b_series = b_series

        # build load vectors Pd, Qd in p.u. aligned with bus index order
        self.P_D = np.zeros(nb)
        self.Q_D = np.zeros(nb)
        for bid, bdata in self.buses.items():
            idx = self.bus_index[bid]
            Pd = bdata[6]  # already in p.u. on Sbase
            Qd = bdata[7]
            self.P_D[idx] = Pd
            self.Q_D[idx] = Qd

    # ------------------------------------------------------------------
    # variables & bounds
    # ------------------------------------------------------------------
    @staticmethod
    def _var_list_insert(var_list, bound_list, variable_list, Var_bound_list):
        for v, bnd in zip(var_list, bound_list):
            variable_list.append(v)
            Var_bound_list.append(list(bnd))
        return variable_list, Var_bound_list

    def _build_variables(self):
        nb = self.n_buses
        na = self.n_arcs
        ng = self.n_gens

        self.variable_list = []
        self.Var_bound_list = []

        # Generators: P_G, Q_G
        self.P_G = sp.symbols(f'P_G0:{ng}')
        # bounds from gens: [Pmin, Pmax]
        P_G_bound = []
        for gid in self.gen_ids:
            gdata = self.gens[gid]
            P_G_bound.append([gdata[1], gdata[2]])
        self.variable_list, self.Var_bound_list = self._var_list_insert(
            self.P_G, P_G_bound, self.variable_list, self.Var_bound_list
        )

        self.Q_G = sp.symbols(f'Q_G0:{ng}')
        Q_G_bound = []
        for gid in self.gen_ids:
            gdata = self.gens[gid]
            Q_G_bound.append([gdata[3], gdata[4]])
        self.variable_list, self.Var_bound_list = self._var_list_insert(
            self.Q_G, Q_G_bound, self.variable_list, self.Var_bound_list
        )

        # Bus voltages: V_R, V_I
        self.V_R = sp.symbols(f'V_R0:{nb}')
        # simple rectangular bounds derived from magnitude in [0.9, 1.1]
        V_R_bound = [[-1.1, 1.1] for _ in range(nb)]
        self.variable_list, self.Var_bound_list = self._var_list_insert(
            self.V_R, V_R_bound, self.variable_list, self.Var_bound_list
        )

        self.V_I = sp.symbols(f'V_I0:{nb}')
        V_I_bound = [[-1.1, 1.1] for _ in range(nb)]
        self.variable_list, self.Var_bound_list = self._var_list_insert(
            self.V_I, V_I_bound, self.variable_list, self.Var_bound_list
        )

        # Voltage magnitude squared: V_sq
        self.V_sq = sp.symbols(f'V_sq0:{nb}')
        V_sq_bound = [[0.9 ** 2, 1.1 ** 2] for _ in range(nb)]
        self.variable_list, self.Var_bound_list = self._var_list_insert(
            self.V_sq, V_sq_bound, self.variable_list, self.Var_bound_list
        )


        
        # Branch flows on arcs: P_ij, Q_ij, S_tot_sq
        na = self.n_arcs

        # thermal limit (per-unit MVA) from line data
        self.P_ij = sp.symbols(f'P_ij0:{na}')
        P_ij_bound = []
        for arc_k, lid in enumerate(self.arc_to_line):
            rate = self.lines[lid][6]
            P_ij_bound.append([-rate, rate])
        self.variable_list, self.Var_bound_list = self._var_list_insert(
            self.P_ij, P_ij_bound, self.variable_list, self.Var_bound_list
        )

        self.Q_ij = sp.symbols(f'Q_ij0:{na}')
        Q_ij_bound = P_ij_bound.copy()
        self.variable_list, self.Var_bound_list = self._var_list_insert(
            self.Q_ij, Q_ij_bound, self.variable_list, self.Var_bound_list
        )

        self.S_tot_sq = sp.symbols(f'S_tot_sq0:{na}')
        S_tot_sq_bound = []
        for arc_k, lid in enumerate(self.arc_to_line):
            rate = self.lines[lid][6]
            S_tot_sq_bound.append([0.0, rate ** 2])
        self.variable_list, self.Var_bound_list = self._var_list_insert(
            self.S_tot_sq, S_tot_sq_bound, self.variable_list, self.Var_bound_list
        )

    # ------------------------------------------------------------------
    # Lagrange multipliers control
    # ------------------------------------------------------------------
    def reset_lambdas(self, value=None):
        """
        Initialize or reset all Lagrange multipliers.

        Cases
        -----
        1) value is None        -> all lambdas = 0
        2) value is float/int   -> all lambdas = that scalar
        3) value is vector      -> assign according to stacking order:
            [lambda_P_bal,
                lambda_Q_bal,
                lambda_P_flow,
                lambda_Q_flow,
                lambda_Vsq,
                lambda_Ssq,
                lambda_ref]
        """

        nb = self.n_buses
        na = self.n_arcs

        # total number of multipliers
        total_dim = 2 * nb + 2 * na + nb + na + 1
        # = P_bal(nb) + Q_bal(nb)
        # + P_flow(nl) + Q_flow(nl)
        # + Vsq(nb)
        # + Ssq(nl)
        # + ref(1)

        # ---------------------------------------------------
        # CASE 1: value is None → set all to 0
        # ---------------------------------------------------
        if value is None:
            scalar = 0.0

            self.lambda_P_bal = [scalar] * nb
            self.lambda_Q_bal = [scalar] * nb
            self.lambda_P_flow = [scalar] * na
            self.lambda_Q_flow = [scalar] * na
            self.lambda_Vsq = [scalar] * nb
            self.lambda_Ssq = [scalar] * na
            self.lambda_ref = scalar

        # ---------------------------------------------------
        # CASE 2: value is scalar
        # ---------------------------------------------------
        elif isinstance(value, (int, float)):
            scalar = float(value)

            self.lambda_P_bal = [scalar] * nb
            self.lambda_Q_bal = [scalar] * nb
            self.lambda_P_flow = [scalar] * na
            self.lambda_Q_flow = [scalar] * na
            self.lambda_Vsq = [scalar] * nb
            self.lambda_Ssq = [scalar] * na
            self.lambda_ref = scalar

        # ---------------------------------------------------
        # CASE 3: value is vector
        # ---------------------------------------------------
        else:
            vec = np.asarray(value, dtype=float).flatten()

            if len(vec) != total_dim:
                raise ValueError(
                    f"Lambda vector length mismatch. "
                    f"Expected {total_dim}, got {len(vec)}."
                )

            idx = 0

            # Bus power balance
            self.lambda_P_bal = vec[idx:idx + nb].tolist()
            idx += nb

            self.lambda_Q_bal = vec[idx:idx + nb].tolist()
            idx += nb

            # Branch flow definitions
            self.lambda_P_flow = vec[idx:idx + na].tolist()
            idx += na

            self.lambda_Q_flow = vec[idx:idx + na].tolist()
            idx += na

            # Voltage magnitude definition
            self.lambda_Vsq = vec[idx:idx + nb].tolist()
            idx += nb

            # Branch S_sq definition
            self.lambda_Ssq = vec[idx:idx + na].tolist()
            idx += na   

            # Reference bus
            self.lambda_ref = float(vec[idx])

        # ---------------------------------------------------
        # rebuild lambda_vec in correct order
        # ---------------------------------------------------
        self.lambda_vec = [
            *self.lambda_P_bal,
            *self.lambda_Q_bal,
            *self.lambda_P_flow,
            *self.lambda_Q_flow,
            *self.lambda_Vsq,
            *self.lambda_Ssq,
            self.lambda_ref
        ]

                        
    # ------------------------------------------------------------------
    # Build Lagrangian (without quadratic penalty)
    # ------------------------------------------------------------------
    def _build_lagrangian(self, ref_bus_id=None):
        """
        Build the (classical) Lagrangian:

            L(x, λ) = f(x) + λ^T h(x)

        using the current values of the Lagrange multipliers stored in the object.
        """
        nb = self.n_buses
        #nl = self.n_lines
        na = self.n_arcs
        ng = self.n_gens

        buses_range = range(nb)

        # generator cost: C(P) = a P^2 + b P + c, aligned with gen_ids
        a_cost = []
        b_cost = []
        c_cost = []
        for gid in self.gen_ids:
            gdata = self.gens[gid]
            a_cost.append(gdata[5])
            b_cost.append(gdata[6])
            c_cost.append(gdata[7])

        # objective
        obj = 0
        for gi in range(ng):
            PGi = self.P_G[gi]
            obj += 0.5 * a_cost[gi] * PGi ** 2 + b_cost[gi] * PGi + c_cost[gi]

        L = obj

        # Convenience aliases
        V_R = self.V_R
        V_I = self.V_I
        V_sq = self.V_sq
        P_ij = self.P_ij
        Q_ij = self.Q_ij
        S_tot_sq = self.S_tot_sq

        G_mat = self.G_mat
        B_mat = self.B_mat
        g_series = self.g_series
        b_series = self.b_series
        arc_collection = self.arc_collection

        P_D = self.P_D
        Q_D = self.Q_D

        # map bus index -> generator sympy index (or None)
        gen_sym_indices_by_bus_idx = {self.bus_index[bid]: gids for bid, gids in self.gen_indices_by_bus.items()}

        # ---------------------------
        # (1) Active power balance constraints
        # ---------------------------
        for i in buses_range:
            ViR = V_R[i]
            ViI = V_I[i]

            # sum_j ( G_ij VjR - B_ij VjI ), sum_j ( G_ij VjI + B_ij VjR )
            sum_GR_BI = 0
            sum_GI_BR = 0
            for j in buses_range:
                VjR = V_R[j]
                VjI = V_I[j]
                Gij = G_mat[i, j]
                Bij = B_mat[i, j]
                sum_GR_BI += Gij * VjR - Bij * VjI
                sum_GI_BR += Gij * VjI + Bij * VjR

            P_inj = ViR * sum_GR_BI + ViI * sum_GI_BR

            # if bus i has generator
            PG_sum = 0
            if i in gen_sym_indices_by_bus_idx:
                for gi in gen_sym_indices_by_bus_idx[i]:
                    PG_sum += self.P_G[gi]

            h_P = PG_sum - P_D[i] - P_inj

            L += self.lambda_P_bal[i] * h_P

        # ---------------------------
        # (2) Reactive power balance constraints
        # ---------------------------
        for i in buses_range:
            ViR = V_R[i]
            ViI = V_I[i]

            sum_GR_BI = 0
            sum_GI_BR = 0
            for j in buses_range:
                VjR = V_R[j]
                VjI = V_I[j]
                Gij = G_mat[i, j]
                Bij = B_mat[i, j]
                sum_GR_BI += Gij * VjR - Bij * VjI
                sum_GI_BR += Gij * VjI + Bij * VjR

            Q_inj = ViI * sum_GR_BI - ViR * sum_GI_BR

            QG_sum = 0
            if i in gen_sym_indices_by_bus_idx:
                for gi in gen_sym_indices_by_bus_idx[i]:
                    QG_sum += self.Q_G[gi]

            h_Q = QG_sum - Q_D[i] - Q_inj

            L += self.lambda_Q_bal[i] * h_Q

        # ---------------------------
        # (3) Branch power-flow definition constraints
        # ---------------------------
        for a, (i, j) in enumerate(arc_collection):
            ViR = V_R[i]
            ViI = V_I[i]
            VjR = V_R[j]
            VjI = V_I[j]

            lid = self.arc_to_line[a]
            ell = self.line_pos[lid]

            g_ij = g_series[ell]
            b_ij = b_series[ell]

            # P_ij definition (rectangular form)
            P_expr = (
                ViR * (g_ij * (ViR - VjR) - b_ij * (ViI - VjI))
                + ViI * (g_ij * (ViI - VjI) + b_ij * (ViR - VjR))
            )

            # Q_ij definition (rectangular form)
            Q_expr = (
                ViI * (g_ij * (ViR - VjR) - b_ij * (ViI - VjI))
                - ViR * (g_ij * (ViI - VjI) + b_ij * (ViR - VjR))
            )

            h_P_flow = self.P_ij[a] - P_expr
            h_Q_flow = self.Q_ij[a] - Q_expr

            L += self.lambda_P_flow[a] * h_P_flow
            L += self.lambda_Q_flow[a] * h_Q_flow

        # ---------------------------
        # (4) Voltage magnitude definition
        # ---------------------------
        for i in buses_range:
            h_Vsq = V_sq[i] - (V_R[i] ** 2 + V_I[i] ** 2)
            L += self.lambda_Vsq[i] * h_Vsq

        # ---------------------------
        # (5) Branch S_tot_sq definition
        # ---------------------------
        for a in range(na):
            h_Ssq = S_tot_sq[a] - (P_ij[a] ** 2 + Q_ij[a] ** 2)
            L += self.lambda_Ssq[a] * h_Ssq

        # ---------------------------
        # (6) Reference bus constraint: V_ref^I = 0
        # ---------------------------
        if ref_bus_id is None:
            # default: first bus id
            ref_bus_id = self.bus_ids[0]
        ref_idx = self.bus_index[ref_bus_id]
        h_ref = V_I[ref_idx]
        L += self.lambda_ref * h_ref

        self.Lagrange = L
        return L

    def build_h_symbolic(self, ref_bus_id=None):
        """
        构造符号形式的约束列表 h_sym(x)，顺序必须和 build_h_func 完全一致：
        [ c_i^P (∀ buses),
          c_i^Q (∀ buses),
          c_{ij}^{P,flow} (∀ lines),
          c_{ij}^{Q,flow} (∀ lines),
          c_i^{Vsq} (∀ buses),
          c_{ij}^S (∀ lines),
          c^{ref} ].
        """
        nb = self.n_buses
        na = self.n_arcs
        ng = self.n_gens
        nl = self.n_lines
        buses_range = range(nb)

        # 符号变量别名
        P_G  = self.P_G
        Q_G  = self.Q_G
        V_R  = self.V_R
        V_I  = self.V_I
        V_sq = self.V_sq
        P_ij = self.P_ij
        Q_ij = self.Q_ij
        S_tot_sq = self.S_tot_sq

        # 数据别名
        G_mat = self.G_mat
        B_mat = self.B_mat
        g_series = self.g_series
        b_series = self.b_series
        arc_collection = self.arc_collection

        P_D = self.P_D
        Q_D = self.Q_D

        # bus -> gen 索引 (0..ng-1)
        gen_sym_index_by_bus_idx = {}
        for bus_id, gen_idx in self.gen_index_by_bus.items():
            bus_idx = self.bus_index[bus_id]
            gen_sym_index_by_bus_idx[bus_idx] = gen_idx

        if ref_bus_id is None:
            ref_bus_id = self.bus_ids[0]
        ref_idx = self.bus_index[ref_bus_id]

        residuals = []

        # 1) Active power balance c_i^P
        for i in buses_range:
            ViR = V_R[i]
            ViI = V_I[i]

            sum_GR_BI = 0
            sum_GI_BR = 0
            for j in buses_range:
                VjR = V_R[j]
                VjI = V_I[j]
                Gij = G_mat[i, j]
                Bij = B_mat[i, j]
                sum_GR_BI += Gij * VjR - Bij * VjI
                sum_GI_BR += Gij * VjI + Bij * VjR

            P_inj = ViR * sum_GR_BI + ViI * sum_GI_BR

            if i in gen_sym_index_by_bus_idx:
                gi = gen_sym_index_by_bus_idx[i]
                cP = P_G[gi] - P_D[i] - P_inj
            else:
                cP = - P_D[i] - P_inj

            residuals.append(cP)

        # 2) Reactive power balance c_i^Q
        for i in buses_range:
            ViR = V_R[i]
            ViI = V_I[i]

            sum_GR_BI = 0
            sum_GI_BR = 0
            for j in buses_range:
                VjR = V_R[j]
                VjI = V_I[j]
                Gij = G_mat[i, j]
                Bij = B_mat[i, j]
                sum_GR_BI += Gij * VjR - Bij * VjI
                sum_GI_BR += Gij * VjI + Bij * VjR

            Q_inj = ViI * sum_GR_BI - ViR * sum_GI_BR

            if i in gen_sym_index_by_bus_idx:
                gi = gen_sym_index_by_bus_idx[i]
                cQ = Q_G[gi] - Q_D[i] - Q_inj
            else:
                cQ = - Q_D[i] - Q_inj

            residuals.append(cQ)

        # 3) Branch power-flow definitions: c_{ij}^{P,flow}, c_{ij}^{Q,flow}
        for a, (i, j) in enumerate(arc_collection):
            ViR = V_R[i]
            ViI = V_I[i]
            VjR = V_R[j]
            VjI = V_I[j]
            
            lid = self.arc_to_line[a]
            ell = self.line_pos[lid]

            g_ij = g_series[ell]
            b_ij = b_series[ell]

            P_expr = (
                ViR * (g_ij * (ViR - VjR) - b_ij * (ViI - VjI))
                + ViI * (g_ij * (ViI - VjI) + b_ij * (ViR - VjR))
            )
            Q_expr = (
                ViI * (g_ij * (ViR - VjR) - b_ij * (ViI - VjI))
                - ViR * (g_ij * (ViI - VjI) + b_ij * (ViR - VjR))
            )

            cP_flow = P_ij[a] - P_expr
            cQ_flow = Q_ij[a] - Q_expr

            residuals.append(cP_flow)
            residuals.append(cQ_flow)

        # 4) Voltage magnitude definition c_i^{Vsq}
        for i in buses_range:
            cV = V_sq[i] - (V_R[i] ** 2 + V_I[i] ** 2)
            residuals.append(cV)

        # 5) Branch S_tot_sq definition c_{ij}^S
        for a in range(na):
            cS = S_tot_sq[a] - (P_ij[a] ** 2 + Q_ij[a] ** 2)
            residuals.append(cS)

        # 6) Reference bus constraint c^{ref} = V_I[ref_idx]
        c_ref = V_I[ref_idx]
        residuals.append(c_ref)

        return residuals

    def build_linear_ALM_Lagrangian_syms(self, x_center, rho, ref_bus_id=None, mu_prox=0.0):
        """
        构造一次“线性化增广拉格朗日” L^(k)(x)：
            L^(k)(x) = f(x)
                    + λ^T [ h(x^k) + J(x^k)(x - x^k) ]
                    + (rho/2) * || h(x^k) + J(x^k)(x - x^k) ||^2
                    + (mu_prox/2) * ||x - x^k||^2

        这里 J(x^k) 用 SymPy 的解析求导得到，而不是数值差分。
        """
        x_syms = self.variable_list
        nvar = len(x_syms)

        x0 = np.asarray(x_center, dtype=float).flatten()
        if x0.size != nvar:
            raise ValueError(f"x_center length mismatch: expected {nvar}, got {x0.size}")

        rho = float(rho)

        # 当前 λ 向量（扁平）
        lam = np.asarray(self.lambda_vec, dtype=float).flatten()

        # ---------- 1) 构造目标函数 f(x) ----------
        nb = self.n_buses
        ng = self.n_gens

        a_cost = []
        b_cost = []
        c_cost = []
        for gid in self.gen_ids:
            gdata = self.gens[gid]
            a_cost.append(gdata[5])
            b_cost.append(gdata[6])
            c_cost.append(gdata[7])

        obj = 0
        for gi in range(ng):
            PGi = self.P_G[gi]
            obj += 0.5 * a_cost[gi] * PGi ** 2 + b_cost[gi] * PGi + c_cost[gi]

        L = obj

        # ---------- 2) 符号构造 h(x) & J(x) ----------
        h_sym_list = self.build_h_symbolic(ref_bus_id=ref_bus_id)
        mcon = len(h_sym_list)

        if lam.size != mcon:
            raise ValueError(f"lambda size {lam.size} != number of constraints {mcon}")

        h_vec = sp.Matrix(h_sym_list)
        J_sym = h_vec.jacobian(x_syms)  # (mcon, nvar)

        self.linear_jacobian = J_sym

        # ---------- 3) 在 x_center 处代值得到 h(x^k) 和 J(x^k) ----------
        subs_dict = {sym: sp.Float(val) for sym, val in zip(x_syms, x0)}

        # h(x^k)
        h0_sym = h_vec.subs(subs_dict)
        h0 = np.array([float(h0_sym[i]) for i in range(mcon)], dtype=float)

        # J(x^k)
        J_num = np.zeros((mcon, nvar), dtype=float)
        J_eval = J_sym.subs(subs_dict)
        for i in range(mcon):
            for j in range(nvar):
                J_num[i, j] = float(J_eval[i, j])

        # ---------- 4) 构造线性化约束 \tilde h_i(x) ----------
        h_lin_exprs = []
        for i in range(mcon):
            expr = sp.Float(h0[i])
            for j in range(nvar):
                coef = J_num[i, j]
                if coef != 0.0:
                    expr += sp.Float(coef) * (x_syms[j] - sp.Float(x0[j]))
            h_lin_exprs.append(expr)

        # ---------- 5) 线性 Lagrange 项 λ^T \tilde h(x) ----------
        for i in range(mcon):
            if lam[i] != 0.0:
                L += sp.Float(lam[i]) * h_lin_exprs[i]

        # ---------- 6) 二次惩罚项 (rho/2) * ||\tilde h(x)||^2 ----------
        if rho > 0.0:
            rho_half = sp.Float(rho) / 2.0
            for expr in h_lin_exprs:
                L += rho_half * (expr ** 2)

        # ---------- 7) proximal 项 (mu_prox/2) * ||x - x0||^2 ----------
        if mu_prox > 0.0:
            mu_half = sp.Float(mu_prox) / 2.0
            for j in range(nvar):
                L += mu_half * (x_syms[j] - sp.Float(x0[j])) ** 2

        self.Lagrange_linear_ALM = L
        return L, self.variable_list, self.Var_bound_list

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_Lagrangian(self, ref_bus_id=None):
        """
        Rebuild and return the current Lagrangian (called L_aug for compatibility),
        together with the flat variable list and bound list.

        Returns
        -------
        L_aug : sympy.Expr
            The Lagrangian expression with current Lagrange multipliers.
        variable_list : list of sympy.Symbol
            Flat list of primal decision variables.
        Var_bound_list : list of [lower, upper]
            Corresponding box bounds for each variable.
        """
        self._build_lagrangian(ref_bus_id=ref_bus_id)
        return self.Lagrange, self.variable_list, self.Var_bound_list
    
    # ------------------------------------------------------------------
    # Equality constraints h(x): build numerical h_func(x)
    # ------------------------------------------------------------------
    def build_h_func(self, ref_bus_id=None):
        """
        Return a callable h_func(x) that evaluates all equality-constraint
        residuals h(x) for a given primal vector x.

        The constraint order is:
        [ c_i^P (∀ buses),
          c_i^Q (∀ buses),
          c_{ij}^{P,flow} (∀ lines),
          c_{ij}^{Q,flow} (∀ lines),
          c_i^{Vsq} (∀ buses),
          c_{ij}^S (∀ lines),
          c^{ref} ].

        x must be a flat array/list in the same order as self.variable_list:
        [P_G (ng),
         Q_G (ng),
         V_R (nb),
         V_I (nb),
         V_sq (nb),
         P_ij (na),
         Q_ij (na),
         S_tot_sq (na)].
        """
        nb = self.n_buses
        nl = self.n_lines
        na = self.n_arcs
        ng = self.n_gens
        buses_range = range(nb)

        # bus -> gen index mapping (in 0..ng-1)
        gen_sym_index_by_bus_idx = {}
        for bus_id, gen_idx in self.gen_index_by_bus.items():
            bus_idx = self.bus_index[bus_id]
            gen_sym_index_by_bus_idx[bus_idx] = gen_idx

        G_mat = self.G_mat
        B_mat = self.B_mat
        g_series = self.g_series
        b_series = self.b_series
        line_collection = self.line_collection
        arc_collection = self.arc_collection
        line_pos = {lid: k for k, lid in enumerate(self.line_ids)}
        

        P_D = self.P_D
        Q_D = self.Q_D

        if ref_bus_id is None:
            ref_bus_id = self.bus_ids[0]
        ref_idx = self.bus_index[ref_bus_id]

        def h_func(x):
            x = np.asarray(x, dtype=float)
            # unpack x according to variable_list structure
            idx = 0
            P_G = x[idx:idx + ng]
            idx += ng
            Q_G = x[idx:idx + ng]
            idx += ng
            V_R = x[idx:idx + nb]
            idx += nb
            V_I = x[idx:idx + nb]
            idx += nb
            V_sq = x[idx:idx + nb]
            idx += nb
            P_ij = x[idx:idx + na]
            idx += na
            Q_ij = x[idx:idx + na]
            idx += na
            S_tot_sq = x[idx:idx + na]
            # idx += na   # not needed afterwards

            residuals = []

            # 1) Active power balance c_i^P
            for i in buses_range:
                ViR = V_R[i]
                ViI = V_I[i]

                sum_GR_BI = 0.0
                sum_GI_BR = 0.0
                for j in buses_range:
                    VjR = V_R[j]
                    VjI = V_I[j]
                    Gij = G_mat[i, j]
                    Bij = B_mat[i, j]
                    sum_GR_BI += Gij * VjR - Bij * VjI
                    sum_GI_BR += Gij * VjI + Bij * VjR

                P_inj = ViR * sum_GR_BI + ViI * sum_GI_BR

                if i in gen_sym_index_by_bus_idx:
                    gi = gen_sym_index_by_bus_idx[i]
                    cP = P_G[gi] - P_D[i] - P_inj
                else:
                    cP = - P_D[i] - P_inj

                residuals.append(cP)

            # 2) Reactive power balance c_i^Q
            for i in buses_range:
                ViR = V_R[i]
                ViI = V_I[i]

                sum_GR_BI = 0.0
                sum_GI_BR = 0.0
                for j in buses_range:
                    VjR = V_R[j]
                    VjI = V_I[j]
                    Gij = G_mat[i, j]
                    Bij = B_mat[i, j]
                    sum_GR_BI += Gij * VjR - Bij * VjI
                    sum_GI_BR += Gij * VjI + Bij * VjR

                Q_inj = ViI * sum_GR_BI - ViR * sum_GI_BR

                if i in gen_sym_index_by_bus_idx:
                    gi = gen_sym_index_by_bus_idx[i]
                    cQ = Q_G[gi] - Q_D[i] - Q_inj
                else:
                    cQ = - Q_D[i] - Q_inj

                residuals.append(cQ)

            # 3) Branch power-flow definition constraints
            for a, (i, j) in enumerate(arc_collection):
                ViR = V_R[i]
                ViI = V_I[i]
                VjR = V_R[j]
                VjI = V_I[j]

                lid = self.arc_to_line[a]
                ell = line_pos[lid]

                g_ij = g_series[ell]
                b_ij = b_series[ell]

                # P_ij definition
                P_expr = (
                    ViR * (g_ij * (ViR - VjR) - b_ij * (ViI - VjI))
                    + ViI * (g_ij * (ViI - VjI) + b_ij * (ViR - VjR))
                )
                # Q_ij definition
                Q_expr = (
                    ViI * (g_ij * (ViR - VjR) - b_ij * (ViI - VjI))
                    - ViR * (g_ij * (ViI - VjI) + b_ij * (ViR - VjR))
                )

                cP_flow = P_ij[a] - P_expr
                cQ_flow = Q_ij[a] - Q_expr

                residuals.append(cP_flow)
                residuals.append(cQ_flow)

            # 4) Voltage magnitude definition c_i^{Vsq}
            for i in buses_range:
                cV = V_sq[i] - (V_R[i] ** 2 + V_I[i] ** 2)
                residuals.append(cV)

            # 5) Branch S_tot_sq definition c_{ij}^S
            for a in range(na):
                cS = S_tot_sq[a] - (P_ij[a] ** 2 + Q_ij[a] ** 2)
                residuals.append(cS)

            # 6) Reference bus constraint c^{ref} = V_I[ref_idx]
            c_ref = V_I[ref_idx]
            residuals.append(c_ref)

            return np.asarray(residuals, dtype=float)

        return h_func
    
    def update_lambda(self, x, alpha, h_func):
        """
        单步对偶更新: λ^{k+1} = λ^k + α h(x^{k+1}),
        并把新的 λ 写回到对象内部 (lambda_P_bal, lambda_Q_bal, ...).

        参数
        ----
        x       : list 或 1D array, 当前的 primal 解 x^{k+1}
        alpha   : float 或 1D array, 对偶步长 α_k
        h_func  : callable, 接受 x, 返回 1D array h(x)

        返回
        ----
        lambda_new  : 1D array, 更新后的 λ^{k+1} (扁平形式)
        h_x         : 1D array, 当前约束残差 h(x^{k+1})
        """
        x_vec = np.asarray(x, dtype=float)

        # 当前 λ（扁平）和约束残差
        lam = np.asarray(self.lambda_vec, dtype=float)
        h_x = np.asarray(h_func(x_vec), dtype=float)

        # 维度检查
        assert lam.shape == h_x.shape, f"lambda 维度 {lam.shape} 与 h(x) 维度 {h_x.shape} 不一致"

        # 对偶更新
        lambda_new = lam + alpha * h_x

        # ---- 把 lambda_new 拆回各组乘子，并写回到 self ----
        nb = self.n_buses
        na = self.n_arcs

        idx = 0
        # λ_P_bal (nb)
        self.lambda_P_bal = lambda_new[idx:idx + nb].tolist()
        idx += nb
        # λ_Q_bal (nb)
        self.lambda_Q_bal = lambda_new[idx:idx + nb].tolist()
        idx += nb
        # λ_P_flow (nl)
        self.lambda_P_flow = lambda_new[idx:idx + na].tolist()
        idx += na
        # λ_Q_flow (nl)
        self.lambda_Q_flow = lambda_new[idx:idx + na].tolist()
        idx += na
        # λ_Vsq (nb)
        self.lambda_Vsq = lambda_new[idx:idx + nb].tolist()
        idx += nb
        # λ_Ssq (nl)
        self.lambda_Ssq = lambda_new[idx:idx + na].tolist()
        idx += na
        # λ_ref (1)
        self.lambda_ref = float(lambda_new[idx])
        idx += 1

        assert idx == len(lambda_new), "lambda_new 拆分时长度对不上，检查顺序是否和 h_func 一致"

        # 更新扁平 lambda_vec，保证下一次用的是最新 λ
        self.lambda_vec = lambda_new.tolist()

        return lambda_new, h_x

    def check_constraints(self, solution, ref_bus_id=None, tol_eq=1e-6, tol_ineq=1e-9):
        """
        Check feasibility of all constraints for a given primal solution vector.

        Inputs
        ------
        solution : list / 1D array
            Primal decision vector in the SAME order as self.variable_list:
            [P_G (ng),
            Q_G (ng),
            V_R (nb),
            V_I (nb),
            V_sq (nb),
            P_ij (nl),
            Q_ij (nl),
            S_tot_sq (nl)].
        ref_bus_id : int or None
            Reference bus id (1-based). If None, use default (first bus).
        tol_eq : float
            Tolerance for equality constraints: |h(x)| <= tol_eq
        tol_ineq : float
            Tolerance for bound constraints: lb - tol_ineq <= x <= ub + tol_ineq

        Returns
        -------
        ok_list : list[bool]
            Boolean list for each constraint check, in the order:
            [ all equalities from h_func (in its stacking order),
            all box-bound checks for each variable (same order as variable_list) ]
        all_ok : bool
            True if all constraints are satisfied.
        """
        import numpy as np

        x = np.asarray(solution, dtype=float).flatten()

        # ---- dimension check ----
        nvar = len(self.variable_list)
        if x.size != nvar:
            raise ValueError(f"solution length mismatch: expected {nvar}, got {x.size}")

        ok_list = []

        # ---- (A) equality constraints: h(x) == 0 ----
        h_func = self.build_h_func(ref_bus_id=ref_bus_id)
        h = np.asarray(h_func(x), dtype=float).flatten()
        ok_eq = (np.abs(h) <= tol_eq).tolist()
        ok_list.extend(ok_eq)

        # ---- (B) box bounds: lb <= x <= ub ----
        for xi, (lb, ub) in zip(x.tolist(), self.Var_bound_list):
            ok_list.append((xi >= lb - tol_ineq) and (xi <= ub + tol_ineq))

        all_ok = all(ok_list)
        return ok_list, all_ok

def PrintQHDACOPFResults(model, solution):
    """
    根据 SympyACOPFModel 的解向量 solution（顺序与 model.variable_list 一致），
    打印类似传统 OPF 输出的结果表，包括：
    - 每个母线的 VR, VI, Pg, Qg, Pl, Ql, Qshunt
    - 每条支路的 Pik, Pki, Qik, Qki
    - 总有功/无功损耗、负荷供应比例
    """

    # -------- 1) 解向量拆包 --------
    x = np.asarray(solution, dtype=float).flatten()

    nb = model.n_buses
    nl = model.n_lines
    ng = model.n_gens
    na = model.n_arcs

    idx = 0
    P_G = x[idx:idx + ng]; idx += ng
    Q_G = x[idx:idx + ng]; idx += ng
    V_R = x[idx:idx + nb]; idx += nb
    V_I = x[idx:idx + nb]; idx += nb
    V_sq = x[idx:idx + nb]; idx += nb
    P_ij = x[idx:idx + na]; idx += na
    Q_ij = x[idx:idx + na]; idx += na
    S_tot_sq = x[idx:idx + na]; idx += na   # 这里暂时未直接使用

    # -------- 2) 一些方便别名 --------
    bus_ids   = model.bus_ids
    line_ids  = model.line_ids
    lines     = model.lines
    gen_ids   = model.gen_ids
    gens      = model.gens

    P_D = model.P_D
    Q_D = model.Q_D

    # gen id -> position in P_G / Q_G
    gen_index_by_id = {gid: k for k, gid in enumerate(gen_ids)}

    # -------- 3) 母线结果 --------
    print("BusID\tVR\tVI\tPg\tQg\tl\tPl\tQl\tQshunt\n")

    Pgtotal = 0.0
    Qgtotal = 0.0
    Ploadtotal = 0.0
    Qloadtotal = 0.0

    for bi, bus_id in enumerate(bus_ids):
        Vr = V_R[bi]
        Vi = V_I[bi]

        # 该母线所有机组出力求和
        Pg_i = 0.0
        Qg_i = 0.0
        for gid in gen_ids:
            if gens[gid][0] == bus_id:
                gi = gen_index_by_id[gid]
                Pg_i += P_G[gi]
                Qg_i += Q_G[gi]

        Pl_i = P_D[bi]
        Ql_i = Q_D[bi]
        Qsh_i = 0.0
        l_i = 1.0

        print(
            "{0:d}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}\t{7:.4f}\t{8:.4f}".format(
                bus_id, Vr, Vi, Pg_i, Qg_i, l_i, Pl_i, Ql_i, Qsh_i
            )
        )

        Pgtotal += Pg_i
        Qgtotal += Qg_i
        Ploadtotal += Pl_i
        Qloadtotal += Ql_i

    print("\n")
    print("TOTAL\t\t\t{0:.4f}\t{1:.4f}\t\t{2:.4f}\t{3:.4f}".format(
        Pgtotal, Qgtotal, Ploadtotal, Qloadtotal
    ))
    print("\n\n")

    # -------- 4) 支路潮流结果 --------
    print("Busi\tBusk\tPik\tPki\tQik\tQki")

    Ploss = 0.0
    Qloss = 0.0

    for lid in line_ids:
        from_bus, to_bus, _, _, _, _, _ = lines[lid]

        # 当前 line 对应的正反向 arc
        a_fwd = model.arc_index[(lid, +1)]
        a_rev = model.arc_index[(lid, -1)]

        Pik = P_ij[a_fwd]
        Pki = P_ij[a_rev]
        Qik = Q_ij[a_fwd]
        Qki = Q_ij[a_rev]

        print(
            "{0:d}\t{1:d}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}".format(
                from_bus, to_bus, Pik, Pki, Qik, Qki
            )
        )

        Ploss += (Pik + Pki)
        Qloss += (Qik + Qki)

    print("\n")
    print("Total Ploss: {0:.4f}".format(Ploss))
    print("Total Qloss: {0:.4f}".format(Qloss))

    # -------- 5) 负荷供应比例 --------
    if Ploadtotal > 1e-8:
        Pl_supplied = Pgtotal - Ploss
        perc_supplied = (Pl_supplied / Ploadtotal) * 100.0
    else:
        perc_supplied = 0.0

    print("Total Load Supplied: {0:.4f}%".format(perc_supplied))

def solve_with_gurobi_from_sympy(L_sym, variable_list, Var_bound_list, verbose=False):
    """
    用 Gurobi 求解一个只有 box 约束的二次目标：
        minimize L_sym(x)
        s.t. lb_i <= x_i <= ub_i

    参数
    ----
    L_sym : sympy.Expr
        以 variable_list 为变量的二次多项式目标
    variable_list : list[sympy.Symbol]
        变量顺序与 ALM 模型一致
    Var_bound_list : list[[lb, ub]]
        每个变量的 box 约束
    verbose : bool
        是否打印 Gurobi 的日志

    返回
    ----
    x_opt : np.ndarray
        最优解向量，顺序与 variable_list 对齐
    """

    nvar = len(variable_list)

    # 1) SymPy 多项式展开
    L_expanded = sp.expand(L_sym)
    poly = sp.Poly(L_expanded, variable_list)

    # 2) 建 Gurobi 模型
    model_g = gp.Model("linear_ALM_QP")
    if not verbose:
        model_g.Params.OutputFlag = 0  # 关掉 Gurobi 输出

    # 创建变量
    gurobi_vars = []
    for i in range(nvar):
        lb, ub = Var_bound_list[i]
        v = model_g.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"x_{i}")
        gurobi_vars.append(v)

    model_g.update()

    # 3) 构造目标函数：按 monomial 加项
    obj = 0.0

    # poly.terms() 返回 [ (exp_tuple, coeff), ... ]
    # exp_tuple 是每个变量的幂次，例如 (2,0,0,...) 表示 x0^2
    for monom, coeff in poly.terms():
        deg = sum(monom)
        coeff_val = float(coeff)

        if deg == 0:
            # 常数项
            obj += coeff_val

        elif deg == 1:
            # 线性项：某个变量幂次为 1，其余为 0
            idx = [i for i, e in enumerate(monom) if e == 1][0]
            obj += coeff_val * gurobi_vars[idx]

        elif deg == 2:
            # 二次项：可能是 x_i^2 或 x_i * x_j
            idxs = [i for i, e in enumerate(monom) if e > 0]
            if len(idxs) == 1:
                # x_i^2
                i0 = idxs[0]
                # e.g. coeff * x_i^2
                obj += coeff_val * gurobi_vars[i0] * gurobi_vars[i0]
            elif len(idxs) == 2:
                # x_i * x_j
                i0, i1 = idxs
                # monom 形如 x_i^1 x_j^1
                obj += coeff_val * gurobi_vars[i0] * gurobi_vars[i1]
            else:
                # 理论上不会出现，因为是二次目标
                raise ValueError(f"Unexpected quadratic term structure: {monom}")
        else:
            # 按设计 L_sym 是二次的，这里不应该出现 >=3 次的情况
            raise ValueError(f"Found degree-{deg} term in quadratic objective: {monom}")

    model_g.setObjective(obj, GRB.MINIMIZE)

    # 4) 求解
    model_g.optimize()

    if model_g.status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi did not find optimal solution, status={model_g.status}")

    # 5) 提取解
    x_opt = np.array([v.X for v in gurobi_vars], dtype=float)

    return x_opt