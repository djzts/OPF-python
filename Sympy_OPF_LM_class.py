import sympy as sp
import numpy as np

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

        # proximal 相关参数
        self.prox_center = None  # 当前 proximal 中心 x^k（numpy 向量）
        self.prox_gamma = 0.0    # proximal 权重 γ

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
        # sort ids to fix ordering
        self.bus_ids = sorted(self.buses.keys())
        self.line_ids = sorted(self.lines.keys())
        self.gen_ids = sorted(self.gens.keys())

        self.n_buses = len(self.bus_ids)
        self.n_lines = len(self.line_ids)
        self.n_gens = len(self.gen_ids)

        # maps: bus id -> bus index [0..n_buses-1]
        self.bus_index = {bid: i for i, bid in enumerate(self.bus_ids)}
        # maps: gen id -> gen index [0..n_gens-1]
        self.gen_index = {gid: i for i, gid in enumerate(self.gen_ids)}
        # map bus id -> generator index (if any)
        self.gen_index_by_bus = {}
        for gid, gdata in self.gens.items():
            bus_id = gdata[0]
            self.gen_index_by_bus[bus_id] = self.gen_index[gid]

        # build line -> (i_idx, j_idx) collection
        self.line_collection = []
        for lid in self.line_ids:
            fbus, tbus = self.lines[lid][0], self.lines[lid][1]
            i = self.bus_index[fbus]
            j = self.bus_index[tbus]
            self.line_collection.append((i, j))

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
    # Proximal 设置：方案 A 用到的 γ 和中心 x^k
    # ------------------------------------------------------------------
    def set_proximal(self, x_center=None, gamma=0.0):
        """
        设置（或关闭）proximal 项：
            (γ/2) * ||x - x_center||^2

        参数
        ----
        x_center : list / 1D array 或 None
            上一轮的解 x^k, 维度必须和 variable_list 一致。
            如果为 None 或 gamma <= 0, 则关闭 proximal 项。
        gamma : float
            proximal 权重 gamma > 0; 太大会步子太小, 太小则稳定性弱。
        """
        if x_center is None or gamma <= 0.0:
            # 关闭 proximal 项
            self.prox_center = None
            self.prox_gamma = 0.0
            return

        x_center = np.asarray(x_center, dtype=float).flatten()
        nvar = len(self.variable_list)
        if x_center.size != nvar:
            raise ValueError(
                f"x_center length mismatch: expected {nvar}, got {x_center.size}"
            )

        self.prox_center = x_center
        self.prox_gamma = float(gamma)

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
        nl = self.n_lines
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

        # Branch flows: P_ij, Q_ij, S_tot_sq
        self.P_ij = sp.symbols(f'P_ij0:{nl}')
        # thermal limit (per-unit MVA) from line data
        P_ij_bound = []
        for lid in self.line_ids:
            rate = self.lines[lid][6]
            P_ij_bound.append([-rate, rate])
        self.variable_list, self.Var_bound_list = self._var_list_insert(
            self.P_ij, P_ij_bound, self.variable_list, self.Var_bound_list
        )

        self.Q_ij = sp.symbols(f'Q_ij0:{nl}')
        Q_ij_bound = P_ij_bound.copy()
        self.variable_list, self.Var_bound_list = self._var_list_insert(
            self.Q_ij, Q_ij_bound, self.variable_list, self.Var_bound_list
        )

        self.S_tot_sq = sp.symbols(f'S_tot_sq0:{nl}')
        S_tot_sq_bound = []
        for lid in self.line_ids:
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
        nl = self.n_lines

        # total number of multipliers
        total_dim = 2 * nb + 2 * nl + nb + nl + 1
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
            self.lambda_P_flow = [scalar] * nl
            self.lambda_Q_flow = [scalar] * nl
            self.lambda_Vsq = [scalar] * nb
            self.lambda_Ssq = [scalar] * nl
            self.lambda_ref = scalar

        # ---------------------------------------------------
        # CASE 2: value is scalar
        # ---------------------------------------------------
        elif isinstance(value, (int, float)):
            scalar = float(value)

            self.lambda_P_bal = [scalar] * nb
            self.lambda_Q_bal = [scalar] * nb
            self.lambda_P_flow = [scalar] * nl
            self.lambda_Q_flow = [scalar] * nl
            self.lambda_Vsq = [scalar] * nb
            self.lambda_Ssq = [scalar] * nl
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
            self.lambda_P_flow = vec[idx:idx + nl].tolist()
            idx += nl

            self.lambda_Q_flow = vec[idx:idx + nl].tolist()
            idx += nl

            # Voltage magnitude definition
            self.lambda_Vsq = vec[idx:idx + nb].tolist()
            idx += nb

            # Branch S_sq definition
            self.lambda_Ssq = vec[idx:idx + nl].tolist()
            idx += nl

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
        nl = self.n_lines
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
        line_collection = self.line_collection

        P_D = self.P_D
        Q_D = self.Q_D

        # map bus index -> generator sympy index (or None)
        gen_sym_index_by_bus_idx = {}
        for bid, gen_idx in self.gen_index_by_bus.items():
            bus_idx = self.bus_index[bid]
            gen_sym_index_by_bus_idx[bus_idx] = gen_idx

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
            if i in gen_sym_index_by_bus_idx:
                gi = gen_sym_index_by_bus_idx[i]
                h_P = self.P_G[gi] - P_D[i] - P_inj
            else:
                # no generator at this bus
                h_P = - P_D[i] - P_inj

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

            if i in gen_sym_index_by_bus_idx:
                gi = gen_sym_index_by_bus_idx[i]
                h_Q = self.Q_G[gi] - Q_D[i] - Q_inj
            else:
                h_Q = - Q_D[i] - Q_inj

            L += self.lambda_Q_bal[i] * h_Q

        # ---------------------------
        # (3) Branch power-flow definition constraints
        # ---------------------------
        for ell, (i, j) in enumerate(line_collection):
            ViR = V_R[i]
            ViI = V_I[i]
            VjR = V_R[j]
            VjI = V_I[j]

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

            h_P_flow = self.P_ij[ell] - P_expr
            h_Q_flow = self.Q_ij[ell] - Q_expr

            L += self.lambda_P_flow[ell] * h_P_flow
            L += self.lambda_Q_flow[ell] * h_Q_flow

        # ---------------------------
        # (4) Voltage magnitude definition
        # ---------------------------
        for i in buses_range:
            h_Vsq = V_sq[i] - (V_R[i] ** 2 + V_I[i] ** 2)
            L += self.lambda_Vsq[i] * h_Vsq

        # ---------------------------
        # (5) Branch S_tot_sq definition
        # ---------------------------
        for ell in range(nl):
            h_Ssq = S_tot_sq[ell] - (P_ij[ell] ** 2 + Q_ij[ell] ** 2)
            L += self.lambda_Ssq[ell] * h_Ssq

        # ---------------------------
        # (6) Reference bus constraint: V_ref^I = 0
        # ---------------------------
        if ref_bus_id is None:
            # default: first bus id
            ref_bus_id = self.bus_ids[0]
        ref_idx = self.bus_index[ref_bus_id]
        h_ref = V_I[ref_idx]
        L += self.lambda_ref * h_ref

        # ---------------------------
        # (7) 方案 A: Proximal 二次正则项 (γ/2) * ||x - x_center||^2
        # ---------------------------
        if self.prox_center is not None and self.prox_gamma > 0.0:
            gamma = self.prox_gamma
            x_syms = self.variable_list
            x0 = self.prox_center
            # 注意：x0 是数值常数，sympy 里 (sym - x0_i)**2 仍然是二次多项式
            for sym, x0_i in zip(x_syms, x0):
                L += 0.5 * gamma * (sym - x0_i) ** 2

        self.Lagrange = L
        return L

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
         P_ij (nl),
         Q_ij (nl),
         S_tot_sq (nl)].
        """
        nb = self.n_buses
        nl = self.n_lines
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
            P_ij = x[idx:idx + nl]
            idx += nl
            Q_ij = x[idx:idx + nl]
            idx += nl
            S_tot_sq = x[idx:idx + nl]
            # idx += nl   # not needed afterwards

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
            for ell, (i, j) in enumerate(line_collection):
                ViR = V_R[i]
                ViI = V_I[i]
                VjR = V_R[j]
                VjI = V_I[j]

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

                cP_flow = P_ij[ell] - P_expr
                cQ_flow = Q_ij[ell] - Q_expr

                residuals.append(cP_flow)
                residuals.append(cQ_flow)

            # 4) Voltage magnitude definition c_i^{Vsq}
            for i in buses_range:
                cV = V_sq[i] - (V_R[i] ** 2 + V_I[i] ** 2)
                residuals.append(cV)

            # 5) Branch S_tot_sq definition c_{ij}^S
            for ell in range(nl):
                cS = S_tot_sq[ell] - (P_ij[ell] ** 2 + Q_ij[ell] ** 2)
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
        nl = self.n_lines

        idx = 0
        # λ_P_bal (nb)
        self.lambda_P_bal = lambda_new[idx:idx + nb].tolist()
        idx += nb
        # λ_Q_bal (nb)
        self.lambda_Q_bal = lambda_new[idx:idx + nb].tolist()
        idx += nb
        # λ_P_flow (nl)
        self.lambda_P_flow = lambda_new[idx:idx + nl].tolist()
        idx += nl
        # λ_Q_flow (nl)
        self.lambda_Q_flow = lambda_new[idx:idx + nl].tolist()
        idx += nl
        # λ_Vsq (nb)
        self.lambda_Vsq = lambda_new[idx:idx + nb].tolist()
        idx += nb
        # λ_Ssq (nl)
        self.lambda_Ssq = lambda_new[idx:idx + nl].tolist()
        idx += nl
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

import numpy as np

def PrintQHDACOPFResults(model, solution):
    """
    根据 SympyACOPFModel 的解向量 solution（顺序与 model.variable_list 一致），
    打印类似传统 OPF 输出的结果表，包括：
    - 每个母线的 VR, VI, Pg, Qg, Pl, Ql, Qshunt
    - 每条支路的 Pik, Pki, Qik, Qki
    - 总有功/无功损耗、负荷供应比例
    """

    # -------- 1. 解向量拆包（顺序来自 build_h_func 的说明） --------
    x = np.asarray(solution, dtype=float).flatten()

    nb = model.n_buses
    nl = model.n_lines
    ng = model.n_gens

    idx = 0
    P_G = x[idx:idx + ng]; idx += ng
    Q_G = x[idx:idx + ng]; idx += ng
    V_R = x[idx:idx + nb]; idx += nb
    V_I = x[idx:idx + nb]; idx += nb
    V_sq = x[idx:idx + nb]; idx += nb
    P_ij = x[idx:idx + nl]; idx += nl
    Q_ij = x[idx:idx + nl]; idx += nl
    S_tot_sq = x[idx:idx + nl]  # 最后这一段这里暂时不用，但可以保留

    # 一些方便的别名
    bus_ids   = model.bus_ids        # 按内部顺序排序好的 bus id 列表
    line_ids  = model.line_ids       # 按内部顺序排序好的 line id 列表
    buses     = model.buses          # 原始 bus 数据
    lines     = model.lines          # 原始 line 数据
    gen_ids   = model.gen_ids
    gens      = model.gens
    bus_index = model.bus_index      # bus_id -> 0..nb-1

    P_D = model.P_D                  # 每个母线的有功负荷（p.u.）
    Q_D = model.Q_D                  # 每个母线的无功负荷（p.u.）

    g_series = model.g_series        # 每条线的串联导纳实部 g_ij
    b_series = model.b_series        # 每条线的串联导纳虚部 b_ij
    line_collection = model.line_collection  # [(i_idx, j_idx)]，与 line_ids 对齐

    # 预计算：每个 gen 在 P_G/Q_G 中的索引
    gen_index_by_id = {gid: k for k, gid in enumerate(gen_ids)}

    # -------- 2. 母线结果表 --------
    print("BusID\tVR\tVI\tPg\tQg\tl\tPl\tQl\tQshunt\n")

    Pgtotal = 0.0
    Qgtotal = 0.0
    Ploadtotal = 0.0
    Qloadtotal = 0.0

    for bi, bus_id in enumerate(bus_ids):
        # 电压
        Vr = V_R[bi]
        Vi = V_I[bi]

        # 该母线上的总发电功率（允许多个机组的话就求和）
        Pg_i = 0.0
        Qg_i = 0.0
        for gid in gen_ids:
            gdata  = gens[gid]
            g_bus  = gdata[0]
            if g_bus == bus_id:
                gi = gen_index_by_id[gid]
                Pg_i += P_G[gi]
                Qg_i += Q_G[gi]

        # 负荷（Pl, Ql）
        Pl_i = P_D[bi]
        Ql_i = Q_D[bi]

        # Qshunt：这里简单设为 0.0，如果你以后要考虑 shunt，可在此计算
        Qsh_i = 0.0

        # l 列（load factor / 状态标志），按你示例直接打印 1.0
        l_i = 1.0

        print(
            "{0:d}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}\t{7:.4f}\t{8:.4f}"
            .format(bus_id, Vr, Vi, Pg_i, Qg_i, l_i, Pl_i, Ql_i, Qsh_i)
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

    # -------- 3. 支路潮流表：Pik, Pki, Qik, Qki --------
    print("Busi\tBusk\tPik\tPki\tQik\tQki")

    Ploss = 0.0
    Qloss = 0.0

    for ell, lid in enumerate(line_ids):
        # 线数据：from_bus, to_bus, r, x, bsh, tap, rate
        from_bus, to_bus, r, x, bsh, tap, rate = lines[lid]
        i_idx, j_idx = line_collection[ell]  # 0-based index in V_R / V_I

        ViR = V_R[i_idx]
        ViI = V_I[i_idx]
        VjR = V_R[j_idx]
        VjI = V_I[j_idx]

        g_ij = g_series[ell]
        b_ij = b_series[ell]

        # 我们的 P_ij/Q_ij 存的是 from_bus -> to_bus 的潮流（i→j）
        Pik = P_ij[ell]
        Qik = Q_ij[ell]

        # 反向潮流 Pki, Qki 用同样公式、交换 i 和 j 计算
        Pki = (
            VjR * (g_ij * (VjR - ViR) - b_ij * (VjI - ViI))
            + VjI * (g_ij * (VjI - ViI) + b_ij * (VjR - ViR))
        )
        Qki = (
            VjI * (g_ij * (VjR - ViR) - b_ij * (VjI - ViI))
            - VjR * (g_ij * (VjI - ViI) + b_ij * (VjR - ViR))
        )

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

    # 负荷供应比例：利用 Pgtotal - Ploss 估算实际供电
    if Ploadtotal > 1e-8:
        Pl_supplied = Pgtotal - Ploss
        perc_supplied = (Pl_supplied / Ploadtotal) * 100.0
    else:
        perc_supplied = 0.0

    print("Total Load Supplied: {0:.4f}%".format(perc_supplied))