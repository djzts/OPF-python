import sympy as sp
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import os
from datetime import datetime
from pathlib import Path

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
        self.arc_collection_single = []
        self.arc_to_line = []

        for lid in self.line_ids:
            fbus, tbus = self.lines[lid][0], self.lines[lid][1]
            i = self.bus_index[fbus]
            j = self.bus_index[tbus]

            # forward arc
            self.arc_ids.append((lid, +1))
            self.arc_collection.append((i, j))
            self.arc_collection_single.append((i, j))
            self.arc_to_line.append(lid)

            # reverse arc
            self.arc_ids.append((lid, -1))
            self.arc_collection.append((j, i))
            self.arc_to_line.append(lid)

        self.n_arcs = len(self.arc_collection)
        self.arc_index = {arc_id: k for k, arc_id in enumerate(self.arc_ids)}

    def build_initial_x0(self):
        """
        Construct a simple and general initial point x^0, following the order in self.variable_list:
        [P_G (ng), Q_G (ng), Wii_RR(nb), Wii_RI(nb), Wii_II(nb), 
        Wij_RR, Wij_RI, Wij_IR, Wij_II, V_sq (nb), P_ij (na), Q_ij (na), S_tot_sq (na)].

        Design principles:
        - P_G: Allocate the total active power load sum(P_D) in proportion to Pmax,
        then clip it to [Pmin, Pmax]
        - Q_G: Allocate the total reactive power load sum(Q_D) in proportion to P_G,
        then clip it to [Qmin, Qmax]
        - Build a temporary rectangular voltage seed from bus Vm and Va
        - V_sq: = Wii_RR + Wii_II
        - P_ij, Q_ij, S_tot_sq: Initialize all to 0
        """

        nb = self.n_buses
        ng = self.n_gens
        na = self.n_arcs
        ncross = self.n_cross_pairs

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

        # -------- 3) Bus voltages: Wii_RR, Wii_RI, Wii_II --------
        V_R_seed = np.zeros(nb)
        V_I_seed = np.zeros(nb)
        for bid in self.bus_ids:
            bdata = self.buses[bid]
            bus_idx = self.bus_index[bid]
            Vm = float(bdata[2])
            Va_deg = float(bdata[3])
            Va_rad = Va_deg * np.pi / 180.0
            V_R_seed[bus_idx] = Vm * np.cos(Va_rad)
            V_I_seed[bus_idx] = Vm * np.sin(Va_rad)

        Wii_RR0 = V_R_seed ** 2
        Wii_RI0 = V_R_seed * V_I_seed
        Wii_II0 = V_I_seed ** 2

        Wij_RR0 = np.zeros(ncross)
        Wij_RI0 = np.zeros(ncross)
        Wij_IR0 = np.zeros(ncross)
        Wij_II0 = np.zeros(ncross)
        for (i, j), p in self.cross_pair_index.items():
            Wij_RR0[p] = V_R_seed[i] * V_R_seed[j]
            Wij_RI0[p] = V_R_seed[i] * V_I_seed[j]
            Wij_IR0[p] = V_I_seed[i] * V_R_seed[j]
            Wij_II0[p] = V_I_seed[i] * V_I_seed[j]

        # 剪裁到变量给定的 box bounds（防止数据比较奇怪时越界）
        V_sq0 = Wii_RR0 + Wii_II0
        V_sq_lb = np.array([0.9 ** 2] * nb)
        V_sq_ub = np.array([1.1 ** 2] * nb)
        V_sq0 = np.minimum(np.maximum(V_sq0, V_sq_lb), V_sq_ub)

        # -------- 4) Branch flows: P_ij, Q_ij, S_tot_sq --------
        P_ij0 = np.zeros(na)
        Q_ij0 = np.zeros(na)
        S_tot_sq0 = np.zeros(na)

        # S_tot_sq 上界来自每条线的 rate^2
        Ssq_lb = np.zeros(na)
        Ssq_ub = np.array([self.lines[lid][6] ** 2 for lid in self.arc_to_line], dtype=float)
        S_tot_sq0 = np.minimum(np.maximum(S_tot_sq0, Ssq_lb), Ssq_ub)

        # -------- 5) 按 variable_list 顺序拼接成 x0 --------
        x0 = np.concatenate([
            P_G0,
            Q_G0,
            Wii_RR0,
            Wii_RI0,
            Wii_II0,
            Wij_RR0,
            Wij_RI0,
            Wij_IR0,
            Wij_II0,
            V_sq0,
            P_ij0,
            Q_ij0,
            S_tot_sq0,
        ])

        assert x0.size == len(self.variable_list), (
            f"Constructed x0 length {x0.size} != variable_list length {len(self.variable_list)}"
        )
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
    def _build_cross_pair_sets(self):
        """
        Build the sparse set of undirected bus pairs {i,j} that actually need
        lifted cross variables.

        We include pairs touched by:
          1) branch-flow equations,
          2) nonzero Ybus off-diagonal terms in bus-balance equations.
        """
        nb = self.n_buses
        cross_pair_set = set()

        # (A) branch pairs used in branch-flow equations
        for (i, j) in self.arc_collection:
            if i != j:
                cross_pair_set.add(tuple(sorted((i, j))))

        # (B) off-diagonal Ybus pairs used in bus-balance equations
        for i in range(nb):
            for j in range(nb):
                if i == j:
                    continue
                if abs(self.G_mat[i, j]) > 1e-12 or abs(self.B_mat[i, j]) > 1e-12:
                    cross_pair_set.add(tuple(sorted((i, j))))

        self.cross_pairs = sorted(cross_pair_set)
        self.n_cross_pairs = len(self.cross_pairs)
        self.cross_pair_index = {
            pair: k for k, pair in enumerate(self.cross_pairs)
        }

        self.conjugate_cross_pairs = []
        self.n_conjugate_cross_pairs = 0

        # optional: per-bus neighbor list for faster balance loops
        self.power_balance_pairs_by_bus = {i: [] for i in range(nb)}
        for (i, j) in self.cross_pairs:
            if abs(self.G_mat[i, j]) > 1e-12 or abs(self.B_mat[i, j]) > 1e-12:
                self.power_balance_pairs_by_bus[i].append(j)
            if abs(self.G_mat[j, i]) > 1e-12 or abs(self.B_mat[j, i]) > 1e-12:
                self.power_balance_pairs_by_bus[j].append(i)

    def _get_constraint_blocks(self):
        """
        Return the ordered equality-constraint blocks.

        Each entry is a tuple:
            (name, size, linearize_in_linear_alm)
        """
        return [
            ("P_bal", self.n_buses, False),
            ("Q_bal", self.n_buses, False),
            ("P_flow", self.n_arcs, False),
            ("Q_flow", self.n_arcs, False),
            ("Vsq", self.n_buses, False),
            ("Ssq", self.n_arcs, True),
            ("ref", 1, False),
        ]

    def _get_constraint_block_slices(self):
        """
        Return {block_name: (start, end, linearize_flag)} using the canonical
        stacking order shared by h(x), lambda, and the ALM builders.
        """
        block_slices = {}
        start = 0
        for name, size, linearize_flag in self._get_constraint_blocks():
            end = start + size
            block_slices[name] = (start, end, linearize_flag)
            start = end
        return block_slices

    @staticmethod
    def _var_list_insert(var_list, bound_list, variable_list, Var_bound_list):
        for v, bnd in zip(var_list, bound_list):
            variable_list.append(v)
            Var_bound_list.append(list(bnd))
        return variable_list, Var_bound_list
    
    # ------------------------------------------------------------------
    # pair helper / unpack
    # ------------------------------------------------------------------
    def _cross_flat(self, i, j):
        return self.cross_pair_index[tuple(sorted((i, j)))]

    def _cross_terms(self, i, j, rr_vals, ri_vals, ir_vals, ii_vals):
        p = self._cross_flat(i, j)
        rr = rr_vals[p]
        ii = ii_vals[p]
        if i <= j:
            ri = ri_vals[p]
            ir = ir_vals[p]
        else:
            ri = ir_vals[p]
            ir = ri_vals[p]
        return rr, ri, ir, ii

    def _unpack_x(self, x):
        x = np.asarray(x, dtype=float).flatten()

        nb = self.n_buses
        ng = self.n_gens
        na = self.n_arcs
        ncross = self.n_cross_pairs

        idx = 0
        P_G = x[idx:idx + ng]; idx += ng
        Q_G = x[idx:idx + ng]; idx += ng

        Wii_RR = x[idx:idx + nb]; idx += nb
        Wii_RI = x[idx:idx + nb]; idx += nb
        Wii_II = x[idx:idx + nb]; idx += nb

        Wij_RR = x[idx:idx + ncross]; idx += ncross
        Wij_RI = x[idx:idx + ncross]; idx += ncross
        Wij_IR = x[idx:idx + ncross]; idx += ncross
        Wij_II = x[idx:idx + ncross]; idx += ncross

        V_sq = x[idx:idx + nb]; idx += nb
        P_ij = x[idx:idx + na]; idx += na
        Q_ij = x[idx:idx + na]; idx += na
        S_tot_sq = x[idx:idx + na]; idx += na

        return {
            "P_G": P_G,
            "Q_G": Q_G,
            "Wii_RR": Wii_RR,
            "Wii_RI": Wii_RI,
            "Wii_II": Wii_II,
            "Wij_RR": Wij_RR,
            "Wij_RI": Wij_RI,
            "Wij_IR": Wij_IR,
            "Wij_II": Wij_II,
            "V_sq": V_sq,
            "P_ij": P_ij,
            "Q_ij": Q_ij,
            "S_tot_sq": S_tot_sq,
        }

    def recover_voltages_from_lifted(self, x=None, vals=None, vr_nonnegative=True, tol=1e-10):
        """
        Reconstruct rectangular bus voltages from diagonal lifted variables.

        Parameters
        ----------
        x : array-like or None
            Flat primal vector in self.variable_list order.
        vals : dict or None
            Pre-unpacked variable dictionary from _unpack_x(x). If provided,
            this is used directly.
        vr_nonnegative : bool
            If True, choose V_R[i] >= 0 by default.
        tol : float
            Numerical tolerance used when diagonal lifted terms are nearly zero.

        Returns
        -------
        dict
            {
                "V_R": np.ndarray,
                "V_I": np.ndarray,
                "V_mag": np.ndarray,
            }
        """
        if vals is None:
            if x is None:
                raise ValueError("Either x or vals must be provided.")
            vals = self._unpack_x(x)

        Wii_RR = np.asarray(vals["Wii_RR"], dtype=float)
        Wii_RI = np.asarray(vals["Wii_RI"], dtype=float)
        Wii_II = np.asarray(vals["Wii_II"], dtype=float)
        V_sq = np.asarray(vals["V_sq"], dtype=float)

        nb = self.n_buses
        V_R = np.zeros(nb, dtype=float)
        V_I = np.zeros(nb, dtype=float)

        for i in range(nb):
            vr_mag = float(np.sqrt(max(Wii_RR[i], 0.0)))
            if not vr_nonnegative:
                vr_mag = -vr_mag
            V_R[i] = vr_mag

            if abs(vr_mag) > tol:
                V_I[i] = Wii_RI[i] / vr_mag
            else:
                vi_mag = float(np.sqrt(max(Wii_II[i], 0.0)))
                if Wii_RI[i] < -tol:
                    V_I[i] = -vi_mag
                else:
                    V_I[i] = vi_mag

        V_mag = np.sqrt(np.maximum(V_sq, 0.0))
        return {
            "V_R": V_R,
            "V_I": V_I,
            "V_mag": V_mag,
        }

    def postprocess_solution(self, x, vr_nonnegative=True, tol=1e-10):
        """
        Build a postprocessed solution dictionary that augments unpacked lifted
        variables with recovered rectangular voltages.
        """
        vals = self._unpack_x(x)
        vals.update(
            self.recover_voltages_from_lifted(
                vals=vals,
                vr_nonnegative=vr_nonnegative,
                tol=tol,
            )
        )
        return vals

    def _build_variables(self):
        nb = self.n_buses
        ng = self.n_gens
        na = self.n_arcs

        self._build_cross_pair_sets()
        ncross = self.n_cross_pairs

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

        # Diagonal lifted terms: only 3 per bus
        self.Wii_RR = sp.symbols(f'Wii_RR0:{nb}')
        self.Wii_RI = sp.symbols(f'Wii_RI0:{nb}')
        self.Wii_II = sp.symbols(f'Wii_II0:{nb}')

        diag_bound_sq = [[0, 1.21] for _ in range(nb)]
        diag_bound_bi_linear = [[-1.21, 1.21] for _ in range(nb)]
        self.variable_list, self.Var_bound_list = self._var_list_insert(
            self.Wii_RR, diag_bound_sq, self.variable_list, self.Var_bound_list
        )
        self.variable_list, self.Var_bound_list = self._var_list_insert(
            self.Wii_RI, diag_bound_bi_linear, self.variable_list, self.Var_bound_list
        )
        self.variable_list, self.Var_bound_list = self._var_list_insert(
            self.Wii_II, diag_bound_sq, self.variable_list, self.Var_bound_list
        )

        # Directed cross lifted terms: 4 per cross pair
        self.Wij_RR = sp.symbols(f'Wij_RR0:{ncross}')
        self.Wij_RI = sp.symbols(f'Wij_RI0:{ncross}')
        self.Wij_IR = sp.symbols(f'Wij_IR0:{ncross}')
        self.Wij_II = sp.symbols(f'Wij_II0:{ncross}')

        cross_bound_R = [[0, 1.21] for _ in range(ncross)]
        cross_bound_I = [[-1.21, 1.21] for _ in range(ncross)]
        self.variable_list, self.Var_bound_list = self._var_list_insert(
            self.Wij_RR, cross_bound_R, self.variable_list, self.Var_bound_list
        )
        self.variable_list, self.Var_bound_list = self._var_list_insert(
            self.Wij_RI, cross_bound_I, self.variable_list, self.Var_bound_list
        )
        self.variable_list, self.Var_bound_list = self._var_list_insert(
            self.Wij_IR, cross_bound_I, self.variable_list, self.Var_bound_list
        )
        self.variable_list, self.Var_bound_list = self._var_list_insert(
            self.Wij_II, cross_bound_I, self.variable_list, self.Var_bound_list
        )

        # Voltage magnitude squared: V_sq
        self.V_sq = sp.symbols(f'V_sq0:{nb}')
        V_sq_bound = [[0.9 ** 2, 1.1 ** 2] for _ in range(nb)]
        self.variable_list, self.Var_bound_list = self._var_list_insert(
            self.V_sq, V_sq_bound, self.variable_list, self.Var_bound_list
        )

        # thermal limit (per-unit MVA) from line data
        self.P_ij = sp.symbols(f'P_ij0:{na}')
        P_ij_bound = [[-self.lines[lid][6], self.lines[lid][6]] for lid in self.arc_to_line]
        self.variable_list, self.Var_bound_list = self._var_list_insert(
            self.P_ij, P_ij_bound, self.variable_list, self.Var_bound_list
        )

        self.Q_ij = sp.symbols(f'Q_ij0:{na}')
        self.variable_list, self.Var_bound_list = self._var_list_insert(
            self.Q_ij, P_ij_bound.copy(), self.variable_list, self.Var_bound_list
        )

        self.S_tot_sq = sp.symbols(f'S_tot_sq0:{na}')
        S_tot_sq_bound = [[0.0, self.lines[lid][6] ** 2] for lid in self.arc_to_line]
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
        block_slices = self._get_constraint_block_slices()
        total_dim = max(end for _, (start, end, _) in block_slices.items())

        # ---------------------------------------------------
        # CASE 1: value is None → set all to 0
        # ---------------------------------------------------
        if value is None:
            vec = np.zeros(total_dim, dtype=float)

        # ---------------------------------------------------
        # CASE 2: value is scalar
        # ---------------------------------------------------
        elif isinstance(value, (int, float)):
            vec = np.full(total_dim, float(value), dtype=float)

        # ---------------------------------------------------
        # CASE 3: value is vector
        # ---------------------------------------------------
        else:
            vec = np.asarray(value, dtype=float).flatten()
            if len(vec) != total_dim:
                raise ValueError(f"Lambda vector length mismatch. Expected {total_dim}, got {len(vec)}.")

        # ---------------------------------------------------
        # rebuild lambda_vec in correct order
        # ---------------------------------------------------
        idx = 0
        self.lambda_P_bal = vec[idx:idx + self.n_buses].tolist(); idx += self.n_buses
        self.lambda_Q_bal = vec[idx:idx + self.n_buses].tolist(); idx += self.n_buses
        self.lambda_P_flow = vec[idx:idx + self.n_arcs].tolist(); idx += self.n_arcs
        self.lambda_Q_flow = vec[idx:idx + self.n_arcs].tolist(); idx += self.n_arcs
        self.lambda_Vsq = vec[idx:idx + self.n_buses].tolist(); idx += self.n_buses
        self.lambda_Ssq = vec[idx:idx + self.n_arcs].tolist(); idx += self.n_arcs
        self.lambda_W_conj = []
        self.lambda_W_diag_rank = []
        self.lambda_W_cross_rr = []
        self.lambda_W_cross_ri = []
        self.lambda_W_cross_ir = []
        self.lambda_W_cross_ii = []
        self.lambda_W_cross_mix = []
        self.lambda_ref = float(vec[idx]); idx += 1

        self.lambda_vec = [
            *self.lambda_P_bal,
            *self.lambda_Q_bal,
            *self.lambda_P_flow,
            *self.lambda_Q_flow,
            *self.lambda_Vsq,
            *self.lambda_Ssq,
            self.lambda_ref,
        ]

                        
    # ------------------------------------------------------------------
    # Build Lagrangian (without quadratic penalty)
    # ------------------------------------------------------------------
    def build_objective_expr(self):
        obj = 0
        for gi, gid in enumerate(self.gen_ids):
            a = self.gens[gid][5]
            b = self.gens[gid][6]
            c = self.gens[gid][7]
            obj += 0.5 * a * self.P_G[gi] ** 2 + b * self.P_G[gi] + c
        self.objective = obj
        return obj

    def _diag_RR(self, i, vals):
        return vals[i]

    def _diag_RI(self, i, vals):
        return vals[i]

    def _diag_II(self, i, vals):
        return vals[i]

    def _cross_RR(self, i, j, vals):
        p = self._cross_flat(i, j)
        return vals[p]

    def _cross_RI(self, i, j, vals):
        p = self._cross_flat(i, j)
        return vals[p] if i <= j else self._cross_IR(i, j, vals)

    def _cross_IR(self, i, j, vals):
        p = self._cross_flat(i, j)
        return vals[p] if i <= j else self._cross_RI(j, i, vals)

    def _cross_II(self, i, j, vals):
        p = self._cross_flat(i, j)
        return vals[p]
    
    def _build_lagrangian(self, ref_bus_id=None):
        """
        Build the (classical) Lagrangian:

            L(x, λ) = f(x) + λ^T h(x)

        using the current values of the Lagrange multipliers stored in the object.
        """
        nb = self.n_buses
        na = self.n_arcs
        ng = self.n_gens

        L = self.build_objective_expr()

        # Convenience aliases
        Wii_RR = self.Wii_RR
        Wii_RI = self.Wii_RI
        Wii_II = self.Wii_II
        Wij_RR = self.Wij_RR
        Wij_RI = self.Wij_RI
        Wij_IR = self.Wij_IR
        Wij_II = self.Wij_II
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
        for i in range(nb):
            PG_sum = 0
            if i in gen_sym_indices_by_bus_idx:
                for gi in gen_sym_indices_by_bus_idx[i]:
                    PG_sum += self.P_G[gi]
            h_P = PG_sum - self.P_D[i]
            # diagonal Y_ii contribution
            Gii = self.G_mat[i, i]
            Bii = self.B_mat[i, i]
            h_P -= Gii * Wii_RR[i] + Gii * Wii_II[i]
            for j in self.power_balance_pairs_by_bus[i]:
                Gij = self.G_mat[i, j]
                Bij = self.B_mat[i, j]
                Wij_RR_ij, Wij_RI_ij, Wij_IR_ij, Wij_II_ij = self._cross_terms(i, j, Wij_RR, Wij_RI, Wij_IR, Wij_II)
                h_P -= Gij * Wij_RR_ij - Bij * Wij_RI_ij + Gij * Wij_II_ij + Bij * Wij_IR_ij
            L += self.lambda_P_bal[i] * h_P

        # ---------------------------
        # (2) Reactive power balance constraints
        # ---------------------------
        for i in range(nb):
            QG_sum = 0
            if i in gen_sym_indices_by_bus_idx:
                for gi in gen_sym_indices_by_bus_idx[i]:
                    QG_sum += self.Q_G[gi]
            h_Q = QG_sum - self.Q_D[i]
            Gii = self.G_mat[i, i]
            Bii = self.B_mat[i, i]
            h_Q += Bii * Wii_II[i] + Bii * Wii_RR[i]
            for j in self.power_balance_pairs_by_bus[i]:
                Gij = self.G_mat[i, j]
                Bij = self.B_mat[i, j]
                Wij_RR_ij, Wij_RI_ij, Wij_IR_ij, Wij_II_ij = self._cross_terms(i, j, Wij_RR, Wij_RI, Wij_IR, Wij_II)
                h_Q -= Gij * Wij_IR_ij - Bij * Wij_II_ij - Gij * Wij_RI_ij - Bij * Wij_RR_ij
            L += self.lambda_Q_bal[i] * h_Q

        # ---------------------------
        # (3) Branch power-flow definition constraints
        # ---------------------------
        for a, (i, j) in enumerate(self.arc_collection):
            lid = self.arc_to_line[a]
            ell = self.line_pos[lid]
            g_ij = self.g_series[ell]
            b_ij = self.b_series[ell]
            Wij_RR_ij, Wij_RI_ij, Wij_IR_ij, Wij_II_ij = self._cross_terms(i, j, Wij_RR, Wij_RI, Wij_IR, Wij_II)

            P_expr = g_ij * (Wii_RR[i] + Wii_II[i] - Wij_RR_ij - Wij_II_ij) + b_ij * (Wij_RI_ij - Wij_IR_ij)
            Q_expr = -b_ij * (Wii_RR[i] + Wii_II[i]) + b_ij * (Wij_RR_ij + Wij_II_ij) + g_ij * (Wij_RI_ij - Wij_IR_ij)

            L += self.lambda_P_flow[a] * (P_ij[a] - P_expr)
            L += self.lambda_Q_flow[a] * (Q_ij[a] - Q_expr)

        # ---------------------------
        # (4) Voltage magnitude definition
        # ---------------------------
        for i in range(nb):
            L += self.lambda_Vsq[i] * (V_sq[i] - (Wii_RR[i] + Wii_II[i]))

        # ---------------------------
        # (5) Branch S_tot_sq definition
        # ---------------------------
        for a in range(na):
            L += self.lambda_Ssq[a] * (S_tot_sq[a] - (P_ij[a] ** 2 + Q_ij[a] ** 2))

        # ---------------------------
        # (6) Reference bus constraint: V_ref^I = 0
        # ---------------------------
        if ref_bus_id is None:
            ref_bus_id = self.bus_ids[0]
        ref_idx = self.bus_index[ref_bus_id]
        L += self.lambda_ref * self.Wii_II[ref_idx]

        self.Lagrange = sp.expand(L)
        return self.Lagrange

    def build_h_symbolic_legacy(self, ref_bus_id=None):
        """
        构造符号形式的约束列表 h_sym(x)，顺序必须和 build_h_func 完全一致：
        [ c_i^P (∀ buses),
          c_i^Q (∀ buses),
          c_{ij}^{P,flow} (∀ lines),
          c_{ij}^{Q,flow} (∀ lines),
          c_i^{Vsq} (∀ buses),
          c_{ij}^S (∀ lines),
          c^{W,cons},
          c^{ref} ].
        """
        nb = self.n_buses
        na = self.n_arcs
        buses_range = range(nb)

        # 符号变量别名
        P_G = self.P_G
        Q_G = self.Q_G
        Wii_RR = self.Wii_RR
        Wii_RI = self.Wii_RI
        Wii_II = self.Wii_II
        Wij_RR = self.Wij_RR
        Wij_RI = self.Wij_RI
        Wij_IR = self.Wij_IR
        Wij_II = self.Wij_II
        V_sq = self.V_sq
        P_ij = self.P_ij
        Q_ij = self.Q_ij
        S_tot_sq = self.S_tot_sq

        if ref_bus_id is None:
            ref_bus_id = self.bus_ids[0]
        ref_idx = self.bus_index[ref_bus_id]

        residuals = []

        # bus -> gen 索引 (0..ng-1)


        # 1) Active power balance c_i^P
        for i in range(nb):
            PG_sum = 0
            # bus -> gen 索引 (0..ng-1)
            for bid, idx_list in self.gen_indices_by_bus.items():
                if self.bus_index[bid] == i:
                    for gi in idx_list:
                        PG_sum += P_G[gi]
            cP = PG_sum - self.P_D[i]
            Gii = self.G_mat[i, i]
            Bii = self.B_mat[i, i]
            cP -= Gii * Wii_RR[i] + Gii * Wii_II[i]
            for j in self.power_balance_pairs_by_bus[i]:
                Gij = self.G_mat[i, j]
                Bij = self.B_mat[i, j]
                Wij_RR_ij, Wij_RI_ij, Wij_IR_ij, Wij_II_ij = self._cross_terms(i, j, Wij_RR, Wij_RI, Wij_IR, Wij_II)
                cP -= Gij * Wij_RR_ij - Bij * Wij_RI_ij + Gij * Wij_II_ij + Bij * Wij_IR_ij
            residuals.append(cP)

        # 2) Reactive power balance c_i^Q
        for i in range(nb):
            QG_sum = 0
            for bid, idx_list in self.gen_indices_by_bus.items():
                if self.bus_index[bid] == i:
                    for gi in idx_list:
                        QG_sum += Q_G[gi]
            cQ = QG_sum - self.Q_D[i]
            Gii = self.G_mat[i, i]
            Bii = self.B_mat[i, i]
            cQ += Bii * Wii_II[i] + Bii * Wii_RR[i]
            for j in self.power_balance_pairs_by_bus[i]:
                Gij = self.G_mat[i, j]
                Bij = self.B_mat[i, j]
                Wij_RR_ij, Wij_RI_ij, Wij_IR_ij, Wij_II_ij = self._cross_terms(i, j, Wij_RR, Wij_RI, Wij_IR, Wij_II)
                cQ -= Gij * Wij_IR_ij - Bij * Wij_II_ij - Gij * Wij_RI_ij - Bij * Wij_RR_ij
            residuals.append(cQ)

        # 3) Branch power-flow definitions: c_{ij}^{P,flow}, c_{ij}^{Q,flow}
        for a, (i, j) in enumerate(self.arc_collection):
            lid = self.arc_to_line[a]
            ell = self.line_pos[lid]
            g_ij = self.g_series[ell]
            b_ij = self.b_series[ell]
            Wij_RR_ij, Wij_RI_ij, Wij_IR_ij, Wij_II_ij = self._cross_terms(i, j, Wij_RR, Wij_RI, Wij_IR, Wij_II)

            P_expr = g_ij * (Wii_RR[i] + Wii_II[i] - Wij_RR_ij - Wij_II_ij) + b_ij * (Wij_RI_ij - Wij_IR_ij)
            Q_expr = -b_ij * (Wii_RR[i] + Wii_II[i]) + b_ij * (Wij_RR_ij + Wij_II_ij) + g_ij * (Wij_RI_ij - Wij_IR_ij)

            cP_flow = P_ij[a] - P_expr
            cQ_flow = Q_ij[a] - Q_expr

            residuals.append(cP_flow)
            residuals.append(cQ_flow)

        # 4) Voltage magnitude definition c_i^{Vsq}
        for i in range(nb):
            residuals.append(V_sq[i] - (Wii_RR[i] + Wii_II[i]))

        # 5) Branch S_tot_sq definition c_{ij}^S
        for a in range(na):
            residuals.append(S_tot_sq[a] - (P_ij[a] ** 2 + Q_ij[a] ** 2))

        # 6) Reference bus constraint c^{ref} = Wii_II[ref_idx]
        c_ref = Wii_II[ref_idx]
        residuals.append(c_ref)

        return residuals

    def build_linear_ALM_Lagrangian_syms(self, x_center, rho, ref_bus_id=None, mu_prox=0.0):
        """
        Build one linearized ALM subproblem:
            L^(k)(x) = f(x)
                    + lambda^T * h_tilde(x)
                    + (rho/2) * ||h_tilde(x)||^2
                    + (mu_prox/2) * ||x - x^k||^2

        Linear blocks are kept in their original symbolic form. Nonlinear blocks
        are selectively linearized at x^k so the resulting subproblem remains a
        quadratic objective with box constraints.
        """
        x_syms = self.variable_list
        nvar = len(x_syms)

        x0 = np.asarray(x_center, dtype=float).flatten()
        if x0.size != nvar:
            raise ValueError(f"x_center length mismatch: expected {nvar}, got {x0.size}")

        rho = float(rho)
        lam = np.asarray(self.lambda_vec, dtype=float).flatten()

        # ---------- 1) Objective ----------
        obj = self.build_objective_expr()
        L = obj

        # ---------- 2) Symbolic h(x) ----------
        h_sym_list = self.build_h_symbolic(ref_bus_id=ref_bus_id)
        mcon = len(h_sym_list)
        if lam.size != mcon:
            raise ValueError(f"lambda size {lam.size} != number of constraints {mcon}")

        block_slices = self._get_constraint_block_slices()
        if not block_slices:
            raise ValueError("Constraint block metadata is empty.")
        total_dim = max(end for _, (start, end, _) in block_slices.items())
        if total_dim != mcon:
            raise ValueError(f"Constraint count mismatch: expected {total_dim}, got {mcon}")

        # ---------- 3) Linearize only the nonlinear blocks ----------
        subs_dict = {sym: sp.Float(val) for sym, val in zip(x_syms, x0)}
        h_tilde_exprs = list(h_sym_list)
        nonlinear_exprs = []
        nonlinear_global_indices = []
        for _, (start, end, linearize_flag) in block_slices.items():
            if not linearize_flag:
                continue
            nonlinear_exprs.extend(h_sym_list[start:end])
            nonlinear_global_indices.extend(range(start, end))

        if nonlinear_exprs:
            h_nl_vec = sp.Matrix(nonlinear_exprs)
            J_nl_sym = h_nl_vec.jacobian(x_syms)
            self.linear_jacobian = J_nl_sym

            h_nl_0_sym = h_nl_vec.subs(subs_dict)
            h_nl_0 = np.array([float(h_nl_0_sym[i]) for i in range(len(nonlinear_exprs))], dtype=float)

            J_nl_eval = J_nl_sym.subs(subs_dict)
            J_nl_num = np.zeros((len(nonlinear_exprs), nvar), dtype=float)
            for i in range(len(nonlinear_exprs)):
                for j in range(nvar):
                    J_nl_num[i, j] = float(J_nl_eval[i, j])

            for local_idx, global_idx in enumerate(nonlinear_global_indices):
                expr = sp.Float(h_nl_0[local_idx])
                for j in range(nvar):
                    coef = J_nl_num[local_idx, j]
                    if coef != 0.0:
                        expr += sp.Float(coef) * (x_syms[j] - sp.Float(x0[j]))
                h_tilde_exprs[global_idx] = sp.expand(expr)
        else:
            self.linear_jacobian = None

        # ---------- 4) Lagrange term ----------
        for i in range(mcon):
            if lam[i] != 0.0:
                L += sp.Float(lam[i]) * h_tilde_exprs[i]

        # ---------- 5) Quadratic penalty ----------
        if rho > 0.0:
            rho_half = sp.Float(rho) / 2.0
            for expr in h_tilde_exprs:
                L += rho_half * (expr ** 2)

        # ---------- 6) Proximal term ----------
        if mu_prox > 0.0:
            mu_half = sp.Float(mu_prox) / 2.0
            for j in range(nvar):
                L += mu_half * (x_syms[j] - sp.Float(x0[j])) ** 2

        self.Lagrange_linear_ALM = sp.expand(L)
        return self.Lagrange_linear_ALM, self.variable_list, self.Var_bound_list

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
    def build_h_func_legacy(self, ref_bus_id=None):
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
          c^{W,cons},
          c^{ref} ].

        x must be a flat array/list in the same order as self.variable_list:
        [P_G (ng),
         Q_G (ng),
         Wii_RR (nb),
         Wii_RI (nb),
         Wii_II (nb),
         Wij_RR (ncross),
         Wij_RI (ncross),
         Wij_IR (ncross),
         Wij_II (ncross),
         V_sq (nb),
         P_ij (na),
         Q_ij (na),
         S_tot_sq (na)].
        """
        nb = self.n_buses
        na = self.n_arcs
        ng = self.n_gens    

        def h_func(x):
            vals = self._unpack_x(x)

            P_G = vals["P_G"]
            Q_G = vals["Q_G"]
            Wii_RR = vals["Wii_RR"]
            Wii_RI = vals["Wii_RI"]
            Wii_II = vals["Wii_II"]
            Wij_RR = vals["Wij_RR"]
            Wij_RI = vals["Wij_RI"]
            Wij_IR = vals["Wij_IR"]
            Wij_II = vals["Wij_II"]
            V_sq = vals["V_sq"]
            P_ij = vals["P_ij"]
            Q_ij = vals["Q_ij"]
            S_tot_sq = vals["S_tot_sq"]

            residuals = []

            # 1) Active power balance c_i^P
            for i in range(nb):
                PG_sum = 0.0
                for bid, idx_list in self.gen_indices_by_bus.items():
                    if self.bus_index[bid] == i:
                        for gi in idx_list:
                            PG_sum += P_G[gi]
                cP = PG_sum - self.P_D[i]
                Gii = self.G_mat[i, i]
                cP -= Gii * Wii_RR[i] + Gii * Wii_II[i]
                for j in self.power_balance_pairs_by_bus[i]:
                    Gij = self.G_mat[i, j]
                    Bij = self.B_mat[i, j]
                    Wij_RR_ij, Wij_RI_ij, Wij_IR_ij, Wij_II_ij = self._cross_terms(i, j, Wij_RR, Wij_RI, Wij_IR, Wij_II)
                    cP -= Gij * Wij_RR_ij - Bij * Wij_RI_ij + Gij * Wij_II_ij + Bij * Wij_IR_ij
                residuals.append(cP)

            # 2) Reactive power balance c_i^Q
            for i in range(nb):
                QG_sum = 0.0
                for bid, idx_list in self.gen_indices_by_bus.items():
                    if self.bus_index[bid] == i:
                        for gi in idx_list:
                            QG_sum += Q_G[gi]
                cQ = QG_sum - self.Q_D[i]
                Gii = self.G_mat[i, i]
                Bii = self.B_mat[i, i]
                cQ += Bii * Wii_II[i] + Bii * Wii_RR[i]
                for j in self.power_balance_pairs_by_bus[i]:
                    Gij = self.G_mat[i, j]
                    Bij = self.B_mat[i, j]
                    Wij_RR_ij, Wij_RI_ij, Wij_IR_ij, Wij_II_ij = self._cross_terms(i, j, Wij_RR, Wij_RI, Wij_IR, Wij_II)
                    cQ -= Gij * Wij_IR_ij - Bij * Wij_II_ij - Gij * Wij_RI_ij - Bij * Wij_RR_ij
                residuals.append(cQ)

            # 3) Branch power-flow definition constraints
            for a, (i, j) in enumerate(self.arc_collection):
                lid = self.arc_to_line[a]
                ell = self.line_pos[lid]
                g_ij = self.g_series[ell]
                b_ij = self.b_series[ell]
                Wij_RR_ij, Wij_RI_ij, Wij_IR_ij, Wij_II_ij = self._cross_terms(i, j, Wij_RR, Wij_RI, Wij_IR, Wij_II)

                P_expr = g_ij * (Wii_RR[i] + Wii_II[i] - Wij_RR_ij - Wij_II_ij) + b_ij * (Wij_RI_ij - Wij_IR_ij)
                Q_expr = -b_ij * (Wii_RR[i] + Wii_II[i]) + b_ij * (Wij_RR_ij + Wij_II_ij) + g_ij * (Wij_RI_ij - Wij_IR_ij)
                cP_flow = P_ij[a] - P_expr
                cQ_flow = Q_ij[a] - Q_expr

                residuals.append(cP_flow)
                residuals.append(cQ_flow)

            # 4) Voltage magnitude definition c_i^{Vsq}
            for i in range(nb):
                cV = V_sq[i] - (Wii_RR[i] + Wii_II[i])
                residuals.append(cV)

            # 5) Branch S_tot_sq definition c_{ij}^S
            for a in range(na):
                cS = S_tot_sq[a] - (P_ij[a] ** 2 + Q_ij[a] ** 2)
                residuals.append(cS)

            # 6) Reference bus constraint c^{ref} = Wii_II[ref_idx]
            c_ref = Wii_II[ref_idx]
            residuals.append(c_ref)
            return np.asarray(residuals, dtype=float)

        return h_func
    
    def update_lambda(self, x, alpha, h_func):
        """
        Single-step dual update: λ^{k+1} = λ^k + α h(x^{k+1}),
        and write the updated λ back into the object
        (lambda_P_bal, lambda_Q_bal, ...).

        Parameters
        ----------
        x       : list or 1D array, the current primal solution x^{k+1}
        alpha   : float or 1D array, the dual step size α_k
        h_func  : callable, takes x as input and returns a 1D array h(x)

        Returns
        -------
        lambda_new  : 1D array, the updated λ^{k+1} (flattened form)
        h_x         : 1D array, the current constraint residual h(x^{k+1})
        """
        x_vec = np.asarray(x, dtype=float)

        # 当前 λ（扁平）和约束残差
        lam = np.asarray(self.lambda_vec, dtype=float)
        h_x = np.asarray(h_func(x_vec), dtype=float)

        # 维度检查
        assert lam.shape == h_x.shape, f"Dimension mismatch: lambda has shape {lam.shape}, while h(x) has shape {h_x.shape}"

        # 对偶更新
        lambda_new = lam + alpha * h_x

        # ---- 把 lambda_new 拆回各组乘子，并写回到 self ----
        idx = 0
        self.lambda_P_bal = lambda_new[idx:idx + self.n_buses].tolist(); idx += self.n_buses
        self.lambda_Q_bal = lambda_new[idx:idx + self.n_buses].tolist(); idx += self.n_buses
        self.lambda_P_flow = lambda_new[idx:idx + self.n_arcs].tolist(); idx += self.n_arcs
        self.lambda_Q_flow = lambda_new[idx:idx + self.n_arcs].tolist(); idx += self.n_arcs
        self.lambda_Vsq = lambda_new[idx:idx + self.n_buses].tolist(); idx += self.n_buses
        self.lambda_Ssq = lambda_new[idx:idx + self.n_arcs].tolist(); idx += self.n_arcs
        self.lambda_ref = float(lambda_new[idx]); idx += 1

        assert idx == len(lambda_new), "Failed to split lambda_new: length mismatch. Check whether the variable order is consistent with h_func."

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
            Wii_RR (nb),
            Wii_RI (nb),
            Wii_II (nb),
            Wij_RR (ncross),
            Wij_RI (ncross),
            Wij_IR (ncross),
            Wij_II (ncross),
            V_sq (nb),
            P_ij (na),
            Q_ij (na),
            S_tot_sq (na)].
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

    def build_h_symbolic(self, ref_bus_id=None):
        """
        Build the symbolic equality-constraint vector h(x) in the fixed order:
        1) active power balance
        2) reactive power balance
        3) branch active flow definitions
        4) branch reactive flow definitions
        5) voltage magnitude auxiliary constraints
        6) branch thermal auxiliary constraints
        7) reference bus lifted constraint
        """
        nb = self.n_buses
        na = self.n_arcs

        P_G = self.P_G
        Q_G = self.Q_G
        Wii_RR = self.Wii_RR
        Wii_RI = self.Wii_RI
        Wii_II = self.Wii_II
        Wij_RR = self.Wij_RR
        Wij_RI = self.Wij_RI
        Wij_IR = self.Wij_IR
        Wij_II = self.Wij_II
        V_sq = self.V_sq
        P_ij = self.P_ij
        Q_ij = self.Q_ij
        S_tot_sq = self.S_tot_sq

        if ref_bus_id is None:
            ref_bus_id = self.bus_ids[0]
        ref_idx = self.bus_index[ref_bus_id]

        residuals = []

        for i in range(nb):
            PG_sum = 0
            for gi in self.gen_indices_by_bus.get(self.bus_ids[i], []):
                PG_sum += P_G[gi]
            cP = PG_sum - self.P_D[i]
            Gii = self.G_mat[i, i]
            cP -= Gii * Wii_RR[i] + Gii * Wii_II[i]
            for j in self.power_balance_pairs_by_bus[i]:
                Gij = self.G_mat[i, j]
                Bij = self.B_mat[i, j]
                Wij_RR_ij, Wij_RI_ij, Wij_IR_ij, Wij_II_ij = self._cross_terms(i, j, Wij_RR, Wij_RI, Wij_IR, Wij_II)
                cP -= Gij * Wij_RR_ij - Bij * Wij_RI_ij + Gij * Wij_II_ij + Bij * Wij_IR_ij
            residuals.append(cP)

        for i in range(nb):
            QG_sum = 0
            for gi in self.gen_indices_by_bus.get(self.bus_ids[i], []):
                QG_sum += Q_G[gi]
            cQ = QG_sum - self.Q_D[i]
            Gii = self.G_mat[i, i]
            Bii = self.B_mat[i, i]
            cQ += Bii * Wii_II[i] + Bii * Wii_RR[i]
            for j in self.power_balance_pairs_by_bus[i]:
                Gij = self.G_mat[i, j]
                Bij = self.B_mat[i, j]
                Wij_RR_ij, Wij_RI_ij, Wij_IR_ij, Wij_II_ij = self._cross_terms(i, j, Wij_RR, Wij_RI, Wij_IR, Wij_II)
                cQ -= Gij * Wij_IR_ij - Bij * Wij_II_ij - Gij * Wij_RI_ij - Bij * Wij_RR_ij
            residuals.append(cQ)

        for a, (i, j) in enumerate(self.arc_collection):
            lid = self.arc_to_line[a]
            ell = self.line_pos[lid]
            g_ij = self.g_series[ell]
            b_ij = self.b_series[ell]
            Wij_RR_ij, Wij_RI_ij, Wij_IR_ij, Wij_II_ij = self._cross_terms(i, j, Wij_RR, Wij_RI, Wij_IR, Wij_II)
            P_expr = g_ij * (Wii_RR[i] + Wii_II[i] - Wij_RR_ij - Wij_II_ij) + b_ij * (Wij_RI_ij - Wij_IR_ij)
            residuals.append(P_ij[a] - P_expr)

        for a, (i, j) in enumerate(self.arc_collection):
            lid = self.arc_to_line[a]
            ell = self.line_pos[lid]
            g_ij = self.g_series[ell]
            b_ij = self.b_series[ell]
            Wij_RR_ij, Wij_RI_ij, Wij_IR_ij, Wij_II_ij = self._cross_terms(i, j, Wij_RR, Wij_RI, Wij_IR, Wij_II)
            Q_expr = -b_ij * (Wii_RR[i] + Wii_II[i]) + b_ij * (Wij_RR_ij + Wij_II_ij) + g_ij * (Wij_RI_ij - Wij_IR_ij)
            residuals.append(Q_ij[a] - Q_expr)

        for i in range(nb):
            residuals.append(V_sq[i] - (Wii_RR[i] + Wii_II[i]))

        for a in range(na):
            residuals.append(S_tot_sq[a] - (P_ij[a] ** 2 + Q_ij[a] ** 2))

        residuals.append(Wii_II[ref_idx])
        return residuals

    def build_h_func(self, ref_bus_id=None):
        """
        Return a callable h_func(x) that evaluates all equality residuals h(x).
        The stacking order matches build_h_symbolic() and the lambda vector.
        """
        nb = self.n_buses
        na = self.n_arcs
        if ref_bus_id is None:
            ref_bus_id = self.bus_ids[0]
        ref_idx = self.bus_index[ref_bus_id]

        def h_func(x):
            vals = self._unpack_x(x)

            P_G = vals["P_G"]
            Q_G = vals["Q_G"]
            Wii_RR = vals["Wii_RR"]
            Wii_RI = vals["Wii_RI"]
            Wii_II = vals["Wii_II"]
            Wij_RR = vals["Wij_RR"]
            Wij_RI = vals["Wij_RI"]
            Wij_IR = vals["Wij_IR"]
            Wij_II = vals["Wij_II"]
            V_sq = vals["V_sq"]
            P_ij = vals["P_ij"]
            Q_ij = vals["Q_ij"]
            S_tot_sq = vals["S_tot_sq"]

            residuals = []

            for i in range(nb):
                PG_sum = 0.0
                for gi in self.gen_indices_by_bus.get(self.bus_ids[i], []):
                    PG_sum += P_G[gi]
                cP = PG_sum - self.P_D[i]
                Gii = self.G_mat[i, i]
                cP -= Gii * Wii_RR[i] + Gii * Wii_II[i]
                for j in self.power_balance_pairs_by_bus[i]:
                    Gij = self.G_mat[i, j]
                    Bij = self.B_mat[i, j]
                    Wij_RR_ij, Wij_RI_ij, Wij_IR_ij, Wij_II_ij = self._cross_terms(i, j, Wij_RR, Wij_RI, Wij_IR, Wij_II)
                    cP -= Gij * Wij_RR_ij - Bij * Wij_RI_ij + Gij * Wij_II_ij + Bij * Wij_IR_ij
                residuals.append(cP)

            for i in range(nb):
                QG_sum = 0.0
                for gi in self.gen_indices_by_bus.get(self.bus_ids[i], []):
                    QG_sum += Q_G[gi]
                cQ = QG_sum - self.Q_D[i]
                Gii = self.G_mat[i, i]
                Bii = self.B_mat[i, i]
                cQ += Bii * Wii_II[i] + Bii * Wii_RR[i]
                for j in self.power_balance_pairs_by_bus[i]:
                    Gij = self.G_mat[i, j]
                    Bij = self.B_mat[i, j]
                    Wij_RR_ij, Wij_RI_ij, Wij_IR_ij, Wij_II_ij = self._cross_terms(i, j, Wij_RR, Wij_RI, Wij_IR, Wij_II)
                    cQ -= Gij * Wij_IR_ij - Bij * Wij_II_ij - Gij * Wij_RI_ij - Bij * Wij_RR_ij
                residuals.append(cQ)

            for a, (i, j) in enumerate(self.arc_collection):
                lid = self.arc_to_line[a]
                ell = self.line_pos[lid]
                g_ij = self.g_series[ell]
                b_ij = self.b_series[ell]
                Wij_RR_ij, Wij_RI_ij, Wij_IR_ij, Wij_II_ij = self._cross_terms(i, j, Wij_RR, Wij_RI, Wij_IR, Wij_II)
                P_expr = g_ij * (Wii_RR[i] + Wii_II[i] - Wij_RR_ij - Wij_II_ij) + b_ij * (Wij_RI_ij - Wij_IR_ij)
                residuals.append(P_ij[a] - P_expr)

            for a, (i, j) in enumerate(self.arc_collection):
                lid = self.arc_to_line[a]
                ell = self.line_pos[lid]
                g_ij = self.g_series[ell]
                b_ij = self.b_series[ell]
                Wij_RR_ij, Wij_RI_ij, Wij_IR_ij, Wij_II_ij = self._cross_terms(i, j, Wij_RR, Wij_RI, Wij_IR, Wij_II)
                Q_expr = -b_ij * (Wii_RR[i] + Wii_II[i]) + b_ij * (Wij_RR_ij + Wij_II_ij) + g_ij * (Wij_RI_ij - Wij_IR_ij)
                residuals.append(Q_ij[a] - Q_expr)

            for i in range(nb):
                residuals.append(V_sq[i] - (Wii_RR[i] + Wii_II[i]))

            for a in range(na):
                residuals.append(S_tot_sq[a] - (P_ij[a] ** 2 + Q_ij[a] ** 2))

            residuals.append(Wii_II[ref_idx])
            return np.asarray(residuals, dtype=float)

        return h_func

def create_qhd_acopf_log_file(model, folder="."):
    """
Create a log file.

Naming convention:
    Buses_<n>_<HH>-<MM>-<SS>_<MM>-<DD>-<YYYY>.txt

where:
    n = number of buses
    HH-MM-SS = file creation time
    MM-DD-YYYY = current date
""" 
    now = datetime.now()
    n = model.n_buses
    time_str = now.strftime("%H-%M-%S")
    date_str = now.strftime("%m-%d-%Y")

    filename = f"Buses_{n}_{time_str}_{date_str}.txt"
    filepath = os.path.join(folder, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("QHD ACOPF Iteration Log\n")
        f.write(f"Created at: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of buses: {n}\n")
        f.write("=" * 100 + "\n\n")

    return filepath


def _build_qhd_acopf_log_filename(model, folder="."):
    """
    日志文件命名规则：
        Buses-n-time-MM-DD-YYYY.txt

    其中：
        n    = 总线数
        time = 文件创建时刻，格式 HH-MM-SS
        date = 当天日期，格式 MM-DD-YYYY
    """
    now = datetime.now()
    n = model.n_buses
    time_str = now.strftime("%H-%M-%S")
    date_str = now.strftime("%m-%d-%Y")
    filename = f"Buses-{n}_{date_str}_{time_str}.txt"
    return str(Path(folder) / filename)


def initialize_qhd_acopf_log(model, folder=".", extra_header=None, option=None, qhd_solver=None):
    """
    首次创建日志文件，并把路径挂在 model._qhd_acopf_log_file 上。
    """
    log_file = _build_qhd_acopf_log_filename(model, folder=folder)
    Path(folder).mkdir(parents=True, exist_ok=True)

    options_str = ["QHD", "Gurobi"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if option == 1:
        solver_string = f"\n--- QHD with {qhd_solver} solver ---"         
    elif option == 2:
        solver_string = f"\n--- QHD with Gurobi solver ---"
    else:
        solver_string = f"\n--- invalid option ---"

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("QHD ACOPF / ALM Iteration Log\n")
        f.write(f"Created at: {now}\n")
        f.write(f"Number of buses: {model.n_buses}\n")
        f.write(f"Number of lines: {model.n_lines}\n")
        f.write(f"Number of generators: {model.n_gens}\n")
        f.write(f"Solver used: {solver_string}\n")

        if extra_header is not None:
            f.write(str(extra_header).rstrip() + "\n")
        f.write("=" * 120 + "\n\n")

    model._qhd_acopf_log_file = log_file
    return log_file


def format_qhd_acopf_results(model, solution, iteration=None, rho=None, alpha=None,
                             h_x=None, lambda_vec=None, objective_value=None,
                             feasibility=None, note=None):
    """
    把当前一轮 ALM / QHD 结果整理成字符串，供写入 txt 日志。
    """
    x = np.asarray(solution, dtype=float).flatten()

    vals = model._unpack_x(x)
    voltage_post = model.recover_voltages_from_lifted(vals=vals)
    P_G = vals["P_G"]
    Q_G = vals["Q_G"]
    Wii_RR = vals["Wii_RR"]
    Wii_RI = vals["Wii_RI"]
    Wii_II = vals["Wii_II"]
    V_sq = vals["V_sq"]
    P_ij = vals["P_ij"]
    Q_ij = vals["Q_ij"]
    S_tot_sq = vals["S_tot_sq"]
    V_R = voltage_post["V_R"]
    V_I = voltage_post["V_I"]
    V_mag = voltage_post["V_mag"]

    bus_ids = model.bus_ids
    line_ids = model.line_ids
    lines = model.lines
    gen_ids = model.gen_ids
    gens = model.gens
    P_D = model.P_D
    Q_D = model.Q_D

    gen_index_by_id = {gid: k for k, gid in enumerate(gen_ids)}

    out = []
    out.append("=" * 120)
    out.append(f"Update time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if iteration is not None:
        out.append(f"Iteration: {iteration}")
    if rho is not None:
        out.append(f"rho: {float(rho):.8g}")
    if alpha is not None:
        out.append(f"alpha: {float(alpha):.8g}")
    if objective_value is not None:
        out.append(f"objective_value: {float(objective_value):.12g}")
    if feasibility is not None:
        out.append(f"feasible: {bool(feasibility)}")
    if note is not None:
        out.append(f"note: {note}")

    if h_x is not None:
        h_x = np.asarray(h_x, dtype=float).flatten()
        out.append(f"max_abs_h: {np.max(np.abs(h_x)):.12e}")
        out.append(f"l2_norm_h: {np.linalg.norm(h_x):.12e}")

    if lambda_vec is not None:
        lam = np.asarray(lambda_vec, dtype=float).flatten()
        out.append(f"lambda_inf_norm: {np.max(np.abs(lam)):.12e}")
        out.append(f"lambda_l2_norm: {np.linalg.norm(lam):.12e}")

    out.append("-" * 120)
    out.append("Bus Results")
    out.append("BusID\tV_R\t\tV_I\t\tVmag\t\tWii_RR\t\tWii_II\t\tPg\t\tQg\t\tPl\t\tQl")

    Pgtotal = 0.0
    Qgtotal = 0.0
    Ploadtotal = 0.0
    Qloadtotal = 0.0

    for bi, bus_id in enumerate(bus_ids):
        Vmag = np.sqrt(max(V_sq[bi], 0.0))

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

        Pgtotal += Pg_i
        Qgtotal += Qg_i
        Ploadtotal += Pl_i
        Qloadtotal += Ql_i

        out.append(
            f"{bus_id}\t{V_R[bi]:.6f}\t{V_I[bi]:.6f}\t{V_mag[bi]:.6f}\t{Wii_RR[bi]:.6f}\t{Wii_II[bi]:.6f}\t{Pg_i:.6f}\t{Qg_i:.6f}\t{Pl_i:.6f}\t{Ql_i:.6f}"
        )

    out.append("")
    out.append("Branch Results")
    out.append("LineID	From	To	Pik		Pki		Qik		Qki		LossP		LossQ")

    total_Ploss = 0.0
    total_Qloss = 0.0

    for lid in line_ids:
        from_bus, to_bus, _, _, _, _, _ = lines[lid]
        a_fwd = model.arc_index[(lid, +1)]
        a_rev = model.arc_index[(lid, -1)]

        Pik = P_ij[a_fwd]
        Pki = P_ij[a_rev]
        Qik = Q_ij[a_fwd]
        Qki = Q_ij[a_rev]

        lossP = Pik + Pki
        lossQ = Qik + Qki
        total_Ploss += lossP
        total_Qloss += lossQ

        out.append(
            f"{lid}	{from_bus}	{to_bus}	{Pik:.6f}	{Pki:.6f}	{Qik:.6f}	{Qki:.6f}	{lossP:.6f}	{lossQ:.6f}"
        )

    out.append("")
    out.append("Summary")
    out.append(f"Total Pg           : {Pgtotal:.6f}")
    out.append(f"Total Qg           : {Qgtotal:.6f}")
    out.append(f"Total Load P       : {Ploadtotal:.6f}")
    out.append(f"Total Load Q       : {Qloadtotal:.6f}")
    out.append(f"Total Active Loss  : {total_Ploss:.6f}")
    out.append(f"Total Reactive Loss: {total_Qloss:.6f}")

    if abs(Ploadtotal) > 1e-12:
        Pl_supplied = Pgtotal - total_Ploss
        perc_supplied = 100.0 * Pl_supplied / Ploadtotal
        out.append(f"Total Load Supplied: {perc_supplied:.6f}%")
    else:
        out.append("Total Load Supplied: N/A")

    if np.any(S_tot_sq < -1e-10):
        out.append(f"Warning: found negative S_tot_sq min = {np.min(S_tot_sq):.6e}")

    out.append("")
    return "\n".join(out)


def append_qhd_acopf_results(model, solution, log_file=None, folder=".", **kwargs):
    """
    追加写入一轮结果到 txt。首次调用若没有 log_file，会自动创建。
    """
    if log_file is None:
        log_file = getattr(model, "_qhd_acopf_log_file", None)

    if log_file is None:
        log_file = initialize_qhd_acopf_log(model, folder=folder)

    text = format_qhd_acopf_results(model, solution, **kwargs)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")

    model._qhd_acopf_log_file = log_file
    return log_file


def format_qhd_acopf_console_results(model, solution, iteration=None, rho=None, alpha=None,
                                     h_x=None, lambda_vec=None, objective_value=None,
                                     feasibility=None, note=None):
    """
    Compact console formatter: keep V_R / V_I in scrolling output, omit lifted W details.
    """
    x = np.asarray(solution, dtype=float).flatten()
    vals = model._unpack_x(x)
    voltage_post = model.recover_voltages_from_lifted(vals=vals)

    P_G = vals["P_G"]
    Q_G = vals["Q_G"]
    P_ij = vals["P_ij"]
    Q_ij = vals["Q_ij"]
    bus_ids = model.bus_ids
    line_ids = model.line_ids
    lines = model.lines
    gen_ids = model.gen_ids
    gens = model.gens
    P_D = model.P_D
    Q_D = model.Q_D
    V_R = voltage_post["V_R"]
    V_I = voltage_post["V_I"]
    V_mag = voltage_post["V_mag"]

    gen_index_by_id = {gid: k for k, gid in enumerate(gen_ids)}

    out = []
    out.append("=" * 120)
    out.append(f"Update time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if iteration is not None:
        out.append(f"Iteration: {iteration}")
    if rho is not None:
        out.append(f"rho: {float(rho):.8g}")
    if alpha is not None:
        out.append(f"alpha: {float(alpha):.8g}")
    if objective_value is not None:
        out.append(f"objective_value: {float(objective_value):.12g}")
    if feasibility is not None:
        out.append(f"feasible: {bool(feasibility)}")
    if note is not None:
        out.append(f"note: {note}")

    if h_x is not None:
        h_x = np.asarray(h_x, dtype=float).flatten()
        out.append(f"max_abs_h: {np.max(np.abs(h_x)):.12e}")
        out.append(f"l2_norm_h: {np.linalg.norm(h_x):.12e}")

    if lambda_vec is not None:
        lam = np.asarray(lambda_vec, dtype=float).flatten()
        out.append(f"lambda_inf_norm: {np.max(np.abs(lam)):.12e}")
        out.append(f"lambda_l2_norm: {np.linalg.norm(lam):.12e}")

    out.append("-" * 120)
    out.append("Bus Results")
    out.append("BusID\tV_R\t\tV_I\t\tVmag\t\tPg\t\tQg\t\tPl\t\tQl")

    for bi, bus_id in enumerate(bus_ids):
        Pg_i = 0.0
        Qg_i = 0.0
        for gid in gen_ids:
            if gens[gid][0] == bus_id:
                gi = gen_index_by_id[gid]
                Pg_i += P_G[gi]
                Qg_i += Q_G[gi]

        out.append(
            f"{bus_id}\t{V_R[bi]:.6f}\t{V_I[bi]:.6f}\t{V_mag[bi]:.6f}\t{Pg_i:.6f}\t{Qg_i:.6f}\t{P_D[bi]:.6f}\t{Q_D[bi]:.6f}"
        )

    out.append("")
    out.append("Branch Results")
    out.append("LineID\tFrom\tTo\tPik\t\tPki\t\tQik\t\tQki\t\tLossP\t\tLossQ")

    for lid in line_ids:
        from_bus, to_bus, _, _, _, _, _ = lines[lid]
        a_fwd = model.arc_index[(lid, +1)]
        a_rev = model.arc_index[(lid, -1)]
        Pik = P_ij[a_fwd]
        Pki = P_ij[a_rev]
        Qik = Q_ij[a_fwd]
        Qki = Q_ij[a_rev]
        out.append(
            f"{lid}\t{from_bus}\t{to_bus}\t{Pik:.6f}\t{Pki:.6f}\t{Qik:.6f}\t{Qki:.6f}\t{(Pik + Pki):.6f}\t{(Qik + Qki):.6f}"
        )

    out.append("")
    return "\n".join(out)


def PrintQHDACOPFResults(model, solution, iteration=None, log_file=None, folder=".",
                         print_to_console=True, **kwargs):
    """
    兼容旧接口：
    - 默认仍可把结果打印到屏幕
    - 同时把结果写入日志 txt
    """
    text = format_qhd_acopf_console_results(model, solution, iteration=iteration, **kwargs)

    if print_to_console:
        print(text)

    log_file = append_qhd_acopf_results(
        model=model,
        solution=solution,
        log_file=log_file,
        folder=folder,
        iteration=iteration,
        **kwargs,
    )
    return log_file

def run_linear_alm_with_logging(model, x0=None, max_iter=20, rho=10.0, alpha=1.0,
                                mu_prox=1e-4, ref_bus_id=None, tol_eq=1e-6,
                                tol_ineq=1e-9, verbose=True, log_folder="."):
    """
    一个可直接运行的 ALM 主循环，并把每一轮结果自动追加到日志 txt。

    返回
    ----
    result : dict
        {
            "x": 最后一轮 primal 解,
            "lambda": 最后一轮对偶变量,
            "h": 最后一轮约束残差,
            "log_file": 日志路径,
            "history": 每轮摘要列表,
            "all_ok": 是否满足约束,
        }
    """
    if x0 is None:
        xk = model.build_initial_x0()
    else:
        xk = np.asarray(x0, dtype=float).flatten()

    h_func = model.build_h_func(ref_bus_id=ref_bus_id)
    log_file = initialize_qhd_acopf_log(
        model,
        folder=log_folder,
        extra_header=(
            f"ALM settings: max_iter={max_iter}, rho={rho}, alpha={alpha}, "
            f"mu_prox={mu_prox}, tol_eq={tol_eq}, tol_ineq={tol_ineq}"
        ),
    )

    history = []
    best_x = xk.copy()
    best_metric = np.inf
    best_ok = False

    for k in range(max_iter):
        L_sym, variable_list, Var_bound_list = model.build_linear_ALM_Lagrangian_syms(
            x_center=xk,
            rho=rho,
            ref_bus_id=ref_bus_id,
            mu_prox=mu_prox,
        )

        x_next = solve_with_gurobi_from_sympy(
            L_sym=L_sym,
            variable_list=variable_list,
            Var_bound_list=Var_bound_list,
            verbose=False,
        )

        lambda_new, h_x = model.update_lambda(x_next, alpha=alpha, h_func=h_func)
        ok_list, all_ok = model.check_constraints(
            x_next,
            ref_bus_id=ref_bus_id,
            tol_eq=tol_eq,
            tol_ineq=tol_ineq,
        )

        metric = float(np.max(np.abs(h_x)))
        if metric < best_metric:
            best_metric = metric
            best_x = x_next.copy()
            best_ok = all_ok

        try:
            obj_val = float(sp.N(model.objective.subs({sym: val for sym, val in zip(model.variable_list, x_next)})))
        except Exception:
            obj_val = None

        history_item = {
            "iter": k,
            "max_abs_h": float(np.max(np.abs(h_x))),
            "l2_norm_h": float(np.linalg.norm(h_x)),
            "lambda_inf": float(np.max(np.abs(lambda_new))),
            "feasible": bool(all_ok),
            "objective": obj_val,
        }
        history.append(history_item)

        append_qhd_acopf_results(
            model=model,
            solution=x_next,
            log_file=log_file,
            iteration=k,
            rho=rho,
            alpha=alpha,
            h_x=h_x,
            lambda_vec=lambda_new,
            objective_value=obj_val,
            feasibility=all_ok,
        )

        if verbose:
            print(
                f"[ALM] iter={k:03d} | max|h|={history_item['max_abs_h']:.3e} "
                f"| ||h||2={history_item['l2_norm_h']:.3e} "
                f"| ||lambda||inf={history_item['lambda_inf']:.3e} "
                f"| feasible={all_ok}"
            )
            print(f"[ALM] log updated: {log_file}")

        xk = x_next

        if all_ok:
            best_x = x_next.copy()
            best_ok = True
            break

    return {
        "x": best_x,
        "lambda": np.asarray(model.lambda_vec, dtype=float).copy(),
        "h": np.asarray(h_func(best_x), dtype=float).copy(),
        "log_file": log_file,
        "history": history,
        "all_ok": best_ok,
    }

def PrintQHDACOPFResults_old(model, solution):
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
    Wii_RR = x[idx:idx + nb]; idx += nb
    idx += nb  # skip Wii_RI
    Wii_II = x[idx:idx + nb]; idx += nb
    idx += model.n_cross_pairs  # skip Wij_RR
    idx += model.n_cross_pairs  # skip Wij_RI
    idx += model.n_cross_pairs  # skip Wij_IR
    idx += model.n_cross_pairs  # skip Wij_II
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
    print("BusID\tWii_RR\tWii_II\tVmag\tPg\tQg\tl\tPl\tQl\tQshunt\n")

    Pgtotal = 0.0
    Qgtotal = 0.0
    Ploadtotal = 0.0
    Qloadtotal = 0.0

    for bi, bus_id in enumerate(bus_ids):
        Vmag = np.sqrt(max(V_sq[bi], 0.0))

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
            "{0:d}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}\t{7:.4f}\t{8:.4f}\t{9:.4f}".format(
                bus_id, Wii_RR[bi], Wii_II[bi], Vmag, Pg_i, Qg_i, l_i, Pl_i, Ql_i, Qsh_i
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


def format_qhd_acopf_results_old(model, solution, iteration=None, rho=None, alpha=None,
                             h_x=None, lambda_vec=None, objective_value=None,
                             feasibility=None, note=None):
    """
    Format one ALM/QHD iterate using the lifted-only variable structure.
    """
    x = np.asarray(solution, dtype=float).flatten()
    vals = model._unpack_x(x)

    P_G = vals["P_G"]
    Q_G = vals["Q_G"]
    Wii_RR = vals["Wii_RR"]
    Wii_II = vals["Wii_II"]
    V_sq = vals["V_sq"]
    P_ij = vals["P_ij"]
    Q_ij = vals["Q_ij"]
    S_tot_sq = vals["S_tot_sq"]

    bus_ids = model.bus_ids
    line_ids = model.line_ids
    lines = model.lines
    gen_ids = model.gen_ids
    gens = model.gens
    P_D = model.P_D
    Q_D = model.Q_D

    gen_index_by_id = {gid: k for k, gid in enumerate(gen_ids)}

    out = []
    out.append("=" * 120)
    out.append(f"Update time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if iteration is not None:
        out.append(f"Iteration: {iteration}")
    if rho is not None:
        out.append(f"rho: {float(rho):.8g}")
    if alpha is not None:
        out.append(f"alpha: {float(alpha):.8g}")
    if objective_value is not None:
        out.append(f"objective_value: {float(objective_value):.12g}")
    if feasibility is not None:
        out.append(f"feasible: {bool(feasibility)}")
    if note is not None:
        out.append(f"note: {note}")

    if h_x is not None:
        h_x = np.asarray(h_x, dtype=float).flatten()
        out.append(f"max_abs_h: {np.max(np.abs(h_x)):.12e}")
        out.append(f"l2_norm_h: {np.linalg.norm(h_x):.12e}")

    if lambda_vec is not None:
        lam = np.asarray(lambda_vec, dtype=float).flatten()
        out.append(f"lambda_inf_norm: {np.max(np.abs(lam)):.12e}")
        out.append(f"lambda_l2_norm: {np.linalg.norm(lam):.12e}")

    out.append("-" * 120)
    out.append("Bus Results")
    out.append("BusID\tV_R\t\tV_I\t\tVmag\t\tWii_RR\t\tWii_RI\t\tWii_II\t\tPg\t\tQg\t\tPl\t\tQl")

    Pgtotal = 0.0
    Qgtotal = 0.0
    Ploadtotal = 0.0
    Qloadtotal = 0.0

    for bi, bus_id in enumerate(bus_ids):
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

        Pgtotal += Pg_i
        Qgtotal += Qg_i
        Ploadtotal += Pl_i
        Qloadtotal += Ql_i

        out.append(
            f"{bus_id}\t{V_R[bi]:.6f}\t{V_I[bi]:.6f}\t{V_mag[bi]:.6f}\t{Wii_RR[bi]:.6f}\t{Wii_RI[bi]:.6f}\t{Wii_II[bi]:.6f}\t{Pg_i:.6f}\t{Qg_i:.6f}\t{Pl_i:.6f}\t{Ql_i:.6f}"
        )

    out.append("")
    out.append("Branch Results")
    out.append("LineID\tFrom\tTo\tPik\t\tPki\t\tQik\t\tQki\t\tLossP\t\tLossQ")

    total_Ploss = 0.0
    total_Qloss = 0.0

    for lid in line_ids:
        from_bus, to_bus, _, _, _, _, _ = lines[lid]
        a_fwd = model.arc_index[(lid, +1)]
        a_rev = model.arc_index[(lid, -1)]

        Pik = P_ij[a_fwd]
        Pki = P_ij[a_rev]
        Qik = Q_ij[a_fwd]
        Qki = Q_ij[a_rev]

        lossP = Pik + Pki
        lossQ = Qik + Qki
        total_Ploss += lossP
        total_Qloss += lossQ

        out.append(
            f"{lid}\t{from_bus}\t{to_bus}\t{Pik:.6f}\t{Pki:.6f}\t{Qik:.6f}\t{Qki:.6f}\t{lossP:.6f}\t{lossQ:.6f}"
        )

    out.append("")
    out.append("Summary")
    out.append(f"Total Pg           : {Pgtotal:.6f}")
    out.append(f"Total Qg           : {Qgtotal:.6f}")
    out.append(f"Total Load P       : {Ploadtotal:.6f}")
    out.append(f"Total Load Q       : {Qloadtotal:.6f}")
    out.append(f"Total Active Loss  : {total_Ploss:.6f}")
    out.append(f"Total Reactive Loss: {total_Qloss:.6f}")

    if abs(Ploadtotal) > 1e-12:
        Pl_supplied = Pgtotal - total_Ploss
        perc_supplied = 100.0 * Pl_supplied / Ploadtotal
        out.append(f"Total Load Supplied: {perc_supplied:.6f}%")
    else:
        out.append("Total Load Supplied: N/A")

    if np.any(S_tot_sq < -1e-10):
        out.append(f"Warning: found negative S_tot_sq min = {np.min(S_tot_sq):.6e}")

    out.append("")
    return "\n".join(out)


def PrintQHDACOPFResults_older(model, solution):
    """
    Backward-compatible console printer using the lifted-only variable layout.
    """
    x = np.asarray(solution, dtype=float).flatten()
    vals = model._unpack_x(x)

    P_G = vals["P_G"]
    Q_G = vals["Q_G"]
    Wii_RR = vals["Wii_RR"]
    Wii_II = vals["Wii_II"]
    V_sq = vals["V_sq"]
    P_ij = vals["P_ij"]
    Q_ij = vals["Q_ij"]

    bus_ids = model.bus_ids
    line_ids = model.line_ids
    lines = model.lines
    gen_ids = model.gen_ids
    gens = model.gens
    P_D = model.P_D
    Q_D = model.Q_D

    gen_index_by_id = {gid: k for k, gid in enumerate(gen_ids)}

    print("BusID\tWii_RR\tWii_II\tVmag\tPg\tQg\tl\tPl\tQl\tQshunt\n")

    Pgtotal = 0.0
    Qgtotal = 0.0
    Ploadtotal = 0.0
    Qloadtotal = 0.0

    for bi, bus_id in enumerate(bus_ids):

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
            "{0:d}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}\t{7:.4f}\t{8:.4f}\t{9:.4f}".format(
                bus_id, Wii_RR[bi], Wii_II[bi], Vmag, Pg_i, Qg_i, l_i, Pl_i, Ql_i, Qsh_i
            )
        )

        Pgtotal += Pg_i
        Qgtotal += Qg_i
        Ploadtotal += Pl_i
        Qloadtotal += Ql_i

    print("\n")
    print("TOTAL\t\t\t\t{0:.4f}\t{1:.4f}\t\t{2:.4f}\t{3:.4f}".format(
        Pgtotal, Qgtotal, Ploadtotal, Qloadtotal
    ))
    print("\n\n")

    print("Busi\tBusk\tPik\tPki\tQik\tQki")

    Ploss = 0.0
    Qloss = 0.0

    for lid in line_ids:
        from_bus, to_bus, _, _, _, _, _ = lines[lid]
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

        Ploss += Pik + Pki
        Qloss += Qik + Qki

    print("\n")
    print("Total Ploss: {0:.4f}".format(Ploss))
    print("Total Qloss: {0:.4f}".format(Qloss))

    if Ploadtotal > 1e-8:
        Pl_supplied = Pgtotal - Ploss
        perc_supplied = (Pl_supplied / Ploadtotal) * 100.0
    else:
        perc_supplied = 0.0

    print("Total Load Supplied: {0:.4f}%".format(perc_supplied))


def PrintQHDACOPFResults_old(model, solution):
    """
    Backward-compatible console printer with postprocessed V_R / V_I recovered
    from lifted diagonal W variables.
    """
    vals = model.postprocess_solution(solution)

    P_G = vals["P_G"]
    Q_G = vals["Q_G"]
    Wii_RR = vals["Wii_RR"]
    Wii_RI = vals["Wii_RI"]
    Wii_II = vals["Wii_II"]
    V_R = vals["V_R"]
    V_I = vals["V_I"]
    V_mag = vals["V_mag"]
    P_ij = vals["P_ij"]
    Q_ij = vals["Q_ij"]

    bus_ids = model.bus_ids
    line_ids = model.line_ids
    lines = model.lines
    gen_ids = model.gen_ids
    gens = model.gens
    P_D = model.P_D
    Q_D = model.Q_D

    gen_index_by_id = {gid: k for k, gid in enumerate(gen_ids)}

    print("BusID\tV_R\tV_I\tVmag\tWii_RR\tWii_RI\tWii_II\tPg\tQg\tl\tPl\tQl\tQshunt\n")

    Pgtotal = 0.0
    Qgtotal = 0.0
    Ploadtotal = 0.0
    Qloadtotal = 0.0

    for bi, bus_id in enumerate(bus_ids):
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
            "{0:d}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}\t{7:.4f}\t{8:.4f}\t{9:.4f}\t{10:.4f}\t{11:.4f}\t{12:.4f}".format(
                bus_id, V_R[bi], V_I[bi], V_mag[bi], Wii_RR[bi], Wii_RI[bi], Wii_II[bi], Pg_i, Qg_i, l_i, Pl_i, Ql_i, Qsh_i
            )
        )

        Pgtotal += Pg_i
        Qgtotal += Qg_i
        Ploadtotal += Pl_i
        Qloadtotal += Ql_i

    print("\n")
    print("TOTAL\t\t\t\t\t\t\t{0:.4f}\t{1:.4f}\t\t{2:.4f}\t{3:.4f}".format(
        Pgtotal, Qgtotal, Ploadtotal, Qloadtotal
    ))
    print("\n\n")

    print("Busi\tBusk\tPik\tPki\tQik\tQki")

    Ploss = 0.0
    Qloss = 0.0

    for lid in line_ids:
        from_bus, to_bus, _, _, _, _, _ = lines[lid]
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

        Ploss += Pik + Pki
        Qloss += Qik + Qki

    print("\n")
    print("Total Ploss: {0:.4f}".format(Ploss))
    print("Total Qloss: {0:.4f}".format(Qloss))

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
