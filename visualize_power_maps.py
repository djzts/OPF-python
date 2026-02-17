import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def _scale_values(vals, vmin=None, vmax=None, out_min=0.5, out_max=8.0):
    arr = np.array([abs(v) for v in vals], dtype=float)
    if vmin is None:
        vmin = arr.min() if arr.size else 0.0
    if vmax is None:
        vmax = arr.max() if arr.size else 1.0
    if math.isclose(vmin, vmax):
        return np.clip(np.full_like(arr, (out_min + out_max) / 2.0), out_min, out_max)
    scaled = (arr - vmin) / (vmax - vmin)
    return out_min + scaled * (out_max - out_min)


def plot_power_map(G, pos=None, power_attr='P', node_voltage_attr='V',
                   title=None, min_edge_width=0.6, max_edge_width=8.0,
                   node_size_scale=1000, arrow_size=18, connection_rad=0.08,
                   ax=None):
    """Draw a directed network power map where:
    - Edge thickness ~ abs(power_attr) (transmission magnitude)
    - Arrow direction shows flow direction
    - Node radius ~ voltage magnitude (node_voltage_attr)

    Parameters:
    - G: networkx.DiGraph (or Graph). Edges should have `power_attr` numeric.
    - pos: dict node->(x,y). If None, uses `pos` node attribute or spring_layout.
    - power_attr: edge attribute name for power values (P or Q)
    - node_voltage_attr: node attribute name for voltage magnitude
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if pos is None:
        # prefer node attribute 'pos' if present
        if all('pos' in G.nodes[n] for n in G.nodes()):
            pos = {n: G.nodes[n]['pos'] for n in G.nodes()}
        else:
            pos = nx.spring_layout(G, seed=42)

    # Node sizes from voltage magnitude
    voltages = [G.nodes[n].get(node_voltage_attr, 1.0) for n in G.nodes()]
    vmax = max(voltages) if voltages else 1.0
    vmin = min(voltages) if voltages else 0.0
    # scale to area (so radius visually maps): node_size_scale * (V/Vmax)
    if math.isclose(vmax, 0.0):
        node_sizes = [node_size_scale * 0.5 for _ in voltages]
    else:
        node_sizes = [node_size_scale * (v / vmax) if vmax else node_size_scale for v in voltages]

    # Edge widths and colors based on power_attr
    edge_values = [G[u][v].get(power_attr, 0.0) for u, v in G.edges()]
    if edge_values:
        widths = _scale_values(edge_values, out_min=min_edge_width, out_max=max_edge_width)
    else:
        widths = []
    # color by sign: positive red, negative blue, zero grey
    edge_colors = []
    for val in edge_values:
        if val > 0:
            edge_colors.append('red')
        elif val < 0:
            edge_colors.append('blue')
        else:
            edge_colors.append('grey')

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='orange', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    # Draw edges with arrows. Use small arc radius to separate parallel edges visually.
    nx.draw_networkx_edges(
        G, pos,
        arrowstyle='-|>',
        arrowsize=arrow_size,
        width=list(widths),
        edge_color=edge_colors,
        connectionstyle=f'arc3,rad={connection_rad}',
        ax=ax,
    )

    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title)

    # build simple legend for edge thickness and node size
    # sample edge thickness legend
    if edge_values:
        abs_vals = np.array([abs(v) for v in edge_values])
        if abs_vals.size:
            sample_vals = np.percentile(abs_vals, [25, 75])
            for sv in sample_vals:
                ax.plot([], [], color='k', linewidth=_scale_values([sv], out_min=min_edge_width, out_max=max_edge_width)[0], label=f'{sv:.2f}')
            ax.legend(title=f'|{power_attr}| samples', loc='lower left')


def plot_active_power_map(G, pos=None, node_voltage_attr='V', ax=None, **kwargs):
    return plot_power_map(G, pos=pos, power_attr='P', node_voltage_attr=node_voltage_attr, title='Active Power Map', ax=ax, **kwargs)


def plot_reactive_power_map(G, pos=None, node_voltage_attr='V', ax=None, **kwargs):
    return plot_power_map(G, pos=pos, power_attr='Q', node_voltage_attr=node_voltage_attr, title='Reactive Power Map', ax=ax, **kwargs)


if __name__ == '__main__':
    # small demo graph
    G = nx.DiGraph()
    # positions for a small test
    pos = {
        1: (0, 0),
        2: (1, 0.2),
        3: (0.5, 0.9),
        4: (-0.6, 0.6),
        5: (-0.8, -0.4),
    }
    for n, p in pos.items():
        G.add_node(n, pos=p, V=1.0 + 0.2 * n)  # increasing voltages

    # add edges with P and Q attributes (positive means flow in edge direction)
    edges = [
        (1, 2, {'P': 50.0, 'Q': 15.0}),
        (2, 3, {'P': -30.0, 'Q': -5.0}),
        (3, 4, {'P': 20.0, 'Q': 8.0}),
        (4, 5, {'P': 10.0, 'Q': 2.0}),
        (5, 1, {'P': -5.0, 'Q': -1.0}),
    ]
    G.add_edges_from(edges)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    plot_active_power_map(G, pos=pos, ax=axs[0])
    plot_reactive_power_map(G, pos=pos, ax=axs[1])
    plt.tight_layout()
    plt.show()
