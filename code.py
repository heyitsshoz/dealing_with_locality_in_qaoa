# Import the NetworkX library for graph creation, manipulation, and analysis
import networkx as nx
# Import NumPy for efficient array operations, mathematical functions, and random generation
import numpy as np
# Import Matplotlib to visualize and draw the generated graph topologies
import matplotlib.pyplot as plt
# Import core Qiskit components: QuantumCircuit for building circuits, transpile for compilation
from qiskit import QuantumCircuit, transpile
# Import ParameterVector to handle dynamic arrays of trainable variational parameters
from qiskit.circuit import ParameterVector
# Import SparsePauliOp to construct the cost Hamiltonian operators efficiently
from qiskit.quantum_info import SparsePauliOp
# Import AerSimulator from Qiskit Aer to execute high-performance simulations
from qiskit_aer import AerSimulator
# Import minimize from SciPy to perform classical parameter optimization
from scipy.optimize import minimize
# Import the OS module to interact with system-level environment variables
import os
# Import the SYS module to manipulate standard input/output streams
import sys
# Import Multiprocessing to parallelize optimization restarts across available system resources
import multiprocessing as mp
# Import Time to track execution durations and timestamp execution logs
import time
import datetime
# Import CSV module to export structural metrics and results to structured files
import csv

# Define a custom logging class to duplicate standard output to both the console and a text file
# WHAT IT DOES: Intercepts everything you would normally `print()` to the terminal.
# WHAT IT MEANS: It allows the script to simultaneously print to your screen AND save that exact text into a log.txt file.
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            if not getattr(f, 'closed', False):
                f.write(obj)
                f.flush() # Forces the system to write to the file immediately (real-time logging)
    def flush(self):
        for f in self.files:
            if not getattr(f, 'closed', False):
                f.flush()

# Helper function to visualize and save the generated base graph topology
# WHAT IT DOES: Takes a NetworkX graph object and draws it using a spring layout.
# WHAT IT MEANS: Gives you a visual PNG file of the starting problem before any shortcuts are added.
def visualize_graph(G, title_str, output_filename):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42) # seed=42 ensures the graph looks exactly the same every time it is drawn
    nx.draw(
        G, pos,
        with_labels=True,
        node_color='lightblue',
        edge_color='gray',
        node_size=500,
        font_size=10,
        font_weight='bold'
    )
    plt.title(title_str)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Visualization] Saved base graph plot to: {output_filename}")

# Helper function to visualize the modified graph with shortcut edges explicitly highlighted
# WHAT IT DOES: Draws the original graph, but overlays the newly calculated shortcut edges.
# WHAT IT MEANS: Visually proves how the diameter is being shrunk by showing the new edges in a dashed green color.
def visualize_modified_graph(G, added_edges, title_str, output_filename):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    G_full = G.copy()
    G_full.add_edges_from(added_edges)
    nx.draw_networkx_nodes(G_full, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G_full, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(G_full, pos, edgelist=G.edges(), edge_color='gray', width=1.5)
    if added_edges:
        nx.draw_networkx_edges(G_full, pos, edgelist=added_edges, edge_color='green', style='dashed', width=2.0)
    plt.title(title_str, fontsize=12, fontweight='bold')
    plt.axis('off')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Visualization] Saved modified shortcut graph plot to: {output_filename}")

# Helper function to visualize the final measured MaxCut state
# WHAT IT DOES: Takes the final binary answer (e.g., '10100') and colors the graph nodes based on the 1s and 0s.
# WHAT IT MEANS: Shows you the physical "cut" the quantum computer found. Red lines represent edges that were successfully cut.
def visualize_cut_graph(G, bitstring, title_str, output_filename):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    bits = [int(b) for b in bitstring[::-1]] # Reverses the string because Qiskit orders qubits right-to-left
    nodes_p0 = [node for node in G.nodes() if bits[node] == 0]
    nodes_p1 = [node for node in G.nodes() if bits[node] == 1]
    cut_edges = [(u, v) for u, v in G.edges() if bits[u] != bits[v]]
    internal_edges = [(u, v) for u, v in G.edges() if bits[u] == bits[v]]
    
    nx.draw_networkx_edges(G, pos, edgelist=internal_edges, edge_color='gray', style='dashed', alpha=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, edge_color='red', width=2.0)
    nx.draw_networkx_nodes(G, pos, nodelist=nodes_p0, node_color='skyblue', node_size=600, label='Partition 0')
    nx.draw_networkx_nodes(G, pos, nodelist=nodes_p1, node_color='orange', node_size=600, label='Partition 1')
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    plt.title(title_str, fontsize=12, fontweight='bold')
    plt.legend(scatterpoints=1, loc='upper right')
    plt.axis('off')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Visualization] Saved marked MaxCut outcome plot to: {output_filename}")

# Core Modular QAOA Circuit Assembly Engine
# WHAT IT DOES: This class builds the quantum circuit, applies the math for the different mixers, and evaluates the circuit's energy.
class OptimizedQAOAEngine:
    def __init__(self, G, e_new, p, mixer_case=1, weights=None, sim_backend=None, max_cut_est=1, graph_type="Regular"):
        self.G = G
        self.e_new = e_new
        self.p = p
        self.mixer_case = mixer_case
        self.weights = weights
        self.n = G.number_of_nodes()
        self.sim = sim_backend if sim_backend else AerSimulator(method='statevector', device='GPU')
        self.max_cut_est = max_cut_est
        self.graph_type = graph_type
        self.opt_level = 0
        
        self.iteration_count = 0
        self.best_cost = float('inf')
        self.best_params = None
        
        self.orig_edges = list(G.edges())
        self.num_orig_edges = len(self.orig_edges)
        
        # Build the Cost Hamiltonian (The math formula representing the MaxCut problem)
        paulis = []
        for u, v in self.orig_edges:
            pauli_str = ['I'] * self.n
            pauli_str[u] = 'Z'; pauli_str[v] = 'Z'
            pauli_str.reverse()
            paulis.append("".join(pauli_str))
        self.hamiltonian = SparsePauliOp(paulis, coeffs=[-0.5] * len(paulis))
        self.offset = 0.5 * len(paulis) # The mathematical offset to convert Ising Energy into MaxCut Score
        
        self.is_dynamic = isinstance(self.e_new, list) and len(self.e_new) > 0 and isinstance(self.e_new[0], list)
        if self.is_dynamic:
            self.num_new_edges_per_layer = [len(layer_e) for layer_e in self.e_new]
        else:
            self.num_new_edges_per_layer = [len(self.e_new)] * p
            
        # Dynamically compute parameter budget
        # WHAT IT MEANS: For BOTH Case 1 and Case 2, the baseline is orig_edges + nodes (Rx) + new_edges (shortcuts).
        self.num_params = sum(self.n + self.num_orig_edges + n_new for n_new in self.num_new_edges_per_layer)
                
        self.params_vec = ParameterVector('θ', self.num_params) # Creates placeholder variables for the optimizer to fill
        
        # Assemble Variational Quantum Circuit
        qc = QuantumCircuit(self.n)
        qc.h(range(self.n))  # Strict initial state: All-plus state vector injection (Equal probability of all cuts)
        
        param_idx = 0  
        for layer in range(p):
            layer_edges = self.e_new[layer] if self.is_dynamic else self.e_new
            layer_weights = self.weights[layer] if self.is_dynamic and self.weights else self.weights
            
            # --- Problem Hamiltonian Section ---
            # WHAT IT DOES: Applies an Rzz gate to every edge in the original graph. This checks if the nodes are in different sets.
            for u, v in self.orig_edges:
                qc.rzz(self.params_vec[param_idx], u, v)
                param_idx += 1
                
            # --- Base Standard Mixer Layer ---
            # WHAT IT DOES: Standard Transverse Field. Applies Rx to every node to independently flip bits.
            for node in range(self.n):
                qc.rx(2 * self.params_vec[param_idx], node)
                param_idx += 1
                    
            # --- Shortcut Driven Driver Section ---
            # WHAT IT DOES: Applies interactions to the newly added shortcut edges.
            if self.mixer_case == 1:
                # Case 1: Standard Rxx only on the shortcuts
                for u, v in layer_edges:
                    w = layer_weights.get(tuple(sorted((u, v))), 1.0) if layer_weights else 1.0
                    qc.rxx(2 * self.params_vec[param_idx] * w, u, v)
                    param_idx += 1
            elif self.mixer_case == 2:
                # Case 2: (XX + YY) Parity-Preserving Exchange on the shortcuts (each with a single independent angle)
                for u, v in layer_edges:
                    w = layer_weights.get(tuple(sorted((u, v))), 1.0) if layer_weights else 1.0
                    qc.rxx(2 * self.params_vec[param_idx] * w, u, v)
                    qc.ryy(2 * self.params_vec[param_idx] * w, u, v)
                    param_idx += 1
        
        # Finalize the circuit for simulation
        qc.save_expectation_value(self.hamiltonian, qc.qubits)
        self.base_qc = transpile(qc, backend=self.sim, optimization_level=self.opt_level, seed_transpiler=42)
        self.circuit_depth = self.base_qc.depth()

    # WHAT IT DOES: The objective function evaluated by the SciPy optimizer.
    def __call__(self, params_values):
        self.iteration_count += 1
        param_dict = {p: v for p, v in zip(self.params_vec, params_values) if p in self.base_qc.parameters}
        bound_qc = self.base_qc.assign_parameters(param_dict, inplace=False)
        try:
            # Runs the circuit with current parameters and extracts the energy cost
            result = self.sim.run(bound_qc).result()
            exp_val = result.data(0).get("expectation_value")
            if exp_val is None: return 1e6 
            cost = -(self.offset + float(exp_val)) # Negative because classical optimizers search for minimums
            
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_params = np.copy(params_values)
                current_ar = (-cost) / self.max_cut_est if self.max_cut_est > 0 else 0
                print(f"    [Trace] Iter: {self.iteration_count} | Cost: {cost:.4f} | AR: {current_ar:.4f} | Params: {self.num_params} | Depth: {self.circuit_depth}")
            
            return cost
        except Exception as e:
            return 1e6

    # WHAT IT DOES: Once optimization is done, this extracts the actual binary answer with the highest probability.
    def get_highest_probability_bitstring(self, optimal_params):
        qc = QuantumCircuit(self.n)
        qc.h(range(self.n))
        
        param_idx = 0  
        for layer in range(self.p):
            layer_edges = self.e_new[layer] if self.is_dynamic else self.e_new
            layer_weights = self.weights[layer] if self.is_dynamic and self.weights else self.weights
            
            for u, v in self.orig_edges:
                qc.rzz(self.params_vec[param_idx], u, v)
                param_idx += 1
                
            for node in range(self.n):
                qc.rx(2 * self.params_vec[param_idx], node)
                param_idx += 1
                
            if self.mixer_case == 1:
                for u, v in layer_edges:
                    w = layer_weights.get(tuple(sorted((u, v))), 1.0) if layer_weights else 1.0
                    qc.rxx(2 * self.params_vec[param_idx] * w, u, v)
                    param_idx += 1
            elif self.mixer_case == 2:
                for u, v in layer_edges:
                    w = layer_weights.get(tuple(sorted((u, v))), 1.0) if layer_weights else 1.0
                    qc.rxx(2 * self.params_vec[param_idx] * w, u, v)
                    qc.ryy(2 * self.params_vec[param_idx] * w, u, v)
                    param_idx += 1
                
        # Saves the full mathematical statevector instead of sampling, guaranteeing we find the absolute peak
        qc.save_statevector()
        meas_qc = transpile(qc, backend=self.sim, optimization_level=self.opt_level, seed_transpiler=42)
        param_dict = {p: v for p, v in zip(self.params_vec, optimal_params) if p in meas_qc.parameters}
        bound_qc = meas_qc.assign_parameters(param_dict, inplace=False)
        
        result = self.sim.run(bound_qc).result()
        statevector = result.get_statevector(bound_qc)
        probabilities = np.abs(statevector)**2 # Born rule: Probability = amplitude squared
        return f"{np.argmax(probabilities):0{self.n}b}"

# WHAT IT DOES: The individual worker thread logic for parallel processing
# WHAT IT MEANS: Prevents the optimizer from getting stuck in a local minimum by giving each thread a different random starting point.
def _worker_optimized_pc(config):
    G, e_new, p, mixer_case, weights, max_iter, max_cut_est, graph_type = config
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    
    sim_backend = AerSimulator(method='statevector', device='GPU', precision='double')
    engine = OptimizedQAOAEngine(G, e_new, p, mixer_case=mixer_case, weights=weights, sim_backend=sim_backend, max_cut_est=max_cut_est, graph_type=graph_type)
    
    init_params = np.random.uniform(-0.05, 0.05, engine.num_params)
    param_bounds = [(-np.pi, np.pi) for _ in range(engine.num_params)]
    
    # Executes the L-BFGS-B gradient-based optimization
    res = minimize(
        engine, init_params, method='L-BFGS-B', bounds=param_bounds, options={'maxiter': max_iter, 'ftol': 1e-7}
    )
    best_bitstring = engine.get_highest_probability_bitstring(res.x)
    return res, -res.fun, res.x, engine.iteration_count, engine.circuit_depth, best_bitstring

# WHAT IT DOES: The conductor that spins up multiple worker threads using Python multiprocessing.
def optimize_qaoa_optimized(G, e_new, p, mixer_case=1, num_restarts=8, weights=None, max_cut_est=1, graph_type="Regular"):
    n_workers = min(mp.cpu_count(), num_restarts)
    max_iter = 1000 
    config = (G, e_new, p, mixer_case, weights, max_iter, max_cut_est, graph_type)
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=n_workers) as pool:
        results = pool.map(_worker_optimized_pc, [config for _ in range(num_restarts)])
        
    best_res, best_value, best_params, total_iters, depth, best_bitstring = max(results, key=lambda x: x[1])
    return best_res, best_value, best_params, total_iters, depth, best_bitstring

# WHAT IT DOES: A classical sanity-check function. Takes a bitstring and counts the actual edges cut.
def evaluate_bitstring_cut(G, bitstring):
    bits = [int(b) for b in bitstring[::-1]]
    return sum(1 for u, v in G.edges() if bits[u] != bits[v])

# WHAT IT DOES: Solves MaxCut perfectly using classical brute force by checking all 2^N possible combinations.
# WHAT IT MEANS: We use this as the denominator to calculate the Approximation Ratio (AR).
def exact_maxcut_solver(G):
    n = G.number_of_nodes()
    edges = np.array(list(G.edges()), dtype=np.int32)
    num_states = 1 << (n - 1)
    states = np.arange(0, num_states, dtype=np.uint32)
    bits = ((states[:, None] >> np.arange(n)) & 1).astype(np.uint8)
    cut = np.sum(bits[:, edges[:, 0]] ^ bits[:, edges[:, 1]], axis=1)
    return int(cut.max())

# --- Comprehensive Graph Topology Generator Block ---
# WHAT IT DOES: Depending on the user's CLI choice, this generates the required graph topology.
def generate_custom_topology(n, choice, context_params):
    actual_n = max(n, 8)
    if choice == 1:  # Regular Graph
        k = context_params.get('k', 3)
        if (actual_n * k) % 2 != 0: actual_n += 1
        for _ in range(1000):
            G = nx.random_regular_graph(k, actual_n)
            if nx.is_connected(G) and nx.diameter(G) >= 3: return nx.convert_node_labels_to_integers(G)
        return nx.cycle_graph(actual_n) # Fallback: 2-Regular graph
        
    elif choice == 2:  # Erdős-Rényi Graph
        p_prob = context_params.get('p', 0.5)
        for _ in range(1000):
            G = nx.erdos_renyi_graph(actual_n, p_prob)
            if nx.is_connected(G) and nx.diameter(G) >= 3: return nx.convert_node_labels_to_integers(G)
        return nx.path_graph(actual_n) # Fallback: Path graph
        
    elif choice == 3:  # Bipartite Graph
        n1 = actual_n // 2; n2 = actual_n - n1
        for _ in range(1000):
            G = nx.bipartite.random_graph(n1, n2, p=0.4)
            if nx.is_connected(G) and nx.diameter(G) >= 3: return nx.convert_node_labels_to_integers(G)
        return nx.path_graph(actual_n) # Fallback: Path graph
        
    elif choice == 4:  # Large Diameter Tree Graph
        for _ in range(500):
            G = nx.random_labeled_tree(actual_n)
            if nx.diameter(G) >= 4: return G
        return nx.path_graph(actual_n) # Fallback: Path graph

# WHAT IT DOES: The algorithm that decides which exact nodes to connect to shrink the diameter.
# WHAT IT MEANS: It finds the two nodes that are furthest apart and connects them, repeating until target diameter is met.
# --- BUG FIX: Increased k_max to 10000 so large graphs like N=16 can successfully shrink all the way down to Diameter 1 ---
def get_edges_to_target_diameter(G, target_diam, k_max=10000):
    G_aug = G.copy()
    added = []
    used_edges = set()
    while len(added) < k_max:
        d = nx.diameter(G_aug)
        if d <= target_diam: break
        lengths = dict(nx.shortest_path_length(G_aug))
        farthest = [(u, v) for u in lengths for v, dist in lengths[u].items() if dist == d and u < v]
        farthest = [e for e in farthest if e not in used_edges and (e[1], e[0]) not in used_edges]
        if not farthest: break
        u, v = farthest[np.random.randint(len(farthest))]
        if not G_aug.has_edge(u, v):
            G_aug.add_edge(u, v)
            added.append((u, v))
            used_edges.add((u, v))
            used_edges.add((v, u))
        else: break
    return added

# Orchestrate Research Experiment Running Loops
# WHAT IT DOES: The main execution script that drives user interaction and logic loops.
def run_experiment():
    print(f"{'='*90}")
    print(f"MODULAR QAOA STRUCTURAL RESEARCH SUITE | L-BFGS-B | DUAL MIXER EXECUTION | MAX_ITER=1000")
    print(f"{'='*90}\n")

    # 1. Graph Topology Selector Panel
    # WHAT IT DOES: Asks the user which foundational graph theory structure to evaluate.
    print("--- BASE GRAPH TOPOLOGY CONFIGURATOR ---")
    print(" 1 : Uniform Random Regular Graph Family")
    print(" 2 : Erdős-Rényi Stochastic Network Model")
    print(" 3 : Random Connected Bipartite Architecture")
    print(" 4 : High Diameter Random Tree Profile")
    try:
        graph_choice = int(input("Select Topology Family Choice [1-4] (default 1): ").strip() or 1)
    except ValueError:
        graph_choice = 1

    # 2. Structural Dimensional Controls
    # WHAT IT DOES: Gathers explicit parameters like Vertex Count, Degree (for Regular), or Probability (for Erdos-Renyi).
    try:
        n_nodes_input = int(input("\nEnter baseline node vertex size N (N >= 8) [default: 12]: ").strip() or 12)
    except ValueError:
        n_nodes_input = 12
    actual_n = max(n_nodes_input, 8)

    # Context Parameters Collection
    context_params = {}
    graph_label = "Regular"
    if graph_choice == 1:
        try:
            k_val = int(input("Enter regular node degree boundary (k) [default: 3]: ").strip() or 3)
        except ValueError:
            k_val = 3
        context_params['k'] = k_val
        graph_label = f"{k_val}-Regular"
    elif graph_choice == 2:
        try:
            p_val = float(input("Enter Erdős-Rényi edge creation probability (p) [default: 0.5]: ").strip() or 0.5)
        except ValueError:
            p_val = 0.5
        context_params['p'] = p_val
        graph_label = f"ErdosRenyi_p{p_val}"
    elif graph_choice == 3:
        graph_label = "Bipartite"
    elif graph_choice == 4:
        graph_label = "Tree"

    try:
        ensemble_size = int(input("Enter structural ensemble size (number of separate random samples) [default: 2]: ").strip() or 2)
    except ValueError:
        ensemble_size = 2

    # Folder/CSV Namespace Management
    # WHAT IT DOES: Dynamically names outputs based on graph choices so nothing gets accidentally overwritten.
    OUTPUT_DIR = f"QAOA_AllMixers_{graph_label}_N{actual_n}_Ens{ensemble_size}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(OUTPUT_DIR, f"detailed_execution_log_{timestamp}.txt")
    log_file = open(log_file_path, "a")
    sys.stdout = Tee(sys.stdout, log_file)

    summary_file_path = os.path.join(OUTPUT_DIR, f"clean_results_summary_{timestamp}.txt")
    with open(summary_file_path, "w") as sf:
        sf.write(f"QAOA RESEARCH STUDY RESULTS SUMMARY - DUAL MIXERS - {time.ctime()}\n")
        sf.write("="*75 + "\n\n")

    print(f"\nInitialization Verified: Folder designated -> {OUTPUT_DIR}")
    
    results = []
    # WHAT IT DOES: Loops through the total number of specific graphs (ensemble size) the user asked to generate.
    for g_idx in range(ensemble_size):
        G = generate_custom_topology(actual_n, graph_choice, context_params)
        d_orig = nx.diameter(G)
        max_cut_est = exact_maxcut_solver(G)
        num_orig_edges = G.number_of_edges()
        
        img_name_base = os.path.join(OUTPUT_DIR, f"base_graph_id{g_idx}_{timestamp}.png")
        visualize_graph(G, f"Base {graph_label} Graph (N={G.number_of_nodes()}, Edges={num_orig_edges}, Diam={d_orig})", img_name_base)
        
        print(f"\n{'*'*85}")
        print(f"INSTANCE {g_idx} | Nodes={G.number_of_nodes()} | Base Edges={num_orig_edges} | Baseline Diameter={d_orig} | MaxCut={max_cut_est}")
        print(f"{'*'*85}")
        
        # WHAT IT DOES: Main testing loop. Gradually adds shortcuts to step the graph down diameter by diameter until it reaches 1.
        for target_diam in range(d_orig, 0, -1):
            if target_diam == d_orig:
                added_edges = []
                weights_dict = None
                G_check = G.copy()
            else:
                added_edges = get_edges_to_target_diameter(G, target_diam=target_diam)
                weights_dict = None  
                G_check = G.copy()
                G_check.add_edges_from(added_edges)
                
            actual_achieved_diam = nx.diameter(G_check)
            added_edges_count = len(added_edges)
            
            if actual_achieved_diam != target_diam and target_diam != 1:
                continue
                
            print(f"\n--- Constructing Shortcut Topology: Target Diam={target_diam} (Achieved: {actual_achieved_diam}), Shortcuts Added={added_edges_count} ---")
            if added_edges_count > 0:
                img_name_mod = os.path.join(OUTPUT_DIR, f"modified_graph_id{g_idx}_Diam{actual_achieved_diam}_{timestamp}.png")
                visualize_modified_graph(G, added_edges, f"Modified Graph (Diam: {d_orig} -> {actual_achieved_diam})\nShortcuts: {added_edges_count}", img_name_mod)
            
            p_sweep_values = [1, 2, 3]
            # WHAT IT DOES: Evaluates the specific shortcut graph at different QAOA layer depths (p).
            for p_layers in p_sweep_values:
                # WHAT IT DOES: Evaluates both the Standard Rxx Mixer (Case 1) and XX+YY Mixer (Case 2) back-to-back automatically.
                for mixer_choice in [1, 2]:
                    
                    # --- TINY CHECK TO SKIP THE REDUNDANT BASELINE RUN ---
                    # When added_edges_count is 0, Case 1 and Case 2 build the exact same circuit.
                    # This skip prevents wasting compute time on an identical calculation.
                    if added_edges_count == 0 and mixer_choice == 2:
                        continue
                        
                    mixer_name = "Standard_Rxx" if mixer_choice == 1 else "XX+YY_Driver"
                    exp_label = f"Mix{mixer_choice}_Diam_{actual_achieved_diam}_p_{p_layers}"
                    applicable_categories = []
                    
                    # Explicity determines which Hypothesis category this run falls under to label it properly.
                    if added_edges_count == 0:
                        applicable_categories.append("Standard_QAOA")
                    elif actual_achieved_diam == 1:
                        applicable_categories.append("Hypo_1")
                    if p_layers == 2 * actual_achieved_diam:
                        applicable_categories.append("Hypo_2")
                    if len(applicable_categories) == 0:
                        applicable_categories.append("Intermediate_Sweep")

                    cat_display = " & ".join(applicable_categories)
                    print(f"  >>> Running Execution Configuration: [{cat_display}] | Mixer: {mixer_name} | (Layers={p_layers}, Shortcuts={added_edges_count})")
                    
                    res, val, opt_params, iters, depth, best_bits = optimize_qaoa_optimized(
                        G, added_edges, p=p_layers, mixer_case=mixer_choice, num_restarts=8, 
                        weights=weights_dict, max_cut_est=max_cut_est, graph_type=graph_label
                    )
                    duration = time.time() - t_start if 't_start' in locals() else 0.0
                    t_start = time.time() # Reset for next slice
                    
                    ar = val / max_cut_est if max_cut_est > 0 else 0
                    params_str = np.array2string(opt_params, precision=4, separator=',', suppress_small=True) if opt_params is not None else "None"
                    measured_cut = evaluate_bitstring_cut(G, best_bits)
                    
                    # WHAT IT DOES: For every category this run satisfied (e.g. Hypo 1 and Hypo 2), export the data and images.
                    for category in applicable_categories:
                        img_name_cut = os.path.join(OUTPUT_DIR, f"outcome_{category}_{exp_label}_id{g_idx}_{timestamp}.png")
                        visualize_cut_graph(G, best_bits, f"[{category}] {mixer_name} (p={p_layers})\nCut Score: {measured_cut}/{max_cut_est} (AR={ar:.2f})", img_name_cut)
                        
                        with open(summary_file_path, "a") as sf:
                            sf.write(f"--- Graph ID: {g_idx} | Mixer: {mixer_name} | Category: {category} | {exp_label} ---\n")
                            sf.write(f"  Circuit Depth: {depth} | Total Params: {len(opt_params)}\n")
                            sf.write(f"  Expectation Cost: {val:.4f} | AR: {ar:.4f} | Classical Cut: {measured_cut}/{max_cut_est}\n\n")
                        
                        results.append({
                            'Hypothesis_Category': category,
                            'Mixer_Type': mixer_name,
                            'Experiment_Label': exp_label,
                            'Graph_Instance_ID': g_idx,
                            'Graph_Type': graph_label,
                            'Total_Vertices': G.number_of_nodes(),
                            'Original_Edges': num_orig_edges,
                            'Base_Diameter': d_orig,
                            'Target_Reduced_Diameter': target_diam,
                            'Achieved_Reduced_Diameter': actual_achieved_diam,
                            'Auxiliary_Shortcuts_Added': added_edges_count,
                            'QAOA_Layers_p': p_layers,
                            'Mixer_Strategy_Case': mixer_choice,
                            'Compiler_Opt_Level': 0, 
                            'Physical_Circuit_Depth': depth,
                            'Optimization_Iterations': iters, 
                            'Converged_Expectation_Value': val,
                            'Approximation_Ratio_AR': ar,
                            'Dominant_Measured_Bitstring': best_bits,
                            'Evaluated_Classical_Cut': measured_cut,
                            'Parameter_Count': len(opt_params),
                            'Trained_Variational_Parameters': params_str
                        })

    # Final cleanup and master CSV saving logic
    if results:
        csv_filename = os.path.join(OUTPUT_DIR, f"qaoa_modular_scaling_results_{timestamp}.csv")
        keys = results[0].keys()
        with open(csv_filename, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
        print(f"\n{'*'*90}\nALL DATA SUCCESSFUL -> Master Folder Located At:\n{os.path.abspath(OUTPUT_DIR)}\n{'*'*90}")
        
    return log_file

if __name__ == '__main__':
    mp.freeze_support()
    log_file_handle = None
    try:
        log_file_handle = run_experiment()
    finally:
        if log_file_handle and not log_file_handle.closed:
            log_file_handle.close()