"""
Experiment 5: Representation Darwin
====================================
A cognitive architecture test to verify:
1. Real Multidimensional Transfer (Geometry, Temporal, Graphs, Logic, Navigation)
2. Abstract Representation Discovery (R_k)
3. Grammar Evolution (auto-modification of the DSL with primitive composability)
4. Scenario B Composition (evolutionary trees of knowledge)
5. Continuous Learning (Manifold Library reuse and survival metrics)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

DEVICE = "cpu"

# Suppress numpy warnings to keep output clean and fast
np.seterr(all='ignore')

# Dictionary to hold dynamically compiled custom primitives and their arities
CUSTOM_PRIMITIVES = {}
CUSTOM_PRIMITIVES_ARITY = {}

# 1. Multi-Domain Task Data Generators

# Domain A: Geometry (Vortex & Spiral)
def make_vortex_data(n=400, seed=42):
    rng = np.random.RandomState(seed)
    r = rng.uniform(1.0, 3.0, n)
    theta = rng.uniform(-1.0, 1.0, n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    r_next = r
    theta_next = theta + 0.3 * r
    x_next = r_next * np.cos(theta_next)
    y_next = r_next * np.sin(theta_next)
    
    X = np.stack([x, y], axis=1)
    Y = np.stack([x_next, y_next], axis=1)
    return X, Y

def make_spiral_data(n=400, seed=43):
    rng = np.random.RandomState(seed)
    r = rng.uniform(1.0, 3.0, n)
    theta = rng.uniform(-1.5, 1.5, n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    r_next = 1.2 * r
    theta_next = theta + 0.5
    x_next = r_next * np.cos(theta_next)
    y_next = r_next * np.sin(theta_next)
    
    X = np.stack([x, y], axis=1)
    Y = np.stack([x_next, y_next], axis=1)
    return X, Y

# Domain B: Temporal Series (Phase Shift)
def make_temporal_data(n=400, seed=44):
    rng = np.random.RandomState(seed)
    t = rng.uniform(0.0, 3.0, n)
    X = t[:, np.newaxis]
    Y = np.stack([np.sin(t + 1.0), np.cos(t + 1.0)], axis=1)
    return X, Y

# Domain C: Graphs & Networks (Node Edge Alignment)
def make_graph_data(n=400, seed=45):
    rng = np.random.RandomState(seed)
    u = rng.randn(n, 2)
    v = rng.randn(n, 2)
    edge_score = u[:, 0]*v[:, 0] + u[:, 1]*v[:, 1]
    
    X = np.concatenate([u, v], axis=1)
    Y = np.stack([edge_score, edge_score * 0.5], axis=1)
    return X, Y

# Domain D: Logic (Boolean XOR - Noise-free)
def make_logic_data(n=400, seed=46):
    rng = np.random.RandomState(seed)
    a = rng.choice([0.0, 1.0], size=n)
    b = rng.choice([0.0, 1.0], size=n)
    
    xor_val = np.logical_xor(a, b).astype(float)
    X = np.stack([a, b], axis=1)
    Y = np.stack([xor_val, xor_val * 0.5], axis=1)
    return X, Y

# Domain E: Grid Navigation (Manhattan Distance)
def make_navigation_data(n=400, seed=47):
    rng = np.random.RandomState(seed)
    x1 = rng.uniform(0.0, 5.0, n)
    y1 = rng.uniform(0.0, 5.0, n)
    x2 = rng.uniform(0.0, 5.0, n)
    y2 = rng.uniform(0.0, 5.0, n)
    
    dist = np.abs(x1 - x2) + np.abs(y1 - y2)
    X = np.stack([x1, y1, x2, y2], axis=1)
    Y = np.stack([dist, dist * 0.5], axis=1)
    return X, Y

# Domain A+B: Composed Wave Vortex (Geometry + Temporal)
def make_composed_data(n=400, seed=48):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-2.0, 2.0, n)
    y = rng.uniform(-2.0, 2.0, n)
    
    r2 = x**2 + y**2
    z1 = r2
    z2 = np.sin(r2)
    
    y1 = z1 + z2
    y2 = z1 - z2
    
    X = np.stack([x, y], axis=1)
    Y = np.stack([y1, y2], axis=1)
    return X, Y


# 2. Dynamic Primitive Compiler
def compile_primitive(expr_str, num_vars):
    """Compiles a string expression tree into a reusable Python function."""
    def func(*args):
        safe_dict = {
            'square': lambda a: a**2,
            'sqrt': lambda a: np.sqrt(np.abs(a)),
            'sin': lambda a: np.sin(a),
            'cos': lambda a: np.cos(a),
            'abs': lambda a: np.abs(a),
            'add': lambda a, b: a + b,
            'sub': lambda a, b: a - b,
            'mul': lambda a, b: a * b,
            'atan2': lambda a, b: np.arctan2(a, b)
        }
        safe_dict.update(CUSTOM_PRIMITIVES)
        
        # Bind inputs v0, v1, ...
        for idx, arg in enumerate(args[:num_vars]):
            safe_dict[f"v{idx}"] = arg
        return eval(expr_str, {"__builtins__": None}, safe_dict)
    return func


# Helper to check parents of a representation pair
def find_parents(expr_pair):
    parents = set()
    for expr in expr_pair:
        for name in CUSTOM_PRIMITIVES.keys():
            rep_base = name.split('_')[0]
            if f"{name}(" in expr:
                parents.add(rep_base)
    return sorted(list(parents))


# 3. Dynamic Grammar Candidate Generator (Pruned step-wise with structural composition constraints)
def generate_expressions(num_vars, X_val):
    terminals = [f"v{i}" for i in range(num_vars)]
    unary_ops = ['square', 'sqrt', 'sin', 'cos', 'abs']
    binary_ops = ['add', 'sub', 'mul']
    
    custom_unary = [name for name, arity in CUSTOM_PRIMITIVES_ARITY.items() if arity == 1]
    custom_binary = [name for name, arity in CUSTOM_PRIMITIVES_ARITY.items() if arity == 2]
    custom_quad = [name for name, arity in CUSTOM_PRIMITIVES_ARITY.items() if arity == 4]
    
    all_unary = unary_ops + custom_unary
    all_binary = binary_ops
    
    # Helper to prune expressions based on signature lookup
    def prune_list(expr_list):
        unique = []
        seen = set()
        for expr in expr_list:
            val = eval_expr(expr, X_val)
            if val is None or np.std(val) < 1e-4:
                continue
            norm_val = (val - np.mean(val)) / (np.std(val) + 1e-6)
            sig = tuple(np.round(norm_val, 2))
            sig_neg = tuple(np.round(-norm_val, 2))
            if sig not in seen and sig_neg not in seen:
                unique.append(expr)
                seen.add(sig)
        return unique

    # Level 0
    L0 = list(terminals)
    
    # Level 1
    L1_cand = []
    for op in all_unary:
        for t in L0:
            L1_cand.append(f"{op}({t})")
            
    # Include custom binary primitives on terminals
    for op in custom_binary:
        for i in range(len(L0)):
            for j in range(len(L0)):
                L1_cand.append(f"{op}({L0[i]},{L0[j]})")
                
    # Include base binary primitives on terminals
    for op in binary_ops:
        for i in range(len(L0)):
            for j in range(len(L0)):
                if op in ['add', 'mul']:
                    if j >= i:
                        L1_cand.append(f"{op}({L0[i]},{L0[j]})")
                else:
                    L1_cand.append(f"{op}({L0[i]},{L0[j]})")
                    
    L1 = prune_list(L1_cand)
    base_set = L0 + L1
    
    # Level 2
    L2_cand = []
    # 1. Unary ops on L1
    for op in all_unary:
        for e in L1:
            L2_cand.append(f"{op}({e})")
            
    # 2. Base binary ops on base_set
    has_any_custom = len(custom_unary) > 0 or len(custom_binary) > 0 or len(custom_quad) > 0
    for op in binary_ops:
        for i in range(len(base_set)):
            for j in range(len(base_set)):
                arg1_has_custom = "R" in base_set[i] and "_f" in base_set[i]
                arg2_has_custom = "R" in base_set[j] and "_f" in base_set[j]
                
                # If there are custom primitives, we only allow Level 2 binary combinations that involve them.
                # If there are no custom primitives yet, we allow all base Level 2 binary combinations.
                if (not has_any_custom) or arg1_has_custom or arg2_has_custom:
                    if op in ['add', 'mul']:
                        if j >= i:
                            L2_cand.append(f"{op}({base_set[i]},{base_set[j]})")
                    else:
                        L2_cand.append(f"{op}({base_set[i]},{base_set[j]})")
                        
    L2 = prune_list(L2_cand)
    all_exprs = base_set + L2
    
    if num_vars >= 2:
        all_exprs.append("atan2(v1,v0)")
        all_exprs.append("atan2(v0,v1)")
        
    if num_vars >= 4:
        for name in custom_quad:
            all_exprs.append(f"{name}(v0,v1,v2,v3)")
            
    return all_exprs


# 4. Expression Evaluator
def eval_expr(expr, X):
    safe_dict = {
        'square': lambda a: a**2,
        'sqrt': lambda a: np.sqrt(np.abs(a)),
        'sin': lambda a: np.sin(a),
        'cos': lambda a: np.cos(a),
        'abs': lambda a: np.abs(a),
        'add': lambda a, b: a + b,
        'sub': lambda a, b: a - b,
        'mul': lambda a, b: a * b,
        'atan2': lambda a, b: np.arctan2(a, b)
    }
    safe_dict.update(CUSTOM_PRIMITIVES)
    
    # Bind variables v0, v1, ...
    for i in range(X.shape[1]):
        safe_dict[f"v{i}"] = X[:, i]
        
    try:
        val = eval(expr, {"__builtins__": None}, safe_dict)
        if np.any(np.isnan(val)) or np.any(np.isinf(val)):
            return None
        return val
    except Exception:
        return None


# 5. Invertibility Validator
def check_invertibility(X_in, Z_in):
    tx = torch.tensor(X_in, dtype=torch.float32)
    tz = torch.tensor(Z_in, dtype=torch.float32)
    
    in_dim = Z_in.shape[1]
    out_dim = X_in.shape[1]
    
    net = nn.Sequential(
        nn.Linear(in_dim, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, out_dim)
    )
    opt = torch.optim.Adam(net.parameters(), 1.5e-2)
    
    for _ in range(250):
        opt.zero_grad()
        loss = F.mse_loss(net(tz), tx)
        loss.backward()
        opt.step()
        
    with torch.no_grad():
        final_loss = F.mse_loss(net(tz), tx).item()
    return final_loss < 1e-2


# 6. Representation Discovery Engine
def discover_representation(X, Y, task_name, is_dynamics):
    num_vars = X.shape[1]
    X_val = X[:100]
    
    # Step-wise pruning generator
    raw_exprs = generate_expressions(num_vars, X_val)
    
    # Prune candidates using signature set lookup
    unique_exprs = []
    seen_signatures = set()
    
    for expr in raw_exprs:
        val = eval_expr(expr, X_val)
        if val is None or np.std(val) < 1e-4:
            continue
        norm_val = (val - np.mean(val)) / (np.std(val) + 1e-6)
        sig = tuple(np.round(norm_val, 3))
        sig_neg = tuple(np.round(-norm_val, 3))
        if sig not in seen_signatures and sig_neg not in seen_signatures:
            unique_exprs.append(expr)
            seen_signatures.add(sig)
            
    print(f"  Unique features after pruning: {len(unique_exprs)}")
    
    # Pre-evaluate unique features on entire training set
    feat_matrix = []
    feat_matrix_Y = []
    valid_features = []
    for expr in unique_exprs:
        val_X = eval_expr(expr, X)
        if val_X is not None:
            feat_matrix.append(val_X)
            valid_features.append(expr)
            if is_dynamics:
                val_Y = eval_expr(expr, Y)
                feat_matrix_Y.append(val_Y)
            
    feat_matrix = np.stack(feat_matrix, axis=1)
    if is_dynamics:
        feat_matrix_Y = np.stack(feat_matrix_Y, axis=1)
        
    # Pre-normalize the features to avoid mean/std overhead inside the loop
    feat_matrix_norm = (feat_matrix - np.mean(feat_matrix, axis=0)) / (np.std(feat_matrix, axis=0) + 1e-6)
    if is_dynamics:
        feat_matrix_Y_norm = (feat_matrix_Y - np.mean(feat_matrix_Y, axis=0)) / (np.std(feat_matrix_Y, axis=0) + 1e-6)
    else:
        Z_out_norm = (Y - np.mean(Y, axis=0)) / (np.std(Y, axis=0) + 1e-6)
    
    # Find pair of features with lowest linear prediction error: Z_out = W * Z_in + B
    candidates = []
    n_feats = len(valid_features)
    for i in range(n_feats):
        for j in range(i+1, n_feats):
            # Construct normalized Z_in directly from pre-normalized features
            Z_in_norm = np.stack([feat_matrix_norm[:, i], feat_matrix_norm[:, j]], axis=1)
            
            # Construct normalized Z_out
            if is_dynamics:
                Z_out_norm = np.stack([feat_matrix_Y_norm[:, i], feat_matrix_Y_norm[:, j]], axis=1)
                
            Z_in_bias = np.concatenate([Z_in_norm, np.ones((len(Z_in_norm), 1))], axis=1)
            try:
                # Direct solve using normal equations
                A = Z_in_bias.T @ Z_in_bias
                B = Z_in_bias.T @ Z_out_norm
                W_B = np.linalg.solve(A, B)
                pred = Z_in_bias @ W_B
                err = np.mean((pred - Z_out_norm)**2)
                if err < 1e-3:
                    f1_in, f2_in = feat_matrix[:, i], feat_matrix[:, j]
                    Z_in = np.stack([f1_in, f2_in], axis=1)
                    candidates.append((err, (valid_features[i], valid_features[j]), Z_in))
            except Exception:
                continue
                
    candidates.sort(key=lambda x: x[0])
    print(f"  Found {len(candidates)} linearizing candidates. Running verification...")
    
    # Verification & Invertibility Check
    for err, pair, Z_in in candidates[:15]:
        if is_dynamics:
            invertible = check_invertibility(X, Z_in)
        else:
            invertible = True # Projection task, no invertibility check needed
            
        if invertible:
            return pair, err, len(raw_exprs)
            
    return None, None, len(raw_exprs)


# 7. Main Darwinian Evaluation Loop
def run_darwin():
    print("=" * 80)
    print("EXPERIMENT 5: REPRESENTATION DARWIN (CONTINUOUS EVOLUTION & TRANSFER)")
    print("=" * 80)
    
    tasks = [
        {"name": "Task 1: Vortex Dynamics (Geometry)", "gen": make_vortex_data, "is_dynamics": True},
        {"name": "Task 2: Spiral Growth (Geometry)", "gen": make_spiral_data, "is_dynamics": True},
        {"name": "Task 3: Phase Shift (Temporal Series)", "gen": make_temporal_data, "is_dynamics": False},
        {"name": "Task 4: Edge Alignment (Graphs)", "gen": make_graph_data, "is_dynamics": False},
        {"name": "Task 5: Boolean XOR (Logic)", "gen": make_logic_data, "is_dynamics": False},
        {"name": "Task 6: Manhattan Path (Navigation)", "gen": make_navigation_data, "is_dynamics": False},
        {"name": "Task 7: Composed Wave Vortex (Geometry + Temporal)", "gen": make_composed_data, "is_dynamics": False}
    ]
    
    manifold_library = {} # Map of abstract representation spaces R_k -> {"exprs": (...), "use_count": int, "parents": list}
    
    for t_idx, task in enumerate(tasks):
        print(f"\n--- Running {task['name']} ---")
        X, Y = task["gen"]()
        num_vars = X.shape[1]
        is_dynamics = task["is_dynamics"]
        
        # 1. Evaluate Cartesian Baseline (normalized fit MSE)
        X_norm = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-6)
        Y_norm = (Y - np.mean(Y, axis=0)) / (np.std(Y, axis=0) + 1e-6)
        X_bias = np.concatenate([X_norm, np.ones((len(X_norm), 1))], axis=1)
        W_base, _, _, _ = np.linalg.lstsq(X_bias, Y_norm, rcond=None)
        base_err = np.mean((X_bias @ W_base - Y_norm)**2)
        print(f"  Cartesian R0 Normalized Fit MSE: {base_err:.4f}")
        
        # 2. Check Manifold Library for reusable representations (Zero-shot check)
        reused = False
        t_start = time.time()
        
        for name, entry in manifold_library.items():
            f1, f2 = entry["exprs"]
            val_f1_in = eval_expr(f1, X)
            val_f2_in = eval_expr(f2, X)
            
            if val_f1_in is not None and val_f2_in is not None:
                Z_in = np.stack([val_f1_in, val_f2_in], axis=1)
                
                if is_dynamics:
                    val_f1_out = eval_expr(f1, Y)
                    val_f2_out = eval_expr(f2, Y)
                    if val_f1_out is None or val_f2_out is None:
                        continue
                    Z_out = np.stack([val_f1_out, val_f2_out], axis=1)
                else:
                    Z_out = Y
                
                # Normalize inputs and targets
                Z_in_norm = (Z_in - np.mean(Z_in, axis=0)) / (np.std(Z_in, axis=0) + 1e-6)
                Z_out_norm = (Z_out - np.mean(Z_out, axis=0)) / (np.std(Z_out, axis=0) + 1e-6)
                
                Z_bias = np.concatenate([Z_in_norm, np.ones((len(Z_in_norm), 1))], axis=1)
                try:
                    A = Z_bias.T @ Z_bias
                    B = Z_bias.T @ Z_out_norm
                    W_Z = np.linalg.solve(A, B)
                    err = np.mean((Z_bias @ W_Z - Z_out_norm)**2)
                    if err < 1e-3:
                        entry["use_count"] += 1
                        reused = True
                        dt = time.time() - t_start
                        print(f"  -> SUCCESS! Reused representation {name} zero-shot in {dt*1000:.2f}ms! (MSE: {err:.2e})")
                        break
                except Exception:
                    continue
                    
        if reused:
            continue
            
        # 3. Spontaneous Discovery Phase
        print("  No reusable representation found in library. Starting search...")
        pair, err, search_space_size = discover_representation(X, Y, task["name"], is_dynamics)
        dt = time.time() - t_start
        
        if pair is not None:
            # Register discovered representation in Library and track parents
            rep_name = f"R{len(manifold_library) + 1}"
            parents = find_parents(pair)
            manifold_library[rep_name] = {
                "exprs": pair,
                "use_count": 1,
                "parents": parents
            }
            print(f"  -> SUCCESS! Discovered representation {rep_name}: {pair} in {dt:.2f}s (MSE: {err:.2e}) | Parents: {parents}")
            
            # INTEGRATE GRAMMAR EVOLUTION:
            # Compile discovered functions into new custom primitives and register them in CUSTOM_PRIMITIVES
            prim_name1 = f"{rep_name}_f1"
            prim_name2 = f"{rep_name}_f2"
            CUSTOM_PRIMITIVES[prim_name1] = compile_primitive(pair[0], num_vars)
            CUSTOM_PRIMITIVES[prim_name2] = compile_primitive(pair[1], num_vars)
            CUSTOM_PRIMITIVES_ARITY[prim_name1] = num_vars
            CUSTOM_PRIMITIVES_ARITY[prim_name2] = num_vars
            print(f"  [Grammar Evolution] Registered compound primitives: {prim_name1}, {prim_name2} (Arity {num_vars})")
        else:
            print(f"  -> FAILED to find invertible representation. Discovery time: {dt:.2f}s")
            
    # 8. Report Darwinian Survival & Reuse Metrics
    print("\n" + "=" * 80)
    print("REPRESENTATION DARWIN - SURVIVAL & EVOLUTION SUMMARY")
    print("=" * 80)
    print(f"Total abstract representations discovered: {len(manifold_library)}")
    for name, entry in manifold_library.items():
        print(f"  - {name}: {entry['exprs']} | Survival Use Count: {entry['use_count']}")
    print(f"Total compiled grammar primitives in vocabulary: {len(CUSTOM_PRIMITIVES)}")
    
    # 9. Print Evolutionary Tree of Knowledge (DAG)
    print("\n" + "=" * 80)
    print("EVOLUTIONARY TREE OF KNOWLEDGE (DAG)")
    print("=" * 80)
    for name, entry in manifold_library.items():
        parents_str = ", ".join(entry["parents"]) if entry["parents"] else "None (Direct Base Coordinates)"
        print(f"  {name} -> Parents: [{parents_str}] | Survival Use: {entry['use_count']}")
    print("=" * 80)


if __name__ == '__main__':
    run_darwin()
