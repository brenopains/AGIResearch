"""
Curriculum de Programacao — alimenta o CGE com codigo em multiplas linguagens.

O sistema aprendera conceitos de programacao:
  - Python: estruturas, OOP, functional, async
  - JavaScript: DOM, eventos, promises, classes
  - C++: ponteiros, templates, STL
  - Rust: ownership, traits, borrowing
  - SQL: queries, joins, indices
  - Bash: scripting, pipes, processos

Depois testamos: gerar codigo para um jogo NOVO que usa FLORB/TREK.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.cognitive_graph_engine import CognitiveGraphEngine
from core.multilingual_encoder import MultilingualEncoder

print("=" * 60)
print("CURRICULO DE PROGRAMACAO")
print("=" * 60)

cge = CognitiveGraphEngine(hdc_memory=None, device="cuda")
enc = MultilingualEncoder()

# ─── Codigos de exemplo por linguagem ─────────────────────

CODE_CURRICULUM = {
    "python_basics": """
Python is a programming language. Variables store data: x = 5, name = "hello".
Lists hold multiple items: items = [1, 2, 3]. Dictionaries map keys to values.
Functions are defined with def: def add(a, b): return a + b.
Classes create objects: class Dog: def bark(self): print("woof").
If statements branch: if x > 0: print("positive"). Loops iterate: for i in range(5).
List comprehensions: squares = [x*x for x in range(10)].
Error handling: try: risky_code() except Exception as e: print(e).
""",
    "python_advanced": """
Decorators wrap functions: @timer def func(): pass.
Generators yield values: def gen(): yield 1; yield 2.
Context managers use 'with': with open("file") as f: data = f.read().
Async/await for concurrency: async def fetch(): await asyncio.sleep(1).
Type hints: def add(a: int, b: int) -> int: return a + b.
Dataclasses: @dataclass class Point: x: float; y: float.
Properties: @property def age(self): return self._age.
""",
    "javascript": """
JavaScript runs in browsers and servers. Variables: let x = 5; const y = "hello".
Functions: function add(a, b) { return a + b; }. Arrow: const add = (a,b) => a+b.
Objects: const person = { name: "Alice", age: 30 }.
Arrays: const items = [1,2,3]; items.push(4); items.map(x => x*2).
Promises: fetch(url).then(res => res.json()).catch(err => console.log(err)).
Async/await: async function load() { const res = await fetch(url); }.
DOM: document.querySelector('.class').addEventListener('click', handler).
Classes: class Animal { constructor(name) { this.name = name; } speak() {} }.
Modules: import { Component } from './module'; export default Component.
""",
    "cpp": """
C++ is a compiled language with manual memory management.
Variables: int x = 5; float y = 3.14; std::string name = "hello".
Functions: int add(int a, int b) { return a + b; }.
Classes: class Dog { public: void bark(); private: int age; }.
Pointers: int* ptr = &x; *ptr = 10. References: int& ref = x.
Templates: template<typename T> T max(T a, T b) { return a > b ? a : b; }.
STL: std::vector<int> vec; vec.push_back(5); std::sort(vec.begin(), vec.end()).
Smart pointers: std::unique_ptr<Dog> dog = std::make_unique<Dog>().
Virtual functions enable polymorphism: virtual void speak() = 0.
""",
    "rust": """
Rust is a systems language with memory safety.
Variables: let x = 5; let mut y = 10; const MAX: u32 = 100.
Functions: fn add(a: i32, b: i32) -> i32 { a + b }.
Ownership: each value has one owner. Ownership transfers on assignment.
Borrowing: & reference borrows without owning. &mut for mutable borrow.
Structs: struct Point { x: f64, y: f64 }.
Enums: enum Result<T, E> { Ok(T), Err(E) }.
Match: match value { Pattern1 => code, _ => default }.
Traits: trait Draw { fn draw(&self); } impl Draw for Circle { fn draw(&self) {} }.
Error handling: fn read() -> Result<String, Error> { Ok(contents) }.
""",
    "sql": """
SQL queries databases. SELECT * FROM users WHERE age > 18.
INSERT INTO users (name, email) VALUES ('Alice', 'alice@example.com').
UPDATE users SET age = 30 WHERE id = 1.
DELETE FROM users WHERE id = 1.
JOIN combines tables: SELECT * FROM orders JOIN users ON orders.user_id = users.id.
GROUP BY aggregates: SELECT department, COUNT(*) FROM employees GROUP BY department.
INDEX speeds queries: CREATE INDEX idx_name ON users(name).
Transactions: BEGIN TRANSACTION; COMMIT; ROLLBACK.
""",
    "bash": """
Bash is a shell scripting language. Variables: name="hello"; echo $name.
Conditionals: if [ -f file.txt ]; then echo "exists"; fi.
Loops: for i in {1..5}; do echo $i; done.
Functions: function greet() { echo "Hello $1"; }.
Pipes: cat file.txt | grep pattern | sort | uniq.
Redirection: command > output.txt 2> error.txt.
Processes: command & (background); kill PID; ps aux.
""",
    "algorithms": """
Sorting: bubble sort O(n^2), quick sort O(n log n), merge sort O(n log n).
Searching: binary search O(log n) on sorted arrays, linear search O(n).
Graph algorithms: BFS finds shortest path, DFS explores all paths.
Dijkstra finds shortest weighted path. A* uses heuristics for pathfinding.
Dynamic programming: Fibonacci with memoization, knapsack problem.
Trees: binary trees, binary search trees, heaps, tries.
Hash tables: O(1) average lookup, O(n) worst case.
Recursion: function calls itself. Base case stops recursion.
Big O notation describes algorithm efficiency: O(1), O(n), O(n^2), O(log n).
""",
    "game_dev": """
Game development uses game loops: while running: process_input(); update(); render().
Game objects have position, velocity, and state: x, y, vx, vy, hp.
Collision detection: check if two rectangles overlap: AABB collision.
Sprite rendering: load images, draw them at positions each frame.
Physics: gravity pulls objects down. Velocity changes position each frame.
State machines manage game states: menu, playing, paused, game_over.
AI behaviors: finite state machines for enemy AI (patrol, chase, attack).
Entity Component System: entities are IDs, components hold data, systems process.
""",
}

# ─── Alimenta CGE ─────────────────────────────────────────

print(f"\nAlimentando CGE com {len(CODE_CURRICULUM)} modulos de programacao...\n")

for module_name, code_text in CODE_CURRICULUM.items():
    t0 = time.time()
    cge.write_text_concepts(code_text)
    elapsed = time.time() - t0
    print(f"  [{module_name:20s}] {len(code_text):4d} chars em {elapsed:.2f}s")

print(f"\nCGE apos curriculum de programacao: {len(cge.nodes)} nodes, {len(cge.edges)} edges")

# ─── Salva estado ─────────────────────────────────────────

import torch
save_path = "D:/novamind_data/backup/cge_programming_state.pth"
torch.save({
    "nodes": cge.nodes,
    "edges": cge.edges,
    "n_nodes": len(cge.nodes),
    "n_edges": len(cge.edges),
}, save_path)
print(f"\nEstado salvo em: {save_path}")

# ─── Teste de consulta ────────────────────────────────────

print(f"\n=== TESTE: CONSULTA MULTILINGUE DE PROGRAMACAO ===")

queries = [
    ("funcao", "funcao em portugues"),
    ("class", "classe em ingles"),
    ("loop", "loop em ingles"),
    ("recursao", "recursao em portugues"),
    ("ponteiro", "ponteiro em portugues"),
    ("ownership", "ownership em rust"),
    ("game loop", "game loop"),
]

for qtext, desc in queries:
    hdv = enc.encode_bipolar(qtext)
    results = cge.query_attention(hdv, top_k=5)
    sims = [(nid, f"{sim:.4f}") for nid, _, sim in results[:5]]
    print(f"\n  '{qtext}' ({desc}):")
    for nid, s in sims:
        print(f"    {nid:35s} sim={s}")
