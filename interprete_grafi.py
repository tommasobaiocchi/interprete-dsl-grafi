"""
finale_grafi.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union, List, FrozenSet, Tuple, Callable, Any
from lark import Lark, Token, Tree
import networkx as nx
import matplotlib.pyplot as plt

# =============================================================================
# 1. DOMINI
# =============================================================================

Nodo = Union[int, str]
Arco = Tuple[Nodo, Nodo]

@dataclass(frozen=True)
class GrafoVal:
    nodi: FrozenSet[Nodo] = field(default_factory=frozenset)
    archi: FrozenSet[Arco] = field(default_factory=frozenset)

    def __repr__(self):
        return f"Grafo({len(self.nodi)} nodi, {len(self.archi)} archi)"

# Aggiunta gestione Closure per le funzioni
type EVal = Union[GrafoVal, FrozenSet[Nodo], Nodo, int, bool, str, 'Closure']
type MVal = EVal

@dataclass(frozen=True)
class Loc:
    address: int
    def __repr__(self): return f"@{self.address}"

@dataclass
class Closure:
    params: List[str]
    body: 'Expr'
    env: 'Environment'
    def __repr__(self): return f"<function {self.params}>"

@dataclass
class PrimitiveOp:
    name: str
    fn: Callable[[List[EVal]], EVal]

type DVal = Union[EVal, Loc, Closure, PrimitiveOp]
type Environment = Callable[[str], DVal]

@dataclass
class State:
    store: Callable[[int], MVal]
    next_loc: int


# =============================================================================
# 2. STATO E MEMORIA
# =============================================================================

def empty_store_impl(loc: int) -> MVal:
    raise ValueError(f"Accesso a locazione non allocata: @{loc}")

def empty_state() -> State:
    return State(store=empty_store_impl, next_loc=0)

def allocate(state: State, value: MVal) -> tuple[Loc, State]:
    loc = Loc(state.next_loc)
    old_store = state.store
    def new_store(l: int) -> MVal:
        if l == loc.address: return value
        return old_store(l)
    return loc, State(new_store, state.next_loc + 1)

def update(state: State, loc: Loc, value: MVal) -> State:
    old_store = state.store
    def new_store(l: int) -> MVal:
        if l == loc.address: return value
        return old_store(l)
    return State(new_store, state.next_loc)

def deref(state: State, loc: Loc) -> MVal:
    return state.store(loc.address)


# =============================================================================
# 3. OPERAZIONI GRAFICI
# =============================================================================

def grafo_op(archi_list: List[Arco]) -> GrafoVal:
    nodi = set()
    archi_normalizzati = set()
    for a, b in archi_list:
        nodi.add(a)
        nodi.add(b)
        # Normalizzazione per grafi non orientati
        if a <= b:
            archi_normalizzati.add((a, b))
        else:
            archi_normalizzati.add((b, a))
    return GrafoVal(frozenset(nodi), frozenset(archi_normalizzati))

def unione_op(a: GrafoVal, b: GrafoVal) -> GrafoVal:
    return GrafoVal(a.nodi | b.nodi, a.archi | b.archi)


# =============================================================================
# 4. AST
# =============================================================================

@dataclass(frozen=True)
class ArcoLit:
    a: Any
    b: Any

@dataclass(frozen=True)
class ListaLit:
    items: List[Any]

@dataclass(frozen=True)
class GrafoLit:
    archi: 'Expr'

@dataclass(frozen=True)
class FunCall:
    name: str
    args: List['Expr']

@dataclass(frozen=True)
class Var:
    name: str

@dataclass(frozen=True)
class Let:
    name: str
    expr: 'Expr'
    body: 'Expr'

type Expr = Union[ArcoLit, ListaLit, GrafoLit, FunCall, Var, Let, int, str, bool]

@dataclass
class VarDecl:
    name: str
    expr: Expr

@dataclass
class Assign:
    name: str
    expr: Expr

@dataclass
class PrintCmd:
    expr: Expr

@dataclass
class WhileCmd:
    cond: Expr
    body: 'CmdBlock'

@dataclass
class FunctionDecl:
    name: str
    params: List[str]
    body: Expr

type Cmd = Union[VarDecl, Assign, PrintCmd, WhileCmd, FunctionDecl]

@dataclass
class CmdBlock:
    commands: List[Cmd]


# =============================================================================
# 5. PARSER (LARK)
# =============================================================================

grammar = r"""
    ?start: block
    block: command*
    ?command: var_decl | assign | print_cmd | while_cmd | function_decl
   
    var_decl: "var" ID "=" expr ";" -> var_decl
    assign: ID "<-" expr ";" -> assign
    print_cmd: "print" expr ";" -> print_cmd
    while_cmd: "while" expr "do" block "done" ";" -> while_cmd
   
    function_decl: "function" ID "(" params ")" "=" expr ";" -> function_decl
    params: [ID ("," ID)*]
    ?expr: let_expr | bin_op | atom
    let_expr: "let" ID "=" expr "in" expr -> let
    ?bin_op: expr "+" atom -> add
           | expr "-" atom -> sub
           | expr "*" atom -> mul
           | expr "/" atom -> div
           | expr "<" atom -> lt
           | expr "==" atom -> eq
          
    ?atom: NUMBER -> num
         | ID -> var
         | "arco" "(" expr "," expr ")" -> arco
         | lista -> lista
         | ID "(" args ")" -> fun_call
         | "grafo" "(" lista ")" -> grafo
         | "unione" "(" expr "," expr ")" -> unione
         | "(" expr ")" -> paren
   
    args: [expr ("," expr)*]
    lista: "[" [expr ("," expr)*] "]"
   
    %import common.CNAME -> ID
    %import common.INT -> NUMBER
    %import common.WS
    %ignore WS
"""

parser = Lark(grammar, parser="lalr")

# =============================================================================
# 6. TRASFORMATORE
# =============================================================================

def transform(t: Tree | Token) -> Any:
    if isinstance(t, Token):
        if t.type == 'NUMBER':
            return int(t.value)
        if t.type == 'STRING':
            return t.value[1:-1]
        if t.type == 'ID':
            return Var(t.value)
        return t.value

    rule = t.data
    children = [transform(c) for c in t.children]

    # REGOLE ATOMICHE
    if rule == 'num':
        return children[0]
    if rule == 'str':
        return children[0]
    if rule == 'var':
        return children[0]
    if rule == 'bool_true':
        return True
    if rule == 'bool_false':
        return False
    if rule == 'paren':
        return children[0]

    # Comandi
    if rule == 'block':
        return CmdBlock(children)
    if rule == 'var_decl':
        return VarDecl(children[0].name, children[1])  # children[0] è Var(name)
    if rule == 'assign':
        return Assign(children[0].name, children[1])
    if rule == 'print_cmd':
        return PrintCmd(children[0])
    if rule == 'while_cmd':
        return WhileCmd(children[0], children[1])

    # Gestione funzione
    if rule == 'function_decl':
        fn_name = children[0].name
        params_raw = children[1] if children[1] is not None else []
        params = [p.name if isinstance(p, Var) else str(p) for p in params_raw]
        body = children[2]
        return FunctionDecl(fn_name, params, body)

    if rule == 'params':
        return children

    # Espressioni
    if rule == 'let':
        return Let(children[0].name, children[1], children[2])

    # Operatori binari
    if rule in ('add', 'sub', 'mul', 'div', 'lt', 'eq'):
        op_map = {'add':'+', 'sub':'-', 'mul':'*', 'div':'/', 'lt':'<', 'eq':'=='}
        return FunCall(op_map[rule], [children[0], children[1]])

    # Operazioni grafiche - FIX APPLICATI
    if rule == 'unione':
        return FunCall("unione", [children[0], children[1]])

    if rule == 'grafo':
        return FunCall("grafo", [children[0]])

    if rule == 'arco':
        return ArcoLit(children[0], children[1])

    # Lista - FIX APPLICATI
    if rule == 'lista':
        return ListaLit(children)

    # Fun call
    if rule == 'fun_call':
        name = children[0].name if isinstance(children[0], Var) else children[0]
        if len(children) > 1:
            args_node = children[1]
            if isinstance(args_node, list):
                args = args_node
            else:
                args = [args_node] if args_node is not None else []
        else:
            args = []
        return FunCall(name, args)

    if rule == 'args':
        return children

    raise ValueError(f"Regola sconosciuta nel trasformatore: {rule}")


# =============================================================================
# 7. AMBIENTE E VALUTATORE
# =============================================================================

def empty_env(name: str) -> DVal:
    raise NameError(f"Variabile non definita: {name}")

def bind(env: Environment, name: str, val: DVal) -> Environment:
    def new_env(n: str) -> DVal:
        return val if n == name else env(n)
    return new_env

def create_base_env() -> Environment:
    env = empty_env
    primitives = {
        "+": PrimitiveOp("+", lambda a: a[0] + a[1]),
        "-": PrimitiveOp("-", lambda a: a[0] - a[1]),
        "*": PrimitiveOp("*", lambda a: a[0] * a[1]),
        "/": PrimitiveOp("/", lambda a: a[0] // a[1]),
        "<": PrimitiveOp("<", lambda a: a[0] < a[1]),
        "==": PrimitiveOp("==", lambda a: a[0] == a[1]),
        "grafo": PrimitiveOp("grafo", lambda a: grafo_op(a[0])),
        "unione": PrimitiveOp("unione", lambda a: unione_op(a[0], a[1])),
    }
    for name, op in primitives.items():
        env = bind(env, name, op)
    return env

def eval_expr(expr: Expr, env: Environment, state: State) -> EVal:
    match expr:
        case int(v) | str(v) | bool(v):
            return v

        case Var(name):
            dval = env(name)
            if isinstance(dval, Loc):
                return deref(state, dval)
            return dval

        case Let(name, e, body):
            val = eval_expr(e, env, state)
            return eval_expr(body, bind(env, name, val), state)

        case ArcoLit(a, b):
            va = eval_expr(a, env, state)
            vb = eval_expr(b, env, state)
            return (va, vb)

        case ListaLit(items):
            return [eval_expr(item, env, state) for item in items]

        case GrafoLit(archi):
            lst = eval_expr(archi, env, state)
            return grafo_op(lst)

        case FunCall(name, args):
            dval = env(name)
            arg_vals = [eval_expr(a, env, state) for a in args]

            if isinstance(dval, PrimitiveOp):
                return dval.fn(arg_vals)

            if isinstance(dval, Closure):
                if len(arg_vals) != len(dval.params):
                    raise TypeError(f"Argomenti errati per {name}: attesi {len(dval.params)}, dati {len(arg_vals)}")
                new_env = dval.env
                for param, val in zip(dval.params, arg_vals):
                    new_env = bind(new_env, param, val)
                return eval_expr(dval.body, new_env, state)

            raise TypeError(f"'{name}' non è una funzione chiamabile")

        case _:
            raise NotImplementedError(f"Expr non gestita: {expr}")


# =============================================================================
# 8. ESECUTORE
# =============================================================================

image_counter = 0

def next_image_name():
    global image_counter
    image_counter += 1
    return f"grafo_{image_counter:03d}.png"

def draw_grafo(g: GrafoVal):
    filename = next_image_name()
    G = nx.Graph()
    G.add_nodes_from(g.nodi)
    G.add_edges_from(g.archi)
    plt.figure(figsize=(8, 6))
    nx.draw(G, nx.spring_layout(G, seed=42), with_labels=True, node_color='lightblue', node_size=700)
    plt.title("Grafo Generato")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"[VISUALIZZAZIONE] Salvato: {filename}")

def exec_cmd(cmd: Cmd, env: Environment, state: State) -> tuple[Environment, State]:
    match cmd:
        case VarDecl(name, expr):
            val = eval_expr(expr, env, state)
            loc, new_state = allocate(state, val)
            return bind(env, name, loc), new_state

        case Assign(name, expr):
            val = eval_expr(expr, env, state)
            dval = env(name)
            if not isinstance(dval, Loc):
                raise ValueError(f"Impossibile assegnare a '{name}': non è una variabile modificabile")
            return env, update(state, dval, val)

        case PrintCmd(expr):
            val = eval_expr(expr, env, state)
            if isinstance(val, GrafoVal):
                draw_grafo(val)
                print(val)
            else:
                print(f"Out: {val}")
            return env, state

        case WhileCmd(cond, body):
            saved_next = state.next_loc
            curr_state = state
            while eval_expr(cond, env, curr_state):
                _, curr_state = exec_block(body, env, curr_state)
            return env, State(curr_state.store, saved_next)

        case FunctionDecl(name, params, body):
            closure = Closure(params, body, env)
            return bind(env, name, closure), state

    raise NotImplementedError(f"Comando non implementato: {cmd}")

def exec_block(block: CmdBlock, env: Environment, state: State):
    curr_env, curr_state = env, state
    for cmd in block.commands:
        curr_env, curr_state = exec_cmd(cmd, curr_env, curr_state)
    return curr_env, curr_state


# =============================================================================
# 9. REPL
# =============================================================================

def repl():
    print("=== REPL - DSL Grafi ===")
    print("Digita 'help' per una lista di comandi, 'exit' per uscire")
    env = create_base_env()
    state = empty_state()

    while True:
        try:
            line = input("> ").strip()
            if not line:
                continue

            if line.lower() == "exit":
                print("Chiusura REPL.")
                break

            if line.lower() == "help":
                print("Esempi:")
                print("  var g = grafo([arco(1,2), arco(2,3)]);")
                print("  print g;")
                print("  function doppio(x) = unione(x, x);")
                print("  print doppio(g);")
                print("  var i = 0; while i < 3 do print i; i <- i + 1; done;")
                continue

            tree = parser.parse(line)
            ast = transform(tree)
            env, state = exec_block(ast, env, state)

        except Exception as e:
            print(f"Errore: {e}")

if __name__ == "__main__":
    repl()