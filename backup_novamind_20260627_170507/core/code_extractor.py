"""
Code Extractor — Extrator de Semântica Lógica via AST
======================================================
Analisa código Python e extrai papéis lógicos (Função, Argumentos, Ação)
usando a AST nativa do Python. Sem regex quebradiços.

Converte:
    def soma(a, b):
        return a + b
Em:
    {'function': 'soma', 'arguments': 'a b', 'action': 'a Add b'}

Esses papéis são então codificados como Role-Filler Binding no espaço HDC.
"""

import ast
from typing import Optional, Dict


class CodeExtractor:
    """Extrator de semântica lógica de código Python."""

    # Mapeamento de operadores AST para texto HDC
    _OP_MAP = {
        'Add': 'Add',
        'Sub': 'Sub',
        'Mult': 'Mult',
        'Div': 'Div',
        'Pow': 'Pow',
        'FloorDiv': 'FloorDiv',
        'Mod': 'Mod',
        'LShift': 'LShift',
        'RShift': 'RShift',
        'BitOr': 'BitOr',
        'BitXor': 'BitXor',
        'BitAnd': 'BitAnd',
        'Eq': 'Eq',
        'NotEq': 'NotEq',
        'Lt': 'Lt',
        'LtE': 'LtE',
        'Gt': 'Gt',
        'GtE': 'GtE',
        'Is': 'Is',
        'IsNot': 'IsNot',
        'In': 'In',
        'NotIn': 'NotIn',
        'And': 'And',
        'Or': 'Or',
    }

    @staticmethod
    def extract(code_string: str) -> Optional[Dict[str, str]]:
        """
        Analisa a AST do Python e extrai os papéis lógicos do código.

        Args:
            code_string: código Python como string

        Returns:
            Dict com 'function', 'arguments', 'action' ou None se inválido
        """
        try:
            tree = ast.parse(code_string)
        except SyntaxError:
            return None

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return CodeExtractor._extract_function(node)

        # Se não achou FunctionDef, tenta módulo simples (ex: x = 1)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = [t.id if isinstance(t, ast.Name) else '' for t in node.targets]
                action = CodeExtractor._extract_expr(node.value)
                return {
                    'function': 'assign',
                    'arguments': ' '.join(t for t in targets if t),
                    'action': action,
                }

        return None

    @staticmethod
    def _extract_function(node: ast.FunctionDef) -> Dict[str, str]:
        """Extrai de um nó FunctionDef da AST."""
        func_name = node.name
        args = [arg.arg for arg in node.args.args]
        args_str = ' '.join(args) if args else ''

        # Procura a ação principal (return, assign, call)
        action_str = CodeExtractor._find_action(node)

        return {
            'function': func_name,
            'arguments': args_str,
            'action': action_str,
        }

    @staticmethod
    def _find_action(node: ast.AST) -> str:
        """Encontra a ação principal dentro de um nó da AST."""
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                return CodeExtractor._extract_expr(child.value)
            if isinstance(child, ast.Assign):
                action = CodeExtractor._extract_expr(child.value)
                targets = [t.id if isinstance(t, ast.Name) else '' for t in child.targets]
                target_str = ' '.join(t for t in targets if t)
                return f'{target_str} assign {action}' if target_str else action
            if isinstance(child, ast.Expr):
                if isinstance(child.value, ast.Call):
                    return CodeExtractor._extract_expr(child.value)
        return ''

    @staticmethod
    def _extract_expr(node: ast.AST) -> str:
        """Extrai uma expressão AST como string semântica."""
        if isinstance(node, ast.BinOp):
            left = CodeExtractor._extract_expr(node.left)
            op = CodeExtractor._OP_MAP.get(node.op.__class__.__name__, 'op')
            right = CodeExtractor._extract_expr(node.right)
            return f'{left} {op} {right}'
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return f'"{node.value}"'
            return str(node.value)
        elif isinstance(node, ast.Call):
            func = CodeExtractor._extract_expr(node.func)
            args = ' '.join(CodeExtractor._extract_expr(a) for a in node.args)
            return f'{func}({args})' if args else f'{func}()'
        elif isinstance(node, ast.Attribute):
            return f'{CodeExtractor._extract_expr(node.value)}.{node.attr}'
        elif isinstance(node, ast.UnaryOp):
            op_map = {'UAdd': 'pos', 'USub': 'neg', 'Not': 'not'}
            op = op_map.get(node.op.__class__.__name__, 'unary')
            return f'{op} {CodeExtractor._extract_expr(node.operand)}'
        elif isinstance(node, ast.Compare):
            left = CodeExtractor._extract_expr(node.left)
            parts = [left]
            for op, comparator in zip(node.ops, node.comparators):
                op_name = CodeExtractor._OP_MAP.get(op.__class__.__name__, '?')
                right = CodeExtractor._extract_expr(comparator)
                parts.append(op_name)
                parts.append(right)
            return ' '.join(parts)
        elif isinstance(node, ast.IfExp):
            test = CodeExtractor._extract_expr(node.test)
            body = CodeExtractor._extract_expr(node.body)
            orelse = CodeExtractor._extract_expr(node.orelse)
            return f'{body} if {test} else {orelse}'
        elif isinstance(node, ast.List):
            elts = [CodeExtractor._extract_expr(e) for e in node.elts]
            return f'[{", ".join(elts)}]'
        elif isinstance(node, ast.Dict):
            keys = [CodeExtractor._extract_expr(k) for k in node.keys] if node.keys else []
            values = [CodeExtractor._extract_expr(v) for v in node.values]
            pairs = [f'{k}:{v}' for k, v in zip(keys, values)]
            return '{' + ', '.join(pairs) + '}'
        elif isinstance(node, ast.Tuple):
            elts = [CodeExtractor._extract_expr(e) for e in node.elts]
            return ' '.join(elts)
        elif isinstance(node, ast.Subscript):
            value = CodeExtractor._extract_expr(node.value)
            slice_ = CodeExtractor._extract_expr(node.slice)
            return f'{value}[{slice_}]'
        elif isinstance(node, ast.Slice):
            lower = CodeExtractor._extract_expr(node.lower) if node.lower else ''
            upper = CodeExtractor._extract_expr(node.upper) if node.upper else ''
            step = CodeExtractor._extract_expr(node.step) if node.step else ''
            parts = [lower, upper, step]
            return ':'.join(parts)
        elif isinstance(node, ast.Lambda):
            args = [a.arg for a in node.args.args]
            body = CodeExtractor._extract_expr(node.body)
            return f'lambda {" ".join(args)} {body}'
        elif isinstance(node, ast.If):
            test = CodeExtractor._extract_expr(node.test)
            return f'if {test}'
        return ''


# ═══════════════════════════════════════════════════════════════════
# CodeSanitizer — Tradução de Símbolos Python para Tokens HDC
# ═══════════════════════════════════════════════════════════════════
# O encode_text() do GroundingEngine normaliza removendo pontuação.
# Para código Python, símbolos como :, (), %, == são a semântica.
# O CodeSanitizer traduz esses símbolos para palavras HDC válidas
# antes da codificação, e reverte na decodificação.
# ═══════════════════════════════════════════════════════════════════

_SYMBOL_MAP = {
    '==': ' _EQ_ ',
    '!=': ' _NEQ_ ',
    '>=': ' _GE_ ',
    '<=': ' _LE_ ',
    '->': ' _ARROW_ ',
    '%': ' _MOD_ ',
    ':': ' _COLON_ ',
    '(': ' _LPAREN_ ',
    ')': ' _RPAREN_ ',
    '[': ' _LBRACK_ ',
    ']': ' _RBRACK_ ',
    '{': ' _LBRACE_ ',
    '}': ' _RBRACE_ ',
    '+': ' _PLUS_ ',
    '-': ' _MINUS_ ',
    '*': ' _MULT_ ',
    '/': ' _DIV_ ',
    '=': ' _ASSIGN_ ',
    ',': ' _COMMA_ ',
    '.': ' _DOT_ ',
    '#': ' _COMMENT_ ',
    '@': ' _DECOR_ ',
    '&': ' _AND_ ',
    '|': ' _OR_ ',
    '^': ' _XOR_ ',
    '~': ' _INVERT_ ',
    '<': ' _LT_ ',
    '>': ' _GT_ ',
}

# Mapeamento reverso para dessanitizar
_REVERSE_MAP = {v.strip(): k for k, v in _SYMBOL_MAP.items()}


def sanitize_code(code_str: str) -> str:
    """Traduz simbolos Python para tokens HDC (palavras validas).

    Antes: def is_even(n): return n % 2 == 0
    Depois: def is_even _LPAREN_ n _RPAREN_ _COLON_ return n _MOD_ 2 _EQ_ 0

    O encode_text() pode entao processar sem perder semantica.
    """
    result = code_str
    # Ordena por tamanho decrescente para evitar substituicoes parciais
    for symbol, token in sorted(_SYMBOL_MAP.items(), key=lambda x: -len(x[0])):
        result = result.replace(symbol, token)
    return result


def desanitize_code(code_str: str) -> str:
    """Reverte a sanitizacao: tokens HDC de volta para simbolos Python.

    Antes: def is_even _LPAREN_ n _RPAREN_ _COLON_ return n _MOD_ 2 _EQ_ 0
    Depois: def is_even(n): return n % 2 == 0
    """
    result = code_str
    for token, symbol in _REVERSE_MAP.items():
        result = result.replace(token, symbol)
    # Limpa espacos duplicados
    import re
    result = re.sub(r' +', ' ', result)
    return result.strip()
