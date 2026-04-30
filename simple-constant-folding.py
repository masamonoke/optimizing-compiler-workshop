from enum import Enum
from typing import Any, Callable, Optional
from dataclasses import dataclass
from copy import deepcopy

class LatticeValueKind(Enum):
    BOTTOM = 0
    TOP = 1
    CONST = 2

@dataclass
class LattiveValue:
    kind: LatticeValueKind
    const: Optional[Any] = None
    name: str = ""

    def join(self, other: LattiveValue) -> LattiveValue:
        if self.kind == LatticeValueKind.BOTTOM:
            return other
        if other.kind == LatticeValueKind.BOTTOM:
            return deepcopy(self)

        if self.kind == LatticeValueKind.TOP or other.kind == LatticeValueKind.TOP:
            return LattiveValue(LatticeValueKind.TOP)

        if self.const == other.const:
            return deepcopy(self)

        return LattiveValue(LatticeValueKind.TOP)

class LatticeVector:
    # TODO: probably instead if passing n pass a vector of default values?
    def __init__(self, n: int = 0, vec: list[LattiveValue] = []) -> None:
        self.n = n
        self.vec = vec

        if n == 0 and len(vec) == 0:
            return

        if len(vec) == 0:
            self.vec = [LattiveValue(LatticeValueKind.BOTTOM) for _ in range(n)]

    def __iter__(self):
        if isinstance(self.vec, list):
            return iter(self.vec)

        raise RuntimeError("Not a list")

    def __len__(self):
        return len(self.vec)

    def __getitem__(self, i):
        return self.vec[i]

    def __setitem__(self, i, v):
        assert i < len(self.vec)
        self.vec[i] = v

    def __str__(self) -> str:
        s = ""
        for v in self.vec:
            assert v is not None
            if v.kind == LatticeValueKind.CONST:
                s += str(v.const)
            elif v.kind == LatticeValueKind.BOTTOM:
                s += "u"
            else:
                s += "-"

        return s

    def __eq__(self, other):
        if not isinstance(other, LatticeVector):
            return False
        return self.vec == other.vec

    def append(self, value: LattiveValue):
        self.vec.append(value)
        self.n += 1

def join(m_in: LatticeVector, pred: LatticeVector) -> LatticeVector:
    return LatticeVector(m_in.n, [a.join(b) for a, b in zip(m_in, pred)])


class BasicBlock:
    def __init__(self, name: str) -> None:
        self.vars: dict[str, Any] = dict()
        # TODO:
        # what is preds and what is succ?
        # as i can see preds is all the blocks from which we can arrive to this block
        self.preds: list[BasicBlock] = []
        self.transfer_function: Optional[Callable[[LatticeVector], LatticeVector]]= None
        self.name: str = name

    def __str__(self) -> str:
        return f"BasicBlock: {self.vars}, preds={[bb.name for bb in self.preds]}"

class Scope:
    def __init__(self) -> None:
        self.all_vars: set[str] = set()
        self.all_blocks: list[BasicBlock] = []

    def replace_in_block(self, block: BasicBlock, variable: str, value: Any):
        if not variable in block.vars:
            raise RuntimeError(f"Variable {variable} is not found")

        block.vars[variable] = value

# TODO: debug function
def log_block_state(block: BasicBlock):
    i = block.vars["i"]
    x = block.vars["x"]
    y = block.vars["y"]
    p = block.vars["p"]
    i = "u" if i is None else i
    x = "u" if x is None else x
    y = "u" if y is None else y
    p = "u" if p is None else p
    print(f"{block.name}: {i}{x}{y}{p}")

def propagate_const(f: Scope, max_iter: int = 10000):
    m: dict[BasicBlock, LatticeVector] = dict()
    n = len(f.all_vars)

    for block in f.all_blocks:
        m[block] = LatticeVector(n)

    changed = True
    iter = 0
    while changed:
        changed = False
        for block in f.all_blocks:
            log_block_state(block)

        for block in f.all_blocks:
            m_in: LatticeVector = LatticeVector(n)
            for pred in block.preds:
                m_in = join(m_in, m[pred])

            assert block.transfer_function is not None

            m_out: LatticeVector = block.transfer_function(m_in)
            print(f"{m_in} -> {m_out}")
            if m[block] != m_out:
                m[block] = m_out
                changed = True

        iter += 1
        if iter > max_iter:
            raise AssertionError("Max iteration exceeded")

    for block in f.all_blocks:
        for v, propagated_value in zip(f.all_vars, m[block]):
            if propagated_value != LatticeValueKind.BOTTOM and propagated_value != LatticeValueKind.TOP:
                f.replace_in_block(block, v, propagated_value.const)


# TODO: set the testcase as test class

# hir
# BB1:
# i = 0
# x = 1
# y = 1
#
# BB2:
# x = x + 1
# i = i + 1
# y = y * y
# if y < 10 goto BB2; else goto BB3
#
# BB3:
# p = x + y

# передаточные функции для hir
# FB1([i, x, y, p]) = [0, 1, 1, p]
# FB2([i, x, y, p]) = [i + 1, x * i, y * y, p]
# FB3([i, x, y, p]) = [i, x, y, x + y]

# дерево переходов
# BB1
# └── BB2
#     ├── BB3
#     └── BB2

# TODO: get all vars from the scope?
def init_bb1() -> BasicBlock:
    # TODO: what with lifetimes?
    bb = BasicBlock("BB1")

    bb.vars = {
        "i": 0,
        "x": 1,
        "y": 1,
        "p": None
    }

    def transfer_function(lattice: LatticeVector) -> LatticeVector:
        trans_lattice = deepcopy(lattice)
        trans_lattice[0] = LattiveValue(LatticeValueKind.CONST, bb.vars["i"], "i")
        trans_lattice[1] = LattiveValue(LatticeValueKind.CONST, bb.vars["x"], "x")
        trans_lattice[2] = LattiveValue(LatticeValueKind.CONST, bb.vars["y"], "y")
        trans_lattice[3] = LattiveValue(LatticeValueKind.BOTTOM, name="p")
        return trans_lattice

    bb.transfer_function = transfer_function

    return bb

def init_bb2(bb1: BasicBlock) -> BasicBlock:
    bb = BasicBlock("BB2")

    def transfer_i(i):
        return i + 1
    def transfer_x(x, i):
        return x * i
    def transfer_y(y):
        return y * y
    def transfer_p(p):
        return p

    bb.vars["i"] = transfer_i(bb1.vars["i"])
    bb.vars["x"] = transfer_x(bb1.vars["x"], bb1.vars["i"])
    bb.vars["y"] = transfer_y(bb1.vars["y"])
    bb.vars["p"] = transfer_p(bb1.vars["p"])

    def transfer_function(lattice: LatticeVector) -> LatticeVector:
        transfered = deepcopy(lattice)

        if lattice[0].kind == LatticeValueKind.CONST:
            transfered[0].const = transfer_i(lattice[0].const)
        if lattice[1].kind == LatticeValueKind.CONST and lattice[0].kind == LatticeValueKind.CONST:
            transfered[1].const = transfer_x(lattice[1].const, lattice[0].const)
        if lattice[2].kind == LatticeValueKind.CONST:
            transfered[2].const = transfer_y(lattice[2].const)
        if lattice[3].kind == LatticeValueKind.CONST:
            transfered[3].const = transfer_p(lattice[3].const)

        return transfered

    bb.transfer_function = transfer_function

    bb.preds.append(bb1)
    bb.preds.append(bb) # loop

    return bb

def init_bb3(bb2: BasicBlock) -> BasicBlock:
    bb = BasicBlock("BB3")

    def transfer_i(i):
        return i
    def transfer_x(x):
        return x
    def transfer_y(y):
        return y
    def transfer_p(x, y):
        return x + y

    bb.vars["i"] = transfer_i(bb2.vars["i"])
    bb.vars["x"] = transfer_x(bb2.vars["x"])
    bb.vars["y"] = transfer_y(bb2.vars["y"])
    bb.vars["p"] = transfer_p(bb2.vars["x"], bb2.vars["y"])

    def transfer_function(lattice: LatticeVector) -> LatticeVector:
        transfered = deepcopy(lattice)
        if lattice[1].kind == LatticeValueKind.CONST and lattice[2].kind == LatticeValueKind.CONST:
            transfered[3] = LattiveValue(LatticeValueKind.CONST, lattice[1].const + lattice[2].const) # undefined is this block

        return transfered

    bb.transfer_function = transfer_function
    bb.preds.append(bb2)

    return bb

def init_scope() -> Scope:
    scope = Scope()
    bb1 = init_bb1()
    bb2 = init_bb2(bb1)
    bb3 = init_bb3(bb2)
    scope.all_blocks.extend([bb1, bb2, bb3])
    for b in scope.all_blocks:
        assert b.vars is not None
        assert isinstance(b.vars, dict)
        for var, _ in b.vars.items():
            scope.all_vars.add(var)

    return scope


if __name__ == "__main__":
    scope = init_scope()
    # TODO: test init
    propagate_const(scope)
    for b in scope.all_blocks:
        print(b)

    # TODO: what to assert
    # TODO: probably I should replace expressions with according const variable or
    # i change transfer_function to instead of evaluating expression just make constant value
