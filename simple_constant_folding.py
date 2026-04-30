"""Реализация алгоритма CP"""

from enum import Enum
from typing import Any, Callable, Optional
from dataclasses import dataclass, field
from copy import deepcopy
import operator
import unittest


class LatticeValueKind(Enum):
    """
    Lattice value states
    """
    BOTTOM = 0
    TOP = 1
    CONST = 2


@dataclass
class LatticeValue:
    """
    Lattice value which can contain a const value or be either TOP or BOTTOM
    """
    kind: LatticeValueKind
    const: Optional[Any] = None

    def join(self, other: LatticeValue) -> LatticeValue:
        """
        Lattice join (https://en.wikipedia.org/wiki/Join_and_meet)
        """
        if self.kind == LatticeValueKind.BOTTOM:
            return other
        if other.kind == LatticeValueKind.BOTTOM:
            return deepcopy(self)

        if LatticeValueKind.TOP in (self.kind, other.kind):
            return LatticeValue(LatticeValueKind.TOP)

        if self.const == other.const:
            return deepcopy(self)

        return LatticeValue(LatticeValueKind.TOP)

    def _op(self, other: LatticeValue | float | int, op: Callable) -> LatticeValue:
        # pylint: disable=line-too-long
        """
        Executes 'self op other'
        The rules are following:
        1. if self or other is TOP then result of op is TOP
        2. if self or other is BOTTOM then result of op it BOTTOM
        3. if other is numeric then result is new LatticeValue(const=self.const + other), self must be CONST
        4. if other is CONST then result if new LatticeValue(const=self.const + other.const), self must be CONST
        """
        if self.kind in (LatticeValueKind.BOTTOM, LatticeValueKind.TOP):
            return LatticeValue(kind=self.kind)

        if isinstance(other, (float, int)):
            assert self.const is not None and self.kind == LatticeValueKind.CONST
            return LatticeValue(LatticeValueKind.CONST, op(self.const, other))

        if other.kind == LatticeValueKind.BOTTOM:
            return LatticeValue(LatticeValueKind.BOTTOM)

        if other.kind == LatticeValueKind.TOP:
            return LatticeValue(LatticeValueKind.TOP)
        assert isinstance(other, LatticeValue)

        return LatticeValue(LatticeValueKind.CONST, op(self.const, other.const))

    def __add__(self, other: LatticeValue | float | int) -> LatticeValue:
        return self._op(other, operator.add)

    def __mul__(self, other: LatticeValue) -> LatticeValue:
        return self._op(other, operator.mul)

    def __sub__(self, other: LatticeValue) -> LatticeValue:
        return self._op(other, operator.sub)

    def __truediv__(self, other: LatticeValue) -> LatticeValue:
        return self._op(other, operator.truediv)


class LatticeVector:
    """
    Vector of LatticeValue's
    """

    def __init__(self, n: int = 0, vec: Optional[list[LatticeValue]] = None,
                 copy: Optional[LatticeVector] = None) -> None:
        if copy is not None:
            self.vec = deepcopy(copy.vec)
            self.n = copy.n

            return

        self.n = n
        self.vec = [] if not vec else vec

        if n == 0 and len(self.vec) == 0:
            return

        if len(self.vec) == 0:
            self.vec = [LatticeValue(LatticeValueKind.BOTTOM)
                        for _ in range(n)]

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

    def append(self, value: LatticeValue):
        """
        Appends new value to vector
        """
        self.vec.append(value)
        self.n += 1


def join(m_in: LatticeVector, pred: LatticeVector) -> LatticeVector:
    """
    Joins two vectors
    """
    return LatticeVector(m_in.n, [a.join(b) for a, b in zip(m_in, pred)])


@dataclass(eq=False)
class BasicBlock:
    """
    Abstracts execution basic block
    """
    name: str
    vars: dict[str, "LatticeValue"] = field(default_factory=dict)
    preds: list["BasicBlock"] = field(default_factory=list)
    transfer_function: Optional[Callable[[
        "LatticeVector"], "LatticeVector"]] = None

    def __str__(self) -> str:
        return f"BasicBlock: {self.vars}, preds={[bb.name for bb in self.preds]}"


class Scope:
    """
    Abstraction over scope which contains basic blocks
    """

    def __init__(self) -> None:
        self.var_set: set[str] = set()
        self.all_vars: list[str] = []
        self.all_blocks: list[BasicBlock] = []

    def define_var(self, var_name: str):
        """
        Declares new var if it is not exist by adding it to list
        """
        if var_name in self.var_set:
            return

        self.var_set.add(var_name)
        self.all_vars.append(var_name)

    def __str__(self) -> str:
        s = str()
        for b in self.all_blocks:
            values = [b.vars[x] for x in self.all_vars]
            s += f"{b.name}: "
            for lattice in values:
                if lattice.kind == LatticeValueKind.BOTTOM:
                    s += "u"
                elif lattice.kind == LatticeValueKind.TOP:
                    s += "-"
                else:
                    s += str(lattice.const)
            s += "\n"

        return s[:-1]

    def replace_in_block(self, block: BasicBlock, variable: str, value: LatticeValue):
        """
        Replace value of variable. Raises exception if variable is not found.
        """
        if not variable in block.vars:
            raise RuntimeError(f"Variable {variable} is not found")

        block.vars[variable] = value


def propagate_const(scope: Scope, max_iter: int = 10000):
    """
    Propagates all const variables in scope
    """
    m: dict[BasicBlock, LatticeVector] = {}
    n = len(scope.all_vars)

    for block in scope.all_blocks:
        m[block] = LatticeVector(n)

    changed = True
    iters = 0
    while changed:
        changed = False

        for block in scope.all_blocks:
            m_in: LatticeVector = LatticeVector(n)
            for pred in block.preds:
                m_in = join(m_in, m[pred])

            assert block.transfer_function is not None

            m_out: LatticeVector = block.transfer_function(m_in)

            if m[block] != m_out:
                m[block] = m_out
                changed = True

        iters += 1
        if iters > max_iter:
            raise AssertionError("Max iteration exceeded")

    for block in scope.all_blocks:
        for v, propagated_value in zip(scope.all_vars, m[block]):
            if propagated_value not in (LatticeValueKind.BOTTOM, LatticeValueKind.TOP):
                scope.replace_in_block(block, v, propagated_value)


class ConstPropagationTest(unittest.TestCase):
    """
    Tests constant propagation algorithm correctness
    """

    def setUp(self) -> None:
        self.scope = self.init_scope()

    def test_const_propagation(self):
        """
        Tests book's hir listing
        """
        before = str(self.scope)
        propagate_const(self.scope)
        after = str(self.scope)

        print(before, '\n')
        print(after)

        self.assertTrue(before == "BB1: 011u\nBB2: 101u\nBB3: 1011")
        self.assertTrue(after == "BB1: 011u\nBB2: --1u\nBB3: --1-")

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

    def init_scope(self):
        """Inits scope with B1, B2 and B3 basic blocks"""
        scope = Scope()
        bb1 = self.init_bb1()
        bb2 = self.init_bb2(bb1)
        bb3 = self.init_bb3(bb2)
        scope.all_blocks.extend([bb1, bb2, bb3])
        for b in scope.all_blocks:
            assert b.vars is not None
            assert isinstance(b.vars, dict)
            for var, _ in b.vars.items():
                scope.define_var(var)

        return scope

    def init_bb1(self) -> BasicBlock:
        """Inits B1"""
        bb = BasicBlock(name="BB1")

        bb.vars = {
            "i": LatticeValue(LatticeValueKind.CONST, 0),
            "x": LatticeValue(LatticeValueKind.CONST, 1),
            "y": LatticeValue(LatticeValueKind.CONST, 1),
            "p": LatticeValue(LatticeValueKind.BOTTOM)
        }

        def transfer_function(lattice: LatticeVector) -> LatticeVector:
            trans_lattice = deepcopy(lattice)
            trans_lattice[0] = bb.vars["i"]
            trans_lattice[1] = bb.vars["x"]
            trans_lattice[2] = bb.vars["y"]
            trans_lattice[3] = bb.vars["p"]
            return trans_lattice

        bb.transfer_function = transfer_function

        return bb

    def init_bb2(self, bb1: BasicBlock) -> BasicBlock:
        """Inits B2, using B1 as its parent"""
        bb = BasicBlock(name="BB2")

        def transfer_i(i: LatticeValue):
            return i + 1

        def transfer_x(x: LatticeValue, i: LatticeValue):
            return x * i

        def transfer_y(y: LatticeValue):
            return y * y

        def transfer_p(p: LatticeValue):
            return p

        bb.vars["i"] = transfer_i(bb1.vars["i"])
        bb.vars["x"] = transfer_x(bb1.vars["x"], bb1.vars["i"])
        bb.vars["y"] = transfer_y(bb1.vars["y"])
        bb.vars["p"] = transfer_p(bb1.vars["p"])

        def transfer_function(lattice: LatticeVector) -> LatticeVector:
            transfered = LatticeVector(copy=lattice)
            transfered[0] = transfer_i(lattice[0])
            transfered[1] = transfer_x(lattice[1], lattice[0])
            transfered[2] = transfer_y(lattice[2])
            transfered[3] = transfer_p(lattice[3])

            return transfered

        bb.transfer_function = transfer_function

        bb.preds.append(bb1)
        bb.preds.append(bb)  # loop

        return bb

    def init_bb3(self, bb2: BasicBlock) -> BasicBlock:
        """Inits B3 using B2 as its parent"""
        bb = BasicBlock(name="BB3")

        def transfer_i(i: LatticeValue):
            return i

        def transfer_x(x: LatticeValue):
            return x

        def transfer_y(y: LatticeValue):
            return y

        def transfer_p(x: LatticeValue, y: LatticeValue):
            return x + y

        bb.vars["i"] = transfer_i(bb2.vars["i"])
        bb.vars["x"] = transfer_x(bb2.vars["x"])
        bb.vars["y"] = transfer_y(bb2.vars["y"])
        bb.vars["p"] = transfer_p(bb2.vars["x"], bb2.vars["y"])

        def transfer_function(lattice: LatticeVector) -> LatticeVector:
            transfered = LatticeVector(copy=lattice)
            transfered[3] = lattice[1] + lattice[2]

            return transfered

        bb.transfer_function = transfer_function
        bb.preds.append(bb2)

        return bb


if __name__ == "__main__":
    unittest.main()
