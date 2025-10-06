import numpy as np
import .nodes as ein

class EinsumInterpreter():
    def __init__(self, xp=None, bindings=None):
        if bindings is None:
            bindings = {}
        if xp is None:
            xp = np
    
    def __call__(self, prgm):
        xp = self.xp
        match prgm:
            case ein.Plan(bodies):
                for body in bodies:
                    prgm = body
            case ein.Produces(args):
                return tuple(self(arg) for arg in args)
            case ein.Einsum(op, tns, idxs, arg):
                # This is the main entry point for einsum execution
                loops = arg.get_loops()
                assert set(idxs).issubset(loops)
                loops = sorted(loops)
                arg = self.eval(arg, loops)
                axis = tuple(i for i in range(len(loops)) if loops[i] not in idxs)
                if op is not None:
                    op = getattr(xp, reduction_ops.get(op, None))
                    val = op(arg, axis=axis)
                else:
                    assert set(idxs) == set(loops)
                    val = arg
                dropped = [idx for idx in loops if idx in idxs]
                axis = [dropped.index(idx) for idx in idxs]
                self.bindings[tns] = xp.transpose(val, axis)


    def eval(self, ex, loops):
        xp = self.xp
        match ex:
            case ein.Literal(val):
                return val
            case ein.Alias(name):
                return self.bindings[name]
            case ein.Call(func, args):
                func = self(func)
                if len(args) == 1:
                    func = getattr(xp, unary_ops[func])
                else:
                    func = getattr(xp, nary_ops[func])
                vals = [self.eval(arg, loops) for arg in args]
                return func(*vals)
            case ein.Access(tns, idxs):
                assert len(idxs) == len(set(idxs))
                perm = [idxs.index(idx) for idx in loops if idx in idxs]
                tns = self.bindings[tns]
                tns = xp.transpose(tns, perm)
                return xp.expand_dims(
                    tns, [i for i in range(len(loops)) if loops[i] not in self.idxs]
                )