import functools


def patch_benchmark(benchmark, monkeypatch, module, func_name):
    fn = getattr(module, func_name)

    @functools.wraps(fn)
    def wrapped(*a, **kw):
        return benchmark(fn, *a, **kw)

    monkeypatch.setattr(module, func_name, wrapped)
