import argparse
import time
from collections import defaultdict

import numpy as np

from sparseappbench.benchmarks.matmul import (
    benchmark_matmul,
    dg_matmul_dense_large,
    dg_matmul_dense_small,
    dg_matmul_sparse_large,
    dg_matmul_sparse_small,
)
from sparseappbench.binsparse_format import BinsparseFormat
from sparseappbench.frameworks.abstract_framework import AbstractFramework
from sparseappbench.frameworks.checker_framework import CheckerFramework
from sparseappbench.frameworks.einsum import einsum
from sparseappbench.frameworks.numpy_framework import NumpyFramework

import finchlite as fl
from finchlite.autoschedule import DefaultLogicFormatter, LogicExecutor, LogicNormalizer
from finchlite.autoschedule.compiler import LogicCompiler
from finchlite.autoschedule.galley_optimize import GalleyLogicalOptimizer
from finchlite.autoschedule.standardize import LogicStandardizer
from finchlite.autoschedule.tensor_stats import (
    BlockedStatsFactory,
    DatabaseStatsFactory,
    DCStatsFactory,
    DenseStatsFactory,
    UniformStatsFactory,
)
from finchlite.finch_notation.interpreter import NotationInterpreter


class AutoschedulerFramework(AbstractFramework):
    def __init__(self, autoscheduler):
        self.autoscheduler = autoscheduler

    def from_benchmark(self, array: BinsparseFormat):
        return NumpyFramework().from_benchmark(array)

    def to_benchmark(self, array) -> BinsparseFormat:
        return BinsparseFormat.from_numpy(np.asarray(array))

    def lazy(self, array):
        return fl.lazy(array)

    def compute(self, array):
        return fl.compute(array, ctx=self.autoscheduler)

    def einsum(self, prgm, **kwargs):
        return einsum(self, prgm, **kwargs)

    def with_fill_value(self, array, value):
        return array

    def __getattr__(self, name):
        return getattr(fl, name)


def galley_autoscheduler(stats_factory):
    return LogicNormalizer(
        LogicExecutor(
            GalleyLogicalOptimizer(
                LogicStandardizer(
                    DefaultLogicFormatter(LogicCompiler(NotationInterpreter()))
                )
            ),
            stats_factory=stats_factory,
        )
    )


def blocked_factory(inner_factory=None):
    if inner_factory is None:
        inner_factory = DatabaseStatsFactory()
    factory = BlockedStatsFactory({}, inner_factory)
    factory.blocks_per_dim = defaultdict(lambda: 8)
    return factory


STATS_DICT = {
    "DenseStats": galley_autoscheduler(DenseStatsFactory()),
    "UniformStats": galley_autoscheduler(UniformStatsFactory()),
    "DatabaseStats": galley_autoscheduler(DatabaseStatsFactory()),
    "DCStats": galley_autoscheduler(DCStatsFactory()),
    "BlockedDenseStats": galley_autoscheduler(blocked_factory(DenseStatsFactory())),
    "BlockedUniformStats": galley_autoscheduler(blocked_factory(UniformStatsFactory())),
    "BlockedDatabaseStats": galley_autoscheduler(
        blocked_factory(DatabaseStatsFactory())
    ),
    "BlockedDCStats": galley_autoscheduler(blocked_factory(DCStatsFactory())),
}

FRAMEWORK_DICT = {
    "numpy": NumpyFramework(),
    "checker": CheckerFramework(),
    "DenseStats": AutoschedulerFramework(STATS_DICT["DenseStats"]),
    "UniformStats": AutoschedulerFramework(STATS_DICT["UniformStats"]),
    "DatabaseStats": AutoschedulerFramework(STATS_DICT["DatabaseStats"]),
    "DCStats": AutoschedulerFramework(STATS_DICT["DCStats"]),
    "BlockedDenseStats": AutoschedulerFramework(STATS_DICT["BlockedDenseStats"]),
    "BlockedUniformStats": AutoschedulerFramework(STATS_DICT["BlockedUniformStats"]),
    "BlockedDatabaseStats": AutoschedulerFramework(STATS_DICT["BlockedDatabaseStats"]),
    "BlockedDCStats": AutoschedulerFramework(STATS_DICT["BlockedDCStats"]),
}
BENCHMARK_DICT = {
    "matmul": benchmark_matmul,
}
DATA_GENERATOR_DICT = {
    "matmul": {
        "matmul_dense_small": dg_matmul_dense_small,
        "matmul_dense_large": dg_matmul_dense_large,
        "matmul_sparse_small": dg_matmul_sparse_small,
        "matmul_sparse_large": dg_matmul_sparse_large,
    }
}


def run_benchmark(framework, benchmark_function, benchmark_data_generator, iters):
    execution_times = []
    for _ in range(iters):
        data = benchmark_data_generator()
        start = time.perf_counter()
        benchmark_function(framework, *data)
        end = time.perf_counter()
        duration = end - start
        execution_times.append(duration)
    print(
        f"Benchmark took an average of {sum(execution_times) / len(execution_times)}\
             seconds"
    )
    return execution_times


def save_benchmark_results(
    results_folder, execution_times, framework, benchmark, data_generator
):
    filename = f"{results_folder}/{framework}_{benchmark}_{data_generator}.csv"
    with open(filename, "w") as f:
        f.write("Framework,Benchmark,Data Generator,Iteration,ExecutionTime\n")
        for i, execution_time in enumerate(execution_times, start=1):
            f.write(f"{framework},{benchmark},{data_generator},{i},{execution_time}\n")


def main(
    frameworks=None,
    framework_names=None,
    benchmarks=None,
    benchmark_names=None,
    data_generators=None,
    data_generator_names=None,
    iters=None,
    results_folder=None,
    args=None,
):
    collected_frameworks = FRAMEWORK_DICT.copy()
    if frameworks is not None:
        for framework_name, framework in frameworks.items():
            collected_frameworks[framework_name] = framework
    frameworks = collected_frameworks

    collected_benchmarks = BENCHMARK_DICT.copy()
    if benchmarks is not None:
        for benchmark_name, benchmark in benchmarks.items():
            collected_benchmarks[benchmark_name] = benchmark
    benchmarks = collected_benchmarks

    collected_data_generators = {
        benchmark_name: generators.copy()
        for benchmark_name, generators in DATA_GENERATOR_DICT.copy().items()
    }
    if data_generators is not None:
        for benchmark_name, generators in data_generators.items():
            for generator_name, generator in generators.items():
                collected_data_generators[benchmark_name][generator_name] = generator
    data_generators = collected_data_generators

    parser = argparse.ArgumentParser(description="Run sparse autoscheduling benchmark")
    parser.add_argument(
        "--framework",
        default=["all"],
        nargs="*",
        help="Execution framework(s) to use",
    )
    parser.add_argument(
        "--benchmark",
        default=["all"],
        nargs="*",
        help="Benchmark(s) to run",
    )
    parser.add_argument(
        "--data-generator",
        default=["all"],
        nargs="*",
        help="Data generator(s) to use",
    )
    parser.add_argument(
        "--iterations",
        default=5,
        type=int,
        help="Number of iterations to run for each benchmark",
    )
    parser.add_argument(
        "--results-folder", default="results", help="Folder to save results"
    )
    args = parser.parse_args(args)

    if framework_names is None:
        if args.framework == ["all"]:
            framework_names = list(FRAMEWORK_DICT.keys())
        else:
            framework_names = args.framework

    if benchmark_names is None:
        if args.benchmark == ["all"]:
            benchmark_names = list(BENCHMARK_DICT.keys())
        else:
            benchmark_names = args.benchmark

    if data_generator_names is None:
        if args.data_generator == ["all"]:
            data_generator_names = [
                generator_name
                for generators in collected_data_generators.values()
                for generator_name in generators
            ]
        else:
            data_generator_names = args.data_generator

    if results_folder is None:
        results_folder = args.results_folder

    if iters is None:
        iters = args.iterations

    data_generator_names = set(data_generator_names)
    for framework_name in framework_names:
        framework = frameworks[framework_name]
        for benchmark_name in benchmark_names:
            benchmark = benchmarks[benchmark_name]
            for data_generator_name, data_generator in data_generators[
                benchmark_name
            ].items():
                if data_generator_name not in data_generator_names:
                    continue
                print(
                    f"Running benchmark {benchmark_name} with framework\
                          {framework_name} and data generator {data_generator_name}"
                )
                execution_times = run_benchmark(
                    framework, benchmark, data_generator, iters
                )
                save_benchmark_results(
                    results_folder,
                    execution_times,
                    framework_name,
                    benchmark_name,
                    data_generator_name,
                )


if __name__ == "__main__":
    main()
