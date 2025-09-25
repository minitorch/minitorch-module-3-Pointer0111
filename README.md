# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py


### 并行优化诊断（Numba Parallel Accelerator）

- **如何运行**
  - 在本仓库根目录执行：
    ```bash
    python project/parallel_check.py
    ```
  - 输出将展示 `MAP / ZIP / REDUCE / MATRIX MULTIPLY` 四个函数在并行优化器下的分析与变换结果。

- **共性解读要点**
  - **Parallel loop listing / loop #ID**: 标注被识别为并行的最外层循环（使用 `prange`）。例如 `loop #3`、`#8`、`#10`、`#11` 等。
  - **Fusing loops**: 并行优化器会尝试融合相近的并行循环。我们这里各自只有一个主并行循环，通常显示 “0 loop(s) fused)”。
  - **After Optimisation**: 子循环多会被标记为 `serial`，这表示优化器保留最外层的大并行循环，内部小循环转为串行以减少线程开销。
  - **Allocation hoisting**: 循环体内的 `np.zeros(..., dtype=np.int32)` 等索引缓冲被“提升”到循环外（hoist），重复利用，以减少每次迭代的分配开销。

- **MAP（tensor_map）**
  - 诊断显示外层 `prange(n)` 成功并行（如 `+--3 is a parallel loop`）。
  - `out_index`、`in_index` 的分配被提升（Allocation hoisting），验证我们“索引缓冲”策略被编译器优化。
  - `aligned` 分支只进行一次位置计算并直接读写同一线性地址，避免广播与多次索引计算。

- **ZIP（tensor_zip）**
  - 外层 `prange(n)` 被识别为并行主循环（`loop #8`）。
  - 与 MAP 类似，多个索引缓冲分配被提升，减少分配成本。
  - 对齐路径下只计算一次线性位置，直接从 `a_storage[pos]` 与 `b_storage[pos]` 读取，写回 `out[pos]`。

- **REDUCE（tensor_reduce）**
  - 外层对每个输出位置的循环 `prange(n)` 被并行化（`loop #10`）。
  - 内层归约循环串行（期望行为），且我们通过预计算 `a_pos` 与 `step`，用线性步进 `a_pos + j * step`，避免在内层做昂贵的索引函数调用。
  - `out_index` 的分配被提升，减少循环内分配。

- **MATRIX MULTIPLY（_tensor_matrix_multiply）**
  - 外层对所有 `(n, i, j)` 的总计索引使用 `prange(total)`，被识别为并行最外层循环（`loop #11`）。
  - 内层 `k` 循环只进行一次乘法与局部累加 `acc += a * b`，没有全局写；结束后统一写回 `out[out_pos]`。
  - 整个实现基于 strides 的线性地址计算，未调用 `to_index` / `index_to_position` 等函数，满足“无索引缓冲/函数调用”的优化要求。