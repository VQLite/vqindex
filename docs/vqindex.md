# VQIndex 2 说明

VQIndex 是在 ScaNN 上封装的向量检索库,提供 C 接口(`libvqindex_api.so`)和 Python 扩展(`vqindex_py`)。
本版本(VQIndex 2)把底层从 ScaNN 1.2.9 换成 ScaNN 1.4.2,不再依赖 TensorFlow,接口与旧版相同。
本文说明工具链、编译、测试、索引文件格式,以及从旧版升级的步骤。

下面的步骤在 Ubuntu 24.04 x86-64 上验证过。旧版支持的 macOS 编译在本版本没有重新验证。

| 章节 | 内容 |
|---|---|
| 1 工具链 | 需要装什么 |
| 2 编译 | 命令、环境变量、指令集、JCC 缺陷、NUMA 部署建议 |
| 3 测试 | 测试脚本和结果解读 |
| 4 索引文件格式 | AH 码的新旧两种存法、dump 格式、转换脚本 |
| 5 与旧版的兼容性和升级步骤 | 换库、转换索引、回退 |
| 6 常见问题 | |

## 1. 工具链

| 工具 | 版本 | 用途 | 获取方式 |
|---|---|---|---|
| Bazel | 7.6.1(需要 7.x) | 构建系统 | `tools/install_bazel.sh` |
| clang | 18 或 19 | C/C++ 编译器 | `apt install clang` |
| Python 3 + 开发头文件 | 实测 3.10、3.12 | Bazel 构建过程使用;`vqindex_py` 针对它编译 | `apt install python3 python3-dev` |
| numpy | 任意 | 只有测试脚本用 | `apt install python3-numpy` 或 `pip install numpy` |
| 外网访问 | - | 首次编译时 Bazel 下载第三方依赖 | - |

说明:

- **Bazel**:`tools/install_bazel.sh` 下载官方 7.6.1 二进制到 `tools/bazel/bazel` 并校验 sha256,`build.sh`
  会自动使用它。该目录不入库。脚本只固定了 Linux x86-64 的校验值,其他平台请自行安装 Bazel 7.x 并通过
  `BAZEL=/path/to/bazel` 指定。版本号同时记录在 `.bazelversion`,用 bazelisk 也可以。
- **clang**:ScaNN 的 SIMD 代码只能用 clang 编译,gcc 13 实测会报约 1000 个 `__m128`/`__m128i` 转换错误。
  上游以 clang 19 为目标;用 clang 18 时 `build.sh` 会自动加 `-frelaxed-template-template-args`。
- **Python**:直接用系统的 `python3` 即可,不需要虚拟环境。`vqindex_py.so` 是 CPython 扩展,只能被
  **编译时所用的那个小版本**(如 3.12)导入;要换版本就用 `PYTHON_BIN` 指定后重新编译。
- **第三方依赖**:abseil、protobuf、eigen、highway、cnpy 等由 Bazel 按 `MODULE.bazel` 自动下载
  (来源为 Bazel Central Registry 和 GitHub),不需要手动安装。

## 2. 编译

```bash
tools/install_bazel.sh        # 只需一次

./build.sh vqindex_py         # Python 扩展 -> test/vqindex_py.so
./build.sh vqindex_api        # C 接口     -> libs/libvqindex_api.so, libs/vqindex_api.h
```

可用的环境变量:

| 变量 | 默认值 | 含义 |
|---|---|---|
| `VQINDEX_COPTS` | `-mavx` | 基线指令集,见下 |
| `PYTHON_BIN` | PATH 中的 `python3` | 构建使用、且 `vqindex_py` 所针对的 Python |
| `CC` | `clang` | 编译器,如 `CC=clang-19` |
| `BAZEL` | `tools/bazel/bazel`,没有则用 PATH 中的 `bazel` | Bazel 可执行文件 |
| `VQINDEX_ALIGN_BRANCHES` | `1` | 让条件跳转避开 32 字节边界,见下面"JCC 缺陷"一节;设为 `0` 关闭 |

### 指令集

- 基线指令集在编译期固定,**运行库的机器必须支持它**,否则会 `Illegal instruction`。
  默认 `-mavx`,Sandy Bridge 及以后的 CPU 都能运行(上游 ScaNN 用的是 `-mavx -mfma`,要求 Haswell 及以后)。
- 想把基线提高到 AVX2/FMA 时(实测与 `-mavx` 没有稳定的速度差别,且产物不能在 Ivy Bridge 这类老 CPU 上运行):

  ```bash
  VQINDEX_COPTS="-mavx2 -mfma" ./build.sh vqindex_api
  ```

- 更宽的指令(AVX2、AVX-512)不靠编译选项,而是**运行时按 CPU 自动选择**。所以同一个库拷到支持 AVX2 的机器上
  会自动走 AVX2 路径。实测 `-mavx` 基线的库在 Xeon Gold 6230R 上比强制 SSE4 快约 20%。
- 不建议用 `-march=native`:产物会绑定到编译机的 CPU 型号,拷到别的机器可能无法运行。

### JCC 缺陷(Skylake 系列 CPU)

Intel Skylake 到 Cascade Lake 这几代 CPU 有一个硬件缺陷(JCC erratum),微码补丁之后,跨越或紧贴 32 字节边界的条件跳转
不能进入微指令缓存,落在热循环里会明显变慢。AH 分区扫描的内层循环是否中招取决于编译器生成的代码布局:
clang 18 下循环末尾的跳转正好顶在边界上,在 9200 万点的线上索引上实测每次查询慢约 9%。

`build.sh` 默认给编译器和链接器加上 `-mbranches-within-32B-boundaries`(链接期是因为开了 ThinLTO,代码在链接时生成),
让汇编器自动填充避开边界。这个选项只在使用 clang、x86-64 时生效,对不受影响的 CPU 没有可观察的代价。

### 部署建议:NUMA

双路机器上建议用 `numactl --interleave=all` 启动服务。索引内存默认会落在加载线程所在的节点上,
高并发时受单个节点的内存带宽限制;交错分布后实测高并发吞吐提高 20%–50%,代价是单线程延迟增加约 15%。

### 耗时与产物

- 首次编译需要下载依赖并全量编译;之后是增量编译。全量编译在双路 Xeon E5-2697 v2(48 线程)上约 7 分钟(不含下载)。
- 修改 `VQINDEX_COPTS`、`CC` 或 `PATH` 会触发全量重编。
- Bazel 的缓存在 `~/.cache/bazel`,工作区里的 `bazel-*` 是指向缓存的软链接,均不入库。

## 3. 测试

```bash
test/run_tests.sh             # 功能测试
test/run_tests.sh --bench     # 功能测试 + SIMD 档位对比
```

运行测试的 Python 必须与编译时相同(默认都是 `python3`),并装有 numpy。测试会写约 500 MB 的临时索引,
默认放在 `$TMPDIR` 下的新目录并在结束后删除;可用 `VQINDEX_TEST_DIR` 指定位置。

| 脚本 | 内容 |
|---|---|
| `test/test.py` | 最小示例:init / add / train / search / stats / release |
| `test/test_add.py` | 增量 add:各种批大小(含 32 点分块边界)、MEMORY 与 FILE 两种存储、dump 后重载的一致性、在重载的索引上继续 add、子进程训练。全部通过时最后一行输出 `FAILED: none` |
| `test/test_concurrency.py` | 加锁相关:多线程并发 add 后向量与 vid 是否对齐、train/add 进行中并发 search 和 stats、内积为负的结果是否带 vid、MEMORY 模式增量 dump、子进程训练的返回值 |
| `test/test_format.py` | dump 格式:新格式、旧格式、两种都写,三者重载后的检索结果逐条一致;在新格式加载的索引上继续 add / train / dump;转换脚本的输出与库 dump 的文件逐字节一致;过期的新格式文件不会被使用;`VQINDEX_DUMP_FORMAT` |
| `test/bench_simd.py` | 打印运行时检测到的指令集;在同一进程内强制切换 SSE4 / AVX2 / AVX-512,对比检索耗时、召回率和结果是否一致 |

单独运行某个脚本:

```bash
mkdir -p /tmp/vq && PYTHONPATH=test python3 test/test.py /tmp/vq
python3 test/test_add.py test /tmp/vq       # 参数:vqindex_py.so 所在目录,临时目录
python3 test/bench_simd.py test /tmp/vq
```

解读结果时注意:

- **分数不是精确内积**。重排用的是 int8 定点量化的向量,分数误差在 1e-3 量级。
- **add 之后要 train 才能检索到**。`add` 只保存向量,`train` 才真正建索引;
  `train_type=2` 是增量方式,沿用已有的分区中心和码本。
- **dump 重载后个别查询结果的顺序可能不同**(分数并列时的次序),`test_add.py` 为此允许 200 条里有 4 条不同。

C 接口 `libvqindex_api.so` 目前没有独立的测试程序,Python 扩展和它共用同一份 `vqindex_api.cc`。
C 接口的调用示例见 README 的 "C API" 一节,各函数的参数约定写在 `vqindex_api.h` 的注释里。

## 4. 索引文件格式(AH 码的两种存法)

索引目录里 AH 码有两种文件,内容等价:

| 文件 | 布局 | 大小 | 谁能读 |
|---|---|---|---|
| `hashed_dataset.npy`(旧格式) | 按入库顺序,每个码 1 字节 | 64 字节/点 | 所有版本,包括 ScaNN 1.2.9 的旧库 |
| `hashed_dataset_packed.bin`(新格式) | 按分区分组,已打包成检索时内存里的布局(每个码 4 位) | 32 字节/点 | 本版本及以后 |

旧格式在加载时要整份读入,再按分区重排、打包,期间多占一份内存;dump 时反过来要解包、合并。
新格式每个分区的数据直接读进最终位置,没有这份开销。2000 万点、128 维索引实测(页缓存已热):

| | 旧格式 | 新格式 |
|---|---|---|
| 加载耗时 | 9.8 秒 | 4.1 秒 |
| 加载时临时多占的内存 | 1.19 GB | 0 |
| dump 耗时 | 13.3 秒 | 4.9 秒 |
| dump 时临时多占的内存 | 3.72 GB | 0.15 GB |

常驻内存两者相同(3.25 GB),检索结果逐条一致。其他文件(`int8_dataset.npy`、`datapoint_to_token.npy` 等)两种格式下完全相同。

**加载**:目录里有新格式文件就用它,没有就用旧格式。新格式文件里记有与 `datapoint_to_token.npy` 对应的校验值,
对不上时(例如旧库在同一目录重新 dump 过,新格式文件已过期)打日志并改用旧格式文件;此时如果没有旧格式文件则加载失败。

**dump**:

| 接口 | 写出的格式 |
|---|---|
| `vqindex_dump(handler)` | 默认格式:新格式。打开索引前设置环境变量 `VQINDEX_DUMP_FORMAT=0/1/2` 可以改变默认值 |
| `vqindex_dump_format(handler, format)` | `DUMP_FORMAT_PACKED`(0)新格式、`DUMP_FORMAT_LEGACY`(1)旧格式、`DUMP_FORMAT_BOTH`(2)两种都写 |
| `vqindex_train_process` 结束时的 dump | 默认格式 |

每次 dump 会删掉本次没有写的那种文件,保证目录里不会同时存在两个不同时刻的码。
写旧格式(1 或 2)时仍有上表"旧格式"一列的内存和耗时开销。Python 扩展对应 `vqindex_py.dump(handler[, format])`。

**格式转换**:`tools/vqindex_convert.py` 在两种格式之间互转,只需要 Python 3 和 numpy,不依赖本库,不加载索引
(只读 `datapoint_to_token.npy` 和被转换的码文件)。2000 万点转换约 11 秒。

```bash
tools/vqindex_convert.py info      <索引目录>            # 查看有哪些格式、新格式文件是否过期
tools/vqindex_convert.py to-packed <索引目录> [--remove-source]   # 旧 -> 新
tools/vqindex_convert.py to-legacy <索引目录> [--remove-source]   # 新 -> 旧
```

索引目录是含 `scann_config.pb` 的那一层(`<index_dir>/index`)。默认保留源文件(两种并存,本库加载时优先用新格式),加 `--remove-source` 才在转换完成后删除源文件。

索引在网络盘上时,直接往网络盘写会很慢,可以用 `--out-dir <目录>` 把结果写到本地盘:该目录里只生成转换出来的那一个文件,
索引目录只读不写。之后把这个文件拷到索引目录(与其他索引文件放在一起)即可,本库加载时不依赖 `scann_assets.pbtxt` 的内容。

```bash
tools/vqindex_convert.py to-packed /mnt/nas/xxx/index --out-dir /data/tmp/xxx
```

转换时源文件从头到尾顺序读、输出顺序写,不依赖页缓存,源文件在网络盘、S3 挂载或缓存是冷的也不会退化成随机 IO。
码在内存缓冲区里重排:内存上限装得下全部码时源文件只读一趟,装不下时每趟处理一部分,源文件顺序读多趟。

- `--memory-gb N`:转换过程的内存占用不超过 N GB(默认取可用内存的一半)。除缓冲区外固定需要每点 4 字节
  (转旧格式 8 字节)加约 0.25 GB,上限低于这个数会直接报错。
- 例:50 GB 的 `hashed_dataset.npy`(约 7.8 亿点)配 `--memory-gb 15`,源文件顺序读 4 趟。
- 实测 2000 万点(码 1.19 GB):不限内存一趟 11 秒、占用 1.4 GB;`--memory-gb 0.6` 分 5 趟,12 秒(转旧格式 6 趟 22 秒),
  实际峰值 0.47 GB。多趟与一趟的输出逐字节相同。

转换期间不要对该索引 dump 或 train。

## 5. 与旧版(ScaNN 1.2.9)的兼容性和升级步骤

### 兼容性

- 原有 API 不变(`vqindex_api.h` 里原有的函数和结构体原样保留,只新增了 `vqindex_dump_format`),
  按旧头文件编译的程序**不用重新编译**,直接换用新的 so。
- 新库不再依赖 `libtensorflow_framework.so.2`,只依赖 libstdc++、libm、libc。
  它要求运行机器的 glibc / libstdc++ 不低于编译机的版本(Ubuntu 24.04 上编出来的库需要 glibc 2.38 以上),
  所以要在需要支持的最老的系统上编译。
- 旧版 dump 的索引新版可以加载,并且能继续 add / train / dump。
- 新版 dump 的索引,旧版只能读旧格式:需要回退到旧库时,用格式 1 或 2 来 dump(或设 `VQINDEX_DUMP_FORMAT`),
  或者用 `tools/vqindex_convert.py to-legacy` 转换。注意不带参数的 `vqindex_dump` 默认只写新格式并删除 `hashed_dataset.npy`。
  这是在当前索引配置(tree-AH + int8 重排)下实测的结果;如果以后在新版里启用旧版没有的配置项,旧版将无法读取。
- `stats` 里的 `index_nlist` 现在返回配置的分区数,旧版返回实际的叶子数。

### 相对旧版修掉的问题

- add / train / dump 互相串行;search 在读索引状态之前先登记,train 不会再在检索进行中把索引换掉;
  并发 add 不会从同一个偏移开始写;stats 不再访问正在被替换的索引。
- 检索结果按结果条数而不是分数正负来映射 vid,内积为负的结果不再丢 vid。
- 子进程训练(`vqindex_train_process`)的返回值和退出方式。
- MEMORY 存储模式下 dump 只追加上次加载/dump 之后新增的部分。

### 升级步骤

1. 用新编出来的 `libvqindex_api.so` 替换旧的,重启服务。调用方程序不用动。
2. 现有索引原样可用。想要更快的加载,二选一:
   - 离线用 `tools/vqindex_convert.py to-packed` 转换(转换期间该索引不能 dump / train;默认保留旧文件,两种并存时加载用新格式);
   - 等下一次 dump,默认就写新格式。
3. 需要保留回退到旧库的可能时,保证目录里有 `hashed_dataset.npy`:转换时不加 `--remove-source`,
   并且设 `VQINDEX_DUMP_FORMAT=2`(两种都写,代价是 dump 时仍有旧格式那份内存和耗时开销)。

### 实测对比

同一台机器(双路 Xeon)、同一份线上索引(9233 万点,128 维,16384 个分区),旧库读旧格式,新库读新格式:

| | 旧版(ScaNN 1.2.9) | 本版本 |
|---|---|---|
| 加载耗时 | 56–62 秒 | 25–28 秒 |
| 加载期间内存峰值 | 20.5 GB | 16.8 GB |
| 加载后常驻内存 | 约 16.8 GB | 约 16.8 GB |
| 检索结果(5500 条查询) | - | 与旧版逐条相同 |
| 并发吞吐(16 / 32 / 64 并发,6 轮交替取中位数) | 1 | 1.12 / 1.01 / 1.13 |

吞吐的差别在轮次之间的波动范围内,可以认为不慢于旧版。没有验证过的:macOS、ARM、非 tree-AH 的索引配置。

## 6. 常见问题

| 现象 | 原因与处理 |
|---|---|
| `Illegal instruction` | 运行机器不支持基线指令集。默认的 `-mavx` 需要 CPU 支持 AVX;若库是用更高的 `VQINDEX_COPTS` 编的,去掉后重新编译 |
| `ImportError` / `undefined symbol: _Py...` | 运行的 Python 版本与编译时不同 |
| `Python.h not found` | 安装 `python3-dev` |
| 大量 `__m128` 相关编译错误 | 用了 gcc,改用 clang |
| `Bazel 7.x is required` | 运行 `tools/install_bazel.sh` |
| 首次编译卡在下载 | 需要能访问 `bcr.bazel.build` 和 `github.com`,可设置 `https_proxy` |
