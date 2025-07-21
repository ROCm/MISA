# Environment Variables for `conv_driver.exe`

The following environment variables can be used to control the behavior of the `conv_driver.exe` driver in the `driver` directory. Unless otherwise noted, all variables are optional and if unset the driver uses its built‑in defaults.

---

## 🔧 File and Configuration Paths

| Variable                        | Description                                                                                                    | Default                                                      |
|---------------------------------|---------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| `IGEMM_HSACO`                   | Path to a HSACO code‑object to load instead of the one generated in the current directory.                     | Generated `.hsaco` in the working directory :contentReference[oaicite:0]{index=0} |
| `IGEMM_CONFIG_FILE`             | Path to a custom JSON/text file containing kernel tuning configurations to use instead of in‑code defaults.   | "igemm_gtc.config" (driver uses built‑in configurations)                   |
| `IGEMM_TENSOR_CAST_HSACO`       | Path to a tensor‑cast HSACO file (for data‑type conversion kernels).                                           | "igemm_gtc_tensor_cast.hsaco"                                                         |
| `TRANSPOSE_HSACO`               | Path to a transpose HSACO file (for tensor transpose kernels).                                                 | "igemm_gtc.hsaco"                                                         |
| `GENERAL_TENSOR_REORDER_HSACO`  | Path to a general tensor‑reorder HSACO file (for arbitrary reorder kernels).                                   | "out/general_tensor_reorder.hsaco"                                                         |

---

## ⚙️ Kernel Selection and Filtering

| Variable                  | Description                                                                                                | Default                          |
|---------------------------|-------------------------------------------------------------------------------------------------------------|----------------------------------|
| `IGEMM_RUN_ONLY_KERNEL`   | If set (e.g. to a kernel identifier), run only that kernel instead of the full search.                      | Run all kernels                  |
| `IGEMM_KVALID_TARGET`     | If set to `1`, skip kernels deemed “invalid” for the target GPU (based on heuristic).                      | `0` (run all valid kernels)      |
| `IGEMM_MODE`              | Select execution mode; e.g. `VECTOR_C` to use the vector‑C path of the driver.                              | Default mode as compiled         |

---

## 🏎 Performance Tuning

| Variable                | Description                                                                                      | Default                                                      |
|-------------------------|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| `IGEMM_SCLK_MHZ`        | GPU SCLK in MHz – used to compute kernel efficiency metrics.                                      | Read from device :contentReference[oaicite:1]{index=1}                          |
| `IGEMM_MAX_MPB`         | Maximum M‑tiles per block to consider during tuning.                                             | Code‑compiled limit                                         |
| `IGEMM_MAX_NPB`         | Maximum N‑tiles per block to consider during tuning.                                             | Code‑compiled limit                                         |
| `IGEMM_MAX_KPB`         | Maximum K‑tiles per block to consider during tuning.                                             | Code‑compiled limit                                         |
| `IGEMM_MAX_GKS`         | Maximum group‑size (GKS) to consider for collective tuning.                                       | Code‑compiled limit                                         |
| `IGEMM_GKS_ITERATIVE`   | If `1`, enable iterative group‑size search instead of exhaustive.                                 | `0` (exhaustive search)                                      |
| `GRID_SIZE`             | Override the computational grid size (number of work‑groups) instead of auto‑computed.             | Auto‑compute based on problem dimensions                     |
| `VECTOR_C`              | Alias for setting `IGEMM_MODE=VECTOR_C`.                                                         | —                                                            |

---

## 🏁 Execution Control

| Variable                       | Description                                                                                          | Default            |
|--------------------------------|------------------------------------------------------------------------------------------------------|--------------------|
| `IGEMM_WARMUP`                 | Number of warm‑up iterations before timing.                                                          | `1`                |
| `IGEMM_REPEAT`                 | Number of timed repetitions to perform for each kernel.                                              | `5`                |
| `IGEMM_SLEEP_MS`               | Milliseconds to sleep between kernel launches (useful for power‑limited profiling).                   | `0`                |
| `IGEMM_RUN_ONLY_KERNEL`        | (See above)                                                                                          | —                  |

---

## 🐞 Logging and Debug

| Variable                      | Description                                                                                          | Default            |
|-------------------------------|------------------------------------------------------------------------------------------------------|--------------------|
| `IGEMM_LOG_FASTEST_CONFIG`    | If `1`, after testing all kernels only the fastest configuration is printed.                         | `0` :contentReference[oaicite:2]{index=2} |
| `IGEMM_VERBOSE`               | If `1`, enable verbose logging of each kernel’s performance metrics.                                 | `0`                |
| `IGEMM_ASSERT_WHEN_INVALID`   | If `1`, abort execution when an “invalid” kernel configuration is encountered.                       | `0`                |
| `DBG_MODE`                    | If `1`, enable extra debug checks in the driver.                                                     | `0`                |
| `PER_PIXEL_CHECK`             | If `1`, perform per‑output‑pixel correctness checks.                                                 | `0`                |
| `PER_PIXEL_CHECK_PRINT`       | If `1`, print detailed per‑pixel check results.                                                     | `0`                |
| `PRINT_EVERY_PIXEL`           | If `1`, print every pixel’s result to stdout (very verbose).                                        | `0`                |
| `PRINT_NRMS`                  | If `1`, print normalized root‑mean‑square errors for each kernel.                                    | `0`                |
| `VALID_FLOAT`                 | If `1`, validate floating‑point outputs against a reference.                                         | `0`                |
| `DUMP_PRED`                   | If `1`, dump predicted output values to disk for analysis.                                           | `0`                |
| `IGEMM_DUMP_GMAP`             | If `1`, dump the generated “gmap” (group‑mapping) to disk.                                           | `0`                |
| `IGEMM_DUMPDIR_ALL`           | Directory path to which all dumps (gmap, outputs, logs) will be written.                             | None               |

---

## 🧪 Miscellaneous

| Variable                  | Description                                                                                          | Default            |
|---------------------------|------------------------------------------------------------------------------------------------------|--------------------|
| `IGEMM_RAND_INT`          | Seed for any random integer operations (e.g. in heuristic skips).                                   | Derived from time  |
| `IGEMM_BENCH_CSV`         | Path to output a CSV file of benchmark results.                                                     | None               |
| `IGEMM_CHECK_TRANSPOSE`   | If `1`, validate the transpose kernels after generation.                                            | `0`                |
| `GRID_SIZE`               | (See Performance Tuning)                                                                             | —                  |

---