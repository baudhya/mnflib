# Bug Fixes and Refactoring — mnflib Production Review

This document records every correctness bug, design flaw, and quality issue found
during the production-readiness audit of the initial `mnflib` implementation,
along with the exact fix applied in each case.

---

## 1. `mnflib/utils.py` — Noise Estimation

### Bug 1.1 — Four-pixel estimator: duplicate term (silent correctness bug)

**File:** `mnflib/utils.py` → `calculate_noise_four_pixel`

**What was wrong:**

The four-pixel noise estimator averages four independent directional differences:
horizontal, vertical, below-horizontal, and diagonal. The original code computed
the "below-horizontal" term (line 3) and the "diagonal" term (line 4) using
identical expressions — the below-horizontal slice was copy-pasted for both:

```python
# WRONG — original code (both line 3 and line 4 were identical)
noise3 = image3D[1:, :, :-1] - image3D[1:, :, 1:]   # below-horizontal
noise4 = image3D[1:, :, :-1] - image3D[1:, :, 1:]   # ← same as noise3 (BUG)
```

The correct diagonal term uses pixels from different rows *and* columns:

```python
noise4 = image3D[:-1, :, :-1] - image3D[1:, :, 1:]  # true diagonal
```

**Impact:** The noise covariance matrix computed by `four_pixel` was biased —
it double-counted the below-horizontal direction and completely omitted the
diagonal direction. This would make `four_pixel` produce a slightly different
but consistently wrong noise estimate, with no error or warning of any kind.

**Fix:**

```python
def calculate_noise_four_pixel(image3D):
    noise = (image3D[:-1, :, :-1] - image3D[:-1, :, 1:])    # horizontal
    noise = noise + (image3D[:-1, :, :-1] - image3D[1:, :, :-1])  # vertical
    noise = noise + (image3D[1:, :, :-1] - image3D[1:, :, 1:])    # below-horizontal
    noise = noise + (image3D[:-1, :, :-1] - image3D[1:, :, 1:])   # diagonal (fixed)
    return noise / 4.0
```

---

### Bug 1.2 — All noise estimators used Python `for` loops over lines

**File:** `mnflib/utils.py`

**What was wrong:**

All three noise estimators (`next_pixel`, `three_pixel`, `four_pixel`) computed
per-line differences using an explicit Python loop:

```python
# WRONG — O(lines) Python overhead
for l in range(lines):
    noise[l, :, :] = image3D[l, :, :-1] - image3D[l, :, 1:]
```

**Impact:** Severe performance degradation on large images. A 2199-line image
ran the Python interpreter 2199 times through the loop body instead of issuing
a single vectorized NumPy call.

**Fix:** All estimators were rewritten as fully vectorized NumPy slice
operations operating on the entire 3-D array at once:

```python
def calculate_noise_next_pixel(image3D):
    return image3D[:, :, :-1] - image3D[:, :, 1:]
```

---

### Bug 1.3 — `save_to_tiff` discarded all geospatial metadata

**File:** `mnflib/utils.py` → `save_to_tiff`

**What was wrong:**

The original `save_to_tiff` created a minimal rasterio profile from scratch,
ignoring the CRS, transform, projection, and tiling information from the
source image:

```python
# WRONG — metadata lost
profile = {
    "driver": "GTiff",
    "dtype": "float32",
    "width": samples,
    "height": lines,
    "count": bands,
}
```

**Impact:** Every output GeoTIFF produced by the library was stripped of its
coordinate reference system and geotransform. The output files could not be
overlaid with the input in any GIS application.

**Fix:** `save_to_tiff` now accepts an optional `profile` parameter. When
provided (sourced from `GeotifImageLoader.profile`), it is used as the base
profile so all geospatial metadata is preserved:

```python
def save_to_tiff(image, filename, profile=None):
    if profile is not None:
        out_profile = profile.copy()
        out_profile.update({"dtype": "float32", "count": bands, ...})
    else:
        out_profile = {minimal fallback}
```

---

## 2. `mnflib/data_structures.py` — Statistics Accumulation

### Bug 2.1 — `ImageStatistics.update_with_line`: incorrect incremental covariance formula

**File:** `mnflib/data_structures.py` → `ImageStatistics.update_with_line`

**What was wrong:**

The Welford-style parallel batch update formula for combining the running
covariance with a new batch requires capturing `old_n` **before** incrementing
`self.n`. The original code incremented `self.n` first and then used it in the
cross-term calculation, which produced an algebraically wrong correction factor:

```python
# WRONG — n already incremented when cross-term is computed
self.n += samples
mean_delta = line_mean - self.means
self.means += (samples / self.n) * mean_delta
self.C += (samples * self.n / ...) * np.outer(...)  # self.n is wrong here
```

**Impact:** The incremental covariance matrix accumulated by `Line_By_Line_MNF`
was numerically incorrect. The error grew with each line processed. The
resulting noise and image covariance matrices were biased, which corrupted the
MNF eigendecomposition and therefore the forward/inverse transforms.

**Fix:**

```python
def update_with_line(self, line, samples):
    if samples <= 0:
        return
    old_n = self.n          # capture BEFORE incrementing
    self.n += samples
    line = line.astype(np.float64, copy=False)
    line_mean = line.mean(axis=1)
    line_centered = line - line_mean[:, None]
    self.C += line_centered @ line_centered.T
    mean_delta = line_mean - self.means
    self.means += (samples / self.n) * mean_delta
    self.C += (samples * old_n / self.n) * np.outer(mean_delta, mean_delta)
```

---

### Bug 2.2 — `ImageStatisticsFull.update` used sample covariance instead of population covariance

**File:** `mnflib/data_structures.py` → `ImageStatisticsFull.update`

**What was wrong:**

`ImageStatisticsFull` (used by whole-image `MNF`) called `np.cov` without
`bias=True`, which divides by `n − 1` (sample covariance). Meanwhile,
`ImageStatistics.get_cov()` (used by `Line_By_Line_MNF`) divides by `n`
(population covariance). This meant the two processing modes produced
numerically different covariance matrices for the same data, making their
MNF results inconsistent with each other.

**Fix:** Added `bias=True` to force population covariance:

```python
self.C = np.cov(image, bias=True)
```

---

### Bug 2.3 — `ImageStatistics.get_cov()` silently returned zeros before any data

**File:** `mnflib/data_structures.py` → `ImageStatistics.get_cov`

**What was wrong:**

If `get_cov()` was called before any line had been processed (`self.n == 0`),
it would divide by zero silently and return a NaN or zero matrix, which would
propagate through the eigendecomposition.

**Fix:** Added an explicit guard:

```python
def get_cov(self):
    if self.n == 0:
        raise RuntimeError("No data has been accumulated yet.")
    return self.C / self.n
```

---

### Bug 2.4 — `MNFConfig` had no `profile` field

**File:** `mnflib/data_structures.py` → `MNFConfig`

**What was wrong:**

`MNFConfig` had no way to carry the rasterio profile from the image loader to
`save_to_tiff`. Even after fixing `save_to_tiff` to accept a profile, there
was no plumbing to pass it through.

**Fix:** Added `profile: Optional[dict] = None` to the `MNFConfig` dataclass,
threaded from `GeotifImageLoader.profile` via `cli/main.py`.

---

## 3. `mnflib/mnf.py` — Whole-Image MNF

### Bug 3.1 — Eigendecomposition computed twice for `RUN_BOTH`

**File:** `mnflib/mnf.py` → `MNF`

**What was wrong:**

When `TransformDirection.RUN_BOTH` was requested, the forward pass computed
and discarded the eigenvectors, and then the inverse pass recomputed them from
scratch. The `scipy.linalg.eigh` call on a large covariance matrix is the
most expensive step in the entire pipeline.

**Fix:** Introduced `_get_eigen()` with a cache:

```python
def _get_eigen(self):
    if self._eigen_cache is not None:
        return self._eigen_cache
    # ... compute eigh, sort eigenvalues ascending, invert F ...
    self._eigen_cache = (eigvecs, F_inv, eigvals)
    return self._eigen_cache
```

Both `_run_forward` and `_run_inverse` now call `_get_eigen()`, so the
decomposition is computed exactly once regardless of direction.

---

### Bug 3.2 — No input validation

**File:** `mnflib/mnf.py` → `MNF.__init__`

**What was wrong:**

`percentageOfBandsInInverse` was never validated, so values outside `(0, 1]`
would silently produce wrong inverse transforms (e.g. keeping zero bands or
more bands than exist). `noiseMatrixCalculation` was never validated, so
unrecognised method names would propagate to `utils.py` and raise an obscure
`AttributeError` deep in processing.

**Fix:**

```python
if not (0 < cfg.percentageOfBandsInInverse <= 1.0):
    raise ValueError("percentageOfBandsInInverse must be in (0, 1]")
valid_methods = {"next_pixel", "three_pixel", "four_pixel"}
if cfg.noiseMatrixCalculation not in valid_methods:
    raise ValueError(f"noiseMatrixCalculation must be one of {valid_methods}")
```

---

## 4. `mnflib/mnf_linebyline.py` — Line-by-Line MNF

### Bug 4.1 — Standalone `RUN_INVERSE` not supported

**File:** `mnflib/mnf_linebyline.py`

**What was wrong:**

`Line_By_Line_MNF` had no mechanism to save covariance statistics from a
forward pass and reload them for a subsequent inverse-only pass. Requesting
`RUN_INVERSE` alone would fail or produce garbage because no covariance had
been accumulated.

**Fix:** Added `_save_stats()` and `_load_stats()` that serialize the
`ImageStatistics` objects to `stats_data/`, mirroring the approach already
used by whole-image `MNF`. A dedicated `_run_inverse_pass()` loads the saved
stats and applies the inverse transform using the fully accumulated covariance.

---

### Bug 4.2 — Dead code left in place

**File:** `mnflib/mnf_linebyline.py`

**What was wrong:**

Two methods — `mnf_linebyline_add_mean` and `mnf_linebyline_remove_mean` —
were defined but never called anywhere in the codebase.

**Fix:** Both methods were removed.

---

## 5. `mnflib/scores.py` — Quality Metrics

### Bug 5.1 — Typo in public API: `mean_squre_error`

**File:** `mnflib/scores.py`

**What was wrong:**

The function was named `mean_squre_error` (missing an 'a'), making it
unusable via the documented API name.

**Fix:** Renamed to `mean_squared_error`.

---

### Bug 5.2 — `get_ssim()` passed a 3-D array to a 2-D SSIM function

**File:** `mnflib/scores.py` → `QualityMetricCalculator.get_ssim`

**What was wrong:**

The original `get_ssim` passed the entire `(lines, bands, samples)` 3-D array
directly to `skimage.metrics.structural_similarity`, which expects a 2-D
image. On small test images this raised a `ValueError` about window size
exceeding image extent; on larger images it silently computed a meaningless
scalar over the wrong axes.

**Fix:** `get_ssim` now computes SSIM per spectral band (each band is a 2-D
`lines × samples` slice) and returns the mean:

```python
def get_ssim(self):
    scores = [
        ssim_score(self.original[:, b, :], self.processed[:, b, :])
        for b in range(self.bands)
    ]
    return float(np.mean(scores))
```

---

### Bug 5.3 — `ssim_score` used a fixed window size that could exceed small images

**File:** `mnflib/scores.py` → `ssim_score`

**What was wrong:**

The default 7×7 SSIM window would raise a `ValueError` on any spatial
dimension smaller than 7 (common in unit tests).

**Fix:** Window size is computed adaptively:

```python
min_dim = min(image_a.shape[-2], image_a.shape[-1])
win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
```

---

### Bug 5.4 — Division by zero in `psnr_score` and `snr`

**File:** `mnflib/scores.py`

**What was wrong:**

`psnr_score` divided by RMSE without checking for zero (identical images).
`snr` divided by standard deviation without checking for zero (constant band).
Both would raise `ZeroDivisionError` or return `nan`.

**Fix:**

```python
def psnr_score(original, denoised):
    rmse = root_mean_squared_error(original, denoised)
    if rmse == 0.0:
        return float("inf")
    ...

def snr(band):
    std = np.std(band)
    if std == 0.0:
        return float("inf")
    ...
```

---

### Bug 5.5 — No shape validation in `QualityMetricCalculator`

**File:** `mnflib/scores.py` → `QualityMetricCalculator.__init__`

**What was wrong:**

Passing mismatched images would produce wrong metric values or crash deep
inside NumPy with an obscure broadcast error.

**Fix:**

```python
if original_image.shape != processed_image.shape:
    raise ValueError(
        f"Shape mismatch: original {original_image.shape} vs "
        f"processed {processed_image.shape}"
    )
```

---

## 6. `mnflib/dataloader.py` — Image Loading

### Bug 6.1 — `image_subset` used incorrect rasterio windowing

**File:** `mnflib/dataloader.py` → `GeotifImageLoader.image_subset`

**What was wrong:**

The original `image_subset` read the full image and then sliced the NumPy
array. For large images this defeated the purpose of subsetting (the full
image was still loaded into memory). Additionally, the slice indices were
applied in the wrong order relative to the `(lines, bands, samples)` layout.

**Fix:** Uses `rasterio.windows.Window` to read only the requested region
directly from disk:

```python
window = rasterio.windows.Window(
    col_off=s.startsSamp, row_off=s.startLine,
    width=s.endSamp - s.startsSamp,
    height=s.endLine - s.startLine,
)
data = src.read(window=window)
```

---

### Bug 6.2 — File-not-found raised bare `Exception`

**File:** `mnflib/dataloader.py`

**What was wrong:**

A missing file raised `Exception("File not found")` instead of the standard
`FileNotFoundError`, making it impossible for callers to catch it specifically.

**Fix:** Raises `FileNotFoundError` with a descriptive message.

---

## 7. New Tests Added

The original library had zero automated tests. The following test modules were
created:

| File | What it covers |
|------|---------------|
| `tests/test_utils.py` | Shape correctness of all three noise estimators; four-pixel correctness verifying all four distinct directional terms |
| `tests/test_data_structures.py` | `ImageStatistics` matches `np.cov(bias=True)` for single line; incremental accumulation matches batch; `get_cov` raises before data; round-trip persistence via `write_to_file`/`read_from_file`; `MNFConfig` keyword construction and optional profile |
| `tests/test_mnf.py` | Whole-image round-trip (100% bands → exact reconstruction within `atol=1e-3`); input validation raises on bad arguments; eigen cache hit (decomposition called once for `RUN_BOTH`) |
| `tests/test_scores.py` | Identical images → MSE=0, PSNR=inf; shape mismatch → `ValueError`; per-band SSIM path; all metric keys present in `calculate_all_metrics` |

---

## 8. Summary Table

| # | File | Category | Severity | Description |
|---|------|----------|----------|-------------|
| 1.1 | `utils.py` | Correctness | **Critical** | Four-pixel estimator: duplicate term, diagonal never computed |
| 1.2 | `utils.py` | Performance | High | Python loops over lines instead of vectorized NumPy |
| 1.3 | `utils.py` | Data loss | High | `save_to_tiff` discarded CRS / geotransform |
| 2.1 | `data_structures.py` | Correctness | **Critical** | Welford update: `old_n` captured after increment → wrong covariance |
| 2.2 | `data_structures.py` | Correctness | High | `ImageStatisticsFull` used sample covariance (`n−1`), inconsistent with `ImageStatistics` |
| 2.3 | `data_structures.py` | Robustness | Medium | `get_cov()` silent NaN before any data |
| 2.4 | `data_structures.py` | Design | Medium | `MNFConfig` missing `profile` field |
| 3.1 | `mnf.py` | Performance | Medium | Eigendecomposition computed twice for `RUN_BOTH` |
| 3.2 | `mnf.py` | Robustness | Medium | No input validation on config parameters |
| 4.1 | `mnf_linebyline.py` | Missing feature | High | Standalone `RUN_INVERSE` not supported |
| 4.2 | `mnf_linebyline.py` | Code quality | Low | Dead methods never called |
| 5.1 | `scores.py` | API | Low | Public function name had typo |
| 5.2 | `scores.py` | Correctness | **Critical** | SSIM computed over 3-D array instead of per-band 2-D slices |
| 5.3 | `scores.py` | Robustness | Medium | Fixed SSIM window size crashes on small images |
| 5.4 | `scores.py` | Robustness | Medium | Division by zero in `psnr_score` and `snr` |
| 5.5 | `scores.py` | Robustness | Medium | No shape validation → silent wrong metrics |
| 6.1 | `dataloader.py` | Correctness | Medium | `image_subset` loaded full image, wrong slice order |
| 6.2 | `dataloader.py` | API | Low | Bare `Exception` instead of `FileNotFoundError` |
