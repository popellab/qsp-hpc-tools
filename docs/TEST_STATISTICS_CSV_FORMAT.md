# Test Statistics CSV Format

## Overview

Test statistics are defined in a CSV file that serves as the single source of truth for both the HPC derivation workflow and local SBI training. The CSV contains metadata about each test statistic along with the Python code needed to compute it from simulation outputs.

## Required Columns

### Core Fields

- **`test_statistic_id`** (string): Unique identifier for the test statistic
  - Example: `"tumor_log_growth_rate_0_60d"`
  - Used to match test statistics across training/validation runs

- **`required_species`** (string): Comma-separated list of species needed for computation
  - Format: `"Compartment.Species1, Compartment.Species2"`
  - Example: `"V_T.C1, V_T.CD8"`
  - Species names with dots will be converted to underscores in Parquet: `V_T.C1` → `V_T_C1`

- **`python_function`** (string): Python function code to compute the test statistic
  - Must define a function named `compute`
  - Signature: `def compute(time: np.ndarray, species1: np.ndarray, ...) -> float`
  - First argument is always `time`, followed by species in the order listed in `required_species`
  - Available imports: `np` (numpy), `numpy`
  - Returns: scalar float value (or `np.nan` for failed computations)

### Observational Data Fields (for calibration/validation)

- **`mean`** (float): Observed mean from experimental data
- **`variance`** (float): Observed variance
- **`ci95_lower`**, **`ci95_upper`** (float): 95% confidence interval bounds
- **`units`** (string): Physical units (e.g., `"1/day"`, `"cells"`, `"dimensionless"`)
- **`n_sources`** (int): Number of data sources aggregated
- **`description`** (string): Human-readable description

## CSV Example

```csv
test_statistic_id,required_species,python_function,mean,variance,ci95_lower,ci95_upper,units,description
tumor_log_growth_rate_0_60d,V_T.C1,"def compute(time, V_T_C1):
    # Compute exponential growth rate from tumor cells over 0-60 days
    t0, t1 = 0, 60
    if len(time) == 0 or len(V_T_C1) == 0:
        return np.nan

    # Interpolate to daily grid
    t_eval = np.arange(t0, t1 + 1, 1.0)
    c1_interp = np.interp(t_eval, time, V_T_C1)

    # Protect against non-positive values
    c1_interp = np.maximum(c1_interp, np.finfo(float).eps)

    # Log-transform and fit line
    y = np.log(c1_interp)
    slope = np.polyfit(t_eval, y, 1)[0]

    return slope",0.0059,8.50e-06,1.86e-04,0.01161,1/day,Tumor exponential growth rate
cd8_treg_ratio_baseline,"V_T.CD8, V_T.Treg","def compute(time, V_T_CD8, V_T_Treg):
    # Compute CD8:Treg ratio at day 0
    t_eval = 0.0

    cd8_0 = np.interp(t_eval, time, V_T_CD8)
    treg_0 = np.interp(t_eval, time, V_T_Treg)

    if not (np.isfinite(cd8_0) and np.isfinite(treg_0)):
        return np.nan
    if cd8_0 < 0 or treg_0 < 0:
        return np.nan

    # Protect against division by zero
    eps_denom = 1.0e-12
    return cd8_0 / max(treg_0, eps_denom)",2.5,0.5,1.2,3.8,dimensionless,Baseline CD8/Treg ratio in tumor
peak_immune_response_7d,"V_L.MDC, V_T.Teff","def compute(time, V_L_MDC, V_T_Teff):
    # Compute peak combined immune response in first 7 days
    t_window_end = 7.0

    # Filter to time window
    mask = time <= t_window_end
    if not np.any(mask):
        return np.nan

    time_window = time[mask]
    mdc_window = V_L_MDC[mask]
    teff_window = V_T_Teff[mask]

    # Combined immune response
    combined = mdc_window + teff_window

    return np.max(combined)",1.5e8,2.0e15,5.0e7,2.5e8,cells,Peak immune response within 7 days
```

## Function Code Guidelines

### Basic Structure

```python
def compute(time, species_1, species_2, ...):
    """
    Compute test statistic from simulation output.

    Args:
        time: np.ndarray of time points (days)
        species_1: np.ndarray of first species values
        species_2: np.ndarray of second species values (if needed)
        ...

    Returns:
        float: computed test statistic value, or np.nan if computation fails
    """
    # 1. Input validation
    if len(time) == 0 or len(species_1) == 0:
        return np.nan

    # 2. Define analysis window
    t_start, t_end = 0, 60

    # 3. Extract/interpolate data
    t_eval = np.arange(t_start, t_end + 1, 1.0)
    interp_values = np.interp(t_eval, time, species_1)

    # 4. Compute statistic
    result = some_computation(interp_values)

    # 5. Validate output
    if not np.isfinite(result):
        return np.nan

    return result
```

### Common Patterns

#### Time Window Extraction
```python
# Filter to specific time range
mask = (time >= t_start) & (time <= t_end)
time_window = time[mask]
species_window = species[mask]
```

#### Interpolation to Regular Grid
```python
# Daily grid from day 0 to day 60
t_eval = np.arange(0, 61, 1.0)
interp_values = np.interp(t_eval, time, species)
```

#### Baseline Value (t=0)
```python
# Extract value at equilibrium (day 0)
baseline_value = np.interp(0.0, time, species)
```

#### Growth Rate Estimation
```python
# Exponential growth: C(t) = C0 * exp(r*t)
# Log-transform: ln(C) = ln(C0) + r*t
# Slope of ln(C) vs t gives growth rate r
y = np.log(np.maximum(species, np.finfo(float).eps))
slope = np.polyfit(time_grid, y, 1)[0]  # r (1/day)
```

#### Fold Change
```python
# Ratio of values at two time points
value_t1 = np.interp(t1, time, species)
value_t0 = np.interp(t0, time, species)
fold_change = value_t1 / max(value_t0, 1e-12)  # Protect against division by zero
```

#### Peak Detection
```python
# Maximum value in time window
peak_value = np.max(species_window)

# Time to peak
peak_idx = np.argmax(species_window)
time_to_peak = time_window[peak_idx]
```

### Error Handling

Always return `np.nan` for invalid computations:

```python
# Check for empty data
if len(time) == 0 or len(species) == 0:
    return np.nan

# Check for non-finite values
if not (np.isfinite(value1) and np.isfinite(value2)):
    return np.nan

# Check for non-physical values
if value < 0:
    return np.nan

# Protect against division by zero
result = numerator / max(denominator, 1e-12)

# Protect against log(0)
log_values = np.log(np.maximum(values, np.finfo(float).eps))
```

## CSV Formatting Notes

### Multiline Function Code

Python function code with newlines should be properly quoted in the CSV:

```python
"def compute(time, V_T_C1):
    result = some_calculation()
    return result"
```

When editing in a spreadsheet application (Excel, Google Sheets):
- Paste the entire function into the cell
- The application will handle quoting automatically
- Newlines within cells are preserved

When editing as plain text:
- Enclose multiline strings in double quotes
- Escape internal quotes: `\"text\"`
- Use actual newlines (not `\n` escape sequences)

### Recommended Workflow

1. **Draft functions in Python IDE** with syntax highlighting
2. **Test functions** with sample data
3. **Copy tested code** into CSV `python_function` column
4. **Validate CSV** loads correctly: `pd.read_csv("test_stats.csv")`

## Backwards Compatibility

The system maintains backwards compatibility:

- **Legacy CSVs without `python_function`**: Worker will skip these and emit warnings
- **Old `model_output_code` column** (MATLAB): Ignored by derivation worker (deprecated)

## Migration from test_stat_functions.py

If you have an existing `test_stat_functions.py` file:

1. **Extract functions** from the Python module
2. **For each function**, add a row to CSV with:
   - `test_statistic_id` matching the function name in registry
   - `required_species` listing the species arguments
   - `python_function` containing the function code
3. **Remove** the old `test_stat_functions.py` file
4. **Update** any documentation referencing the old approach

Example migration:

**Old approach** (`test_stat_functions.py`):
```python
def tumor_growth_rate_0_60d(time, V_T_C1):
    # ... function code ...
    return slope

TEST_STAT_REGISTRY = {
    "tumor_growth_rate_0_60d": tumor_growth_rate_0_60d,
}

def get_test_stat_function(test_stat_id):
    return TEST_STAT_REGISTRY[test_stat_id]
```

**New approach** (CSV row):
```csv
tumor_growth_rate_0_60d,V_T.C1,"def compute(time, V_T_C1):
    # ... function code ...
    return slope"
```

Note the function is now named `compute` (standardized) instead of the test statistic ID.
