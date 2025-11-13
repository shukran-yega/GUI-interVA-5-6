# VMan3 Data Processing Toolkit

A Python package for processing and quality checking VA data.
This package uses interva6 algorthim based off the interva2022 R scripts

## Features
- Automatically marks skipped questions based on relevance expressions
- Comprehensive data cleaning pipeline
- Handles complex ODK relevance expressions
- Case-insensitive variable matching
- Detailed debugging output

## Installation

```bash
pip install vman3
```

### Local development (to run tests/test.py)
```bash
cd vman3
pip install -r requirements.txt
# Linux/macOS
export PYTHONPATH=.
# Windows PowerShell
$env:PYTHONPATH = "."
python -m tests.test
# or
python tests/test.py
```

## Simple usage

### Clean data (mirrors tests/test.py)
```python
import pandas as pd
import vman3 as vman
from importlib.resources import files

try:
    # Load packaged sample data
    data = pd.read_csv(files("vman3.data").joinpath("sample_data2.csv"))

    # Clean
    cleaned = vman.clean(data, verbose=True)

    # Quick summary
    print((data.shape, cleaned.shape))
except Exception as e:
    # Print any error to terminal
    print(f"clean error: {e}")
```

### Transform for CCVA
```python
import pandas as pd
import vman3 as vman
from importlib.resources import files

try:
    # Load WHO 2016 sample
    data = pd.read_csv(files("vman3.data").joinpath("sample_data.csv"))

    # Transform to InterVA5-style inputs
    ccva_df = vman.pycrossva(
        input_data=data,
        input_format="2016WHOv151",
        output_format="InterVA5",
        raw_data_id="instanceID",
        lower=True,
        verbose=3,
    )
    print(ccva_df.shape)
except Exception as e:
    print(f"transform error: {e}")
```