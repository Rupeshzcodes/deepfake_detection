import sys
try:
    import pandas as pd
    print(f"Pandas: {pd.__version__} [OK]")
except ImportError as e:
    print(f"Pandas: [FAIL] - {e}")

try:
    import plotly
    print(f"Plotly: {plotly.__version__} [OK]")
except ImportError as e:
    print(f"Plotly: [FAIL] - {e}")
