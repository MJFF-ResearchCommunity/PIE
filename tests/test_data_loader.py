# test_data_loader.py
import logging
from pie_clean import * # DataLoader and constants

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Test loading specific modalities
data = DataLoader.load(
    data_path="./PPMI",
    modalities=[
        SUBJECT_CHARACTERISTICS,
        MEDICAL_HISTORY
    ]
)

# Print summary of loaded data
print("\n=== Data Loading Test Results ===")

# Check subject characteristics
if not data[SUBJECT_CHARACTERISTICS].empty:
    print(f"Subject characteristics: {len(data[SUBJECT_CHARACTERISTICS])} rows")
    print("First few rows:")
    print(data[SUBJECT_CHARACTERISTICS].head(3))
else:
    print("Subject characteristics: No data loaded")

# Check medical history
if data[MEDICAL_HISTORY]:
    print(f"Medical history tables: {len(data[MEDICAL_HISTORY])} tables")
    for table_name, df in data[MEDICAL_HISTORY].items():
        print(f"  - {table_name}: {len(df)} rows")
    
    # Show a sample from the first table
    if data[MEDICAL_HISTORY]:
        first_table = next(iter(data[MEDICAL_HISTORY].values()))
        print("\nSample from first medical history table:")
        print(first_table.head(3))
else:
    print("Medical history: No data loaded")
