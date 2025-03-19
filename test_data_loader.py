# test_data_loader.py
import logging
from pie.data_loader import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Test loading specific modalities
data = DataLoader.load(
    data_path="./PPMI",
    modalities=[
        DataLoader.SUBJECT_CHARACTERISTICS,
        DataLoader.MEDICAL_HISTORY
    ]
)

# Print summary of loaded data
print("\n=== Data Loading Test Results ===")

# Check subject characteristics
if not data[DataLoader.SUBJECT_CHARACTERISTICS].empty:
    print(f"Subject characteristics: {len(data[DataLoader.SUBJECT_CHARACTERISTICS])} rows")
    print("First few rows:")
    print(data[DataLoader.SUBJECT_CHARACTERISTICS].head(3))
else:
    print("Subject characteristics: No data loaded")

# Check medical history
if data[DataLoader.MEDICAL_HISTORY]:
    print(f"Medical history tables: {len(data[DataLoader.MEDICAL_HISTORY])} tables")
    for table_name, df in data[DataLoader.MEDICAL_HISTORY].items():
        print(f"  - {table_name}: {len(df)} rows")
    
    # Show a sample from the first table
    if data[DataLoader.MEDICAL_HISTORY]:
        first_table = next(iter(data[DataLoader.MEDICAL_HISTORY].values()))
        print("\nSample from first medical history table:")
        print(first_table.head(3))
else:
    print("Medical history: No data loaded")