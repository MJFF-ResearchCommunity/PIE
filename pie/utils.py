import pandas as pd
import sys


def convert_xlsx_to_arff(xlsx_file, arff_file):
    # Read the first sheet of the Excel file into a DataFrame
    df = pd.read_excel(xlsx_file, sheet_name=0)
    
    with open(arff_file, 'w', encoding='utf-8') as f:
        # Use the file name (without extension) as the relation name
        relation_name = xlsx_file.split('.')[0]
        f.write(f"@RELATION {relation_name}\n\n")
        
        # Write attribute declarations based on column types
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                f.write(f"@ATTRIBUTE {col} NUMERIC\n")
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                # The DATE attribute uses a date format. Adjust the format if necessary.
                f.write(f'@ATTRIBUTE {col} DATE "yyyy-MM-dd\'T\'HH:mm:ss"\n')
            else:
                f.write(f"@ATTRIBUTE {col} STRING\n")
                
        f.write("\n@DATA\n")
        
        # Write the data rows
        for _, row in df.iterrows():
            values = []
            for col in df.columns:
                value = row[col]
                if pd.isnull(value):
                    # Missing values are represented by ?
                    values.append("?")
                else:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        values.append(str(value))
                    elif pd.api.types.is_datetime64_any_dtype(df[col]):
                        # Format datetime values into a string that matches the DATE format
                        values.append(f'"{value.strftime("%Y-%m-%dT%H:%M:%S")}"')
                    else:
                        # For strings, escape any quotes and enclose the value in double quotes
                        value_str = str(value).replace('"', '\\"')
                        values.append(f'"{value_str}"')
            f.write(",".join(values) + "\n")