import pandas as pd
import re

def extract_first_numeric(val):
    if pd.isnull(val):
        return None
    match = re.search(r"\d+\.?\d*", str(val))
    return float(match.group()) if match else None

def clean_sheet(df, sheet_name):
    # List of columns to clean; some may not exist in every sheet.
    columns_to_clean = ['Temperature', 'Humidity', 'Wind Speed', 'Barometer', 'Dewpoint', 'Visibility', 'Wind Chill']
    for col in columns_to_clean:
        if col in df.columns:
            df[col] = df[col].apply(extract_first_numeric)
        else:
            print(f"[!] Column '{col}' not found in sheet '{sheet_name}'. Skipping.")
    
    # Add a column for the state (sheet name)
    df["State"] = sheet_name
    return df

def clean_state_data(input_file="data/chicago_weather_output.xlsx", output_file="data/cleaned_state_data.csv"):
    # Read all sheets (each representing a state) from the Excel file
    sheets = pd.read_excel(input_file, sheet_name=None)
    
    cleaned_dfs = []
    for sheet_name, df in sheets.items():
        print(f"Processing sheet: {sheet_name}")
        df_cleaned = clean_sheet(df, sheet_name)
        cleaned_dfs.append(df_cleaned)
        
    # Combine all sheets into a single DataFrame
    combined_df = pd.concat(cleaned_dfs, ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    print(f"[✓] Combined cleaned data saved to {output_file}")

if __name__ == "__main__":
    clean_state_data()