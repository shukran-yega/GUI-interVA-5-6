import re
import pandas as pd
import argparse
import chardet
import numpy as np
from typing import Tuple, Dict
from pycrossva.transform import transform, SUPPORTED_INPUTS, SUPPORTED_OUTPUTS

import os
from importlib.resources import files, as_file
from typing import Optional
def parse_odk_relevance_to_mask(data_df: pd.DataFrame, relevance_expr: str, verbose: bool = False) -> pd.Series:
    # Work with a clean copy of the DataFrame
    eval_df = data_df.copy()
    col_case_mapping = {col.lower(): col for col in eval_df.columns}


    # Convert all columns to string type for safer evaluation
    for col in eval_df.columns:
        if pd.api.types.is_object_dtype(eval_df[col]):
            eval_df[col] = eval_df[col].astype(str)
    
    
    expr = str(relevance_expr).strip()
    

    # Normalize whitespace and handle spaces
    expr = ' '.join(expr.split())
    expr = str(relevance_expr).strip()


    # Handle selected()
    def convert_selected(match):
        var = match.group(1).strip().lower()
        value = match.group(2).strip()
        actual_col = col_case_mapping.get(var, 'False')
        return f"({actual_col} == '{value}')"

    expr = re.sub(
        r"selected\(\s*\{?([^}]+)\}?\s*,\s*'([^']+)'\s*\)", 
        convert_selected, 
        expr, 
        flags=re.IGNORECASE
    )
    
    if 'selected' in expr or 'True' in expr:
        print(f"Warning: Raw boolean in expression: {expr}")

    # Replace logical operators and comparison symbols
    expr = (expr
            .replace(" and ", " & ")
            .replace(" or ", " | ")
            .replace("not(", "~(")
            .replace("! =", " != ")
            .replace(">==", " >= ")
            .replace("<==", " <= ")
            .replace("?=", "==")
            )

    # Handle string-length()
    expr = re.sub(r'string-length\(\s*([^)]+)\s*\)\s*==\s*0', r'(\1 == "")', expr)
    expr = re.sub(r'string-length\(\s*([^)]+)\s*\)\s*>=\s*1', r'(\1 != "")', expr)

    # Validate parentheses
    if expr.count("(") != expr.count(")"):
        if verbose:
            print(f"Parentheses mismatch in expression: {expr}")
        return pd.Series(False, index=eval_df.index)

    try:
        return eval_df.eval(expr, engine='python', local_dict={col: eval_df[col] for col in eval_df.columns})
    except Exception as e:
        print(f"Error evaluating expression: {expr} \nThe error is: {str(e)}")
        return pd.Series(False, index=eval_df.index)

def clean(
        data_df: pd.DataFrame, 
        dictionary_df: Optional[pd.DataFrame] = None, 
        verbose: bool = False
) -> pd.DataFrame:
    """
    Populates NaN/NULL with 'skipped' in variables that were conditionally not shown.

    Parameters:
    - data_df: DataFrame containing the raw VA data

    Optional Parameters:
    - dictionary_df: DataFrame containing the xForm-style dictionary
    - verbose: Print debugging info if True

    Returns:
    - Updated DataFrame with 'skipped' in hidden-but-null fields
    """

    # drop all colums with sufix _check. These do not provide any relevant informaton
    data_df = data_df.drop(columns=[col for col in data_df.columns if "_check" in col.lower()])

    if verbose:
        print(f"Number of NULLs before cleaning {data_df.isna().sum().sum():,}")  
    # Load default dictionary if none provided
    if dictionary_df is None:
        try:
            # Using importlib.resources for modern Python package resource handling
            from importlib.resources import files, as_file
            ref = files('vman3.data').joinpath('dictionary.csv')
            with as_file(ref) as dict_path:
                dictionary_df = pd.read_csv(dict_path)
                if verbose:
                    print("Loaded default dictionary from package data")
        except Exception as e:
            raise ValueError("Could not load default dictionary from package") from e


    # Preprocess dictionary - remove problematic rows
    dictionary_df = dictionary_df[
        (dictionary_df['relevant'].notna()) & 
        (~dictionary_df['relevant'].str.contains('\?', na=False))  # Filter expressions with ?
    ].copy()

    # Create case mapping for DataFrame columns
    col_case_mapping = {col.lower(): col for col in data_df.columns}
    
    # Clean column names in dictionary and also remove '_check' suffix
    dictionary_df['name'] = dictionary_df['name'].str.lower().str.strip()
    dictionary_df = dictionary_df[~dictionary_df['name'].str.contains('_check', case=False, na=False)]
    
    
    # Only apply logic to select_one/text questions with a relevance rule
    target_vars = dictionary_df[
        dictionary_df['type'].str.contains('select_one|text', na=False) &
        dictionary_df['relevant'].notna()
    ][['name', 'relevant']]
    
    for _, row in target_vars.iterrows():
        dict_var_name = row['name'].strip()
        relevance = str(row['relevant']).strip()
        
        # Find matching column (case-insensitive)
        df_var_name = col_case_mapping.get(dict_var_name.lower())
            
        if df_var_name is None:
            if verbose:
                print(f"Skipping {dict_var_name}: not found in dataset.")
            continue

        try:
            # Get the mask for when the question should be shown
            should_show = parse_odk_relevance_to_mask(data_df, relevance, verbose=verbose)
        
            # if verbose and isinstance(should_show, pd.Series):
            #     print(f"[DEBUG] Number of rows to be updated:\n{should_show.value_counts()}")
            
            if isinstance(should_show, pd.Series):
                # Mark as 'skipped' when:
                # 1. The question should NOT be shown (not should_show)
                # 2. The value is currently null/NA
                mask = (~should_show) & (
                    data_df[df_var_name].isna() | 
                    (data_df[df_var_name].astype(str).str.strip().str.upper().isin(["NULL", "NA", ""]))
                )
                # Convert column to object/string type before assigning 'skipped'
                if mask.any():  # Only convert if there are values to replace
                    if pd.api.types.is_numeric_dtype(data_df[df_var_name]):
                        data_df[df_var_name] = data_df[df_var_name].astype(object)
                    data_df.loc[mask, df_var_name] = 'skipped'

                # if verbose:
                #     print(f"[DEBUG] Processed {dict_var_name} (matched to {df_var_name}): set {mask.sum()} values to 'skipped'")
        except Exception as e:
            if verbose:
                print(f"Error processing '{dict_var_name}' with relevance '{relevance}': {str(e)}")

    # Create case-insensitive column name mapping
    col_case_mapping = {col.lower(): col for col in data_df.columns}

    # Get the actual column names with case preserved
    age_col = col_case_mapping.get('ageinyears')
    age_col2 = col_case_mapping.get('ageinyears2')
    neonatal_col = col_case_mapping.get('isneonatal')
    age_adult_col = col_case_mapping.get('age_adult')  # Note: underscore remains important

    if age_col and age_col2 in data_df.columns:
        data_df[age_col] = data_df[age_col].fillna(data_df[age_col2])
        if verbose:
            print(f"Updated {age_col} with values from {age_col2}")

    if age_col and neonatal_col in data_df.columns:
        data_df.loc[data_df[age_col].isna() & (data_df[neonatal_col] == 1), age_col] = 0
        if verbose:
            print(f"Set {age_col} to 0 for neonatal cases")

    if age_col and age_adult_col in data_df.columns:
        data_df[age_col] = data_df[age_col].fillna(
            data_df[age_adult_col].where(
                (data_df[age_adult_col].notna()) & 
                (data_df[age_adult_col] != 999) & 
                (data_df[age_adult_col] <= 120)
            )
        )
        if verbose:
            print(f"Updated {age_col} adults if NULL with valid values from {age_adult_col}")

    if verbose:
        print(f"Number of NULLs after cleaning {data_df.isna().sum().sum():,}")  
        print("\n[DEBUG check_input] Processing Complete")

    return data_df

def pycrossva(
    input_data,
    input_format: str,
    output_format: str,
    raw_data_id: str = None,
    lower: bool = False,
    verbose: int = 2
) -> pd.DataFrame:
    """
    Converts VA data from a CSV file to the desired CCVA structure using pyCrossVA.
    Orignal source: https://github.com/verbal-autopsy-software/pyCrossVA

    Parameters:
    - input_csv (str): Path to the input CSV file containing VA data.
    - input_format (str): Source format (must be in SUPPORTED_INPUTS).
    - output_format (str): Target format (must be in SUPPORTED_OUTPUTS).
    - raw_data_id (str, optional): Column name for unique record ID.
    - lower (bool, optional): Convert column names to lowercase.
    - verbose (int, optional): Verbosity level (0-5).

    Returns:
    - pd.DataFrame: Transformed CCVA data.

    Raises:
    - ValueError: If input/output formats are not supported or file cannot be read.
    """
    # Validate formats
    if input_format not in SUPPORTED_INPUTS:
        raise ValueError(f"Input format '{input_format}' not supported. Choose from: {SUPPORTED_INPUTS}")
    if output_format not in SUPPORTED_OUTPUTS:
        raise ValueError(f"Output format '{output_format}' not supported. Choose from: {SUPPORTED_OUTPUTS}")

    # Accept either CSV path or DataFrame
    if isinstance(input_data, str):
        try:
            va_data = pd.read_csv(input_data)
        except Exception as e:
            raise ValueError(f"Could not read input CSV file: {e}")
    elif isinstance(input_data, pd.DataFrame):
        va_data = input_data.copy()
    else:
        raise TypeError("input_data must be a file path (str) or a pandas DataFrame.")

    # Transform using pyCrossVA
    ccva_data = transform(
        (input_format, output_format),
        va_data,
        raw_data_id=raw_data_id,
        lower=lower,
        verbose=verbose
    )
    return ccva_data

def interva5(
    va_data: pd.DataFrame,
    hiv: str = "h",
    malaria: str = "h",
    write: bool = True,
    directory: str = "VA_output",
    filename: str = "VA5_result",
    output: str = "extended",
    append: bool = False,
    return_checked_data: bool = True
):
    """
    Run InterVA5 algorithm on VA data.

    Parameters:
    - va_data (pd.DataFrame): Input VA data in ccva format
    - hiv (str): HIV prevalence ('h', 'l', etc.).
    - malaria (str): Malaria prevalence ('h', 'l', etc.).
    - write (bool): Whether to write output files.
    - directory (str): Output directory.
    - filename (str): Output file name.
    - output (str): Output type ('extended', etc.).
    - append (bool): Append to output file.
    - return_checked_data (bool): Return checked data.

    Returns:
    - Output from InterVA5.
    """
    from interva.interva5 import InterVA5
    iva5set = InterVA5(
        va_data,
        hiv=hiv,
        malaria=malaria,
        write=write,
        directory=directory,
        filename=filename,
        output=output,
        append=append,
        return_checked_data=return_checked_data
    )
    iv5out = iva5set.run()
    return iv5out

def interva6(
    va_data: pd.DataFrame,
    hiv: str = "h",
    malaria: str = "h",
    covid: str = "v",
    write: bool = True,
    directory: str = "VA_output",
    filename: str = "VA6_result",
    output: str = "extended"
    
):
    from interva.interva6 import interva6
    itva6 = interva6() 
    iv6out = itva6.run(
        va_data,
        hiv=hiv,
        malaria=malaria,
        covid=covid,
        write=write,
        directory=directory,
        filename=filename,
        output=output
    )
    #print(f"Analysis completed. Processed {iv6out['settings']['num_processed']} records.")
    #print(f"Successfully analyzed {iv6out['settings']['num_successful']} records.")
    return iv6out

def detectwhoqn(vadata):
    """
    Detect which version of WHO VA questionnaire the data comes from.
    Returns 'who2016', 'who2022', or 'unknown' if cannot determine.
    """
    # Read the data (assuming first row is header)
    if isinstance(vadata, str):
        df = pd.read_csv(vadata)
    else:
        df = vadata
    
    # Get all column names
    columns = set(df.columns)
    
    # Define key signature variables for each version
    # These are variables that are unique to each version or have different patterns
    
    # 2016 signature variables
    who2016_signatures = {
        'Id10077note',    # Unique to 2016
        'Id10077_note',   # Unique to 2016
        'Id10444',        # Unique to 2016
        'Id10445',        # Unique to 2016
        'Id10446',        # Unique to 2016
        'Id10450',        # Unique to 2016
        'Id10451',        # Unique to 2016
        'Id10453',        # Unique to 2016
        'Id10454',        # Unique to 2016
        'Id10260_check2', # Unique to 2016
        'id1036X_check',  # Unique to 2016
    }
    
    # 2022 signature variables
    who2022_signatures = {
        'language',       # New in 2022
        'Id10476_audio',  # New in 2022
        'noteon',         # New in 2022
        'notenarr',       # New in 2022
        'note_s_s',       # New in 2022
        'nmh',            # New in 2022
        'botecrn',        # New in 2022
        'noteccd',        # New in 2022
        'noteend',        # New in 2022
    }
    
    # Count matches for each version
    count_2016 = len(columns.intersection(who2016_signatures))
    count_2022 = len(columns.intersection(who2022_signatures))
    
    
    
    # Make decision based on signature matches
    if count_2016 > count_2022 and count_2016 >= 3:
        return 'who2016'
    elif count_2022 > count_2016 and count_2022 >= 3:
        return 'who2022'
    else:
        # If still unclear, use some key discriminators
        if 'Id10077_note' in columns:
            return 'who2016'
        elif 'language' in columns:
            return 'who2022'
        else:
            return 'unknown'

def get_csmf(va_cod:pd.DataFrame, top: int = 10,groupcode: bool = False):
    
    # Normalize column names to lowercase (or uppercase) before processing
    va_cod.columns = va_cod.columns.str.lower()

    # Extract data
    ids = []
    prob_data = []
    
    for i in range(len(va_cod)):
        row = va_cod.iloc[i]
        if 'wholeprob' in row.index and row['wholeprob'] is not None:
            if 'id' in row.index:
                ids.append(row['id'])
            else:
                ids.append(i)
            prob_data.append(row['wholeprob'])
    if ids and prob_data:
        cod_df = pd.DataFrame(prob_data, index=ids)
        print(cod_df.head())

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--verbose", type=bool, required=False, help="Print output to terminal")
    args = parser.parse_args()
    
    with open(args.input, 'rb') as file:
        result = chardet.detect(file.read())
        encoding = result['encoding']

    # Read file with detected encoding
    print("\n Reading the input file")
    df = pd.read_csv(args.input,encoding = encoding,low_memory = False)
    clean(df, verbose=args.verbose)