import argparse
import pandas as pd
import vman3 as vman
import csv  # Needed for quoting options
from importlib.resources import files

def test_functions():
    # Sample data
    data_path = files("vman3.data").joinpath("sample_data2.csv")
    data = pd.read_csv(data_path)

    # Test change_null_toskipped
    cleaned_data = vman.clean(data, verbose=True)
    cleaned_data.to_csv(files("vman3.data").joinpath("cleaned_data.csv"),index=False,encoding='utf-8',quoting=csv.QUOTE_NONNUMERIC)

    # Print summary of changes
    print(f"Input data dimensions: {data.shape[0]} rows × {data.shape[1]} columns")
    print(f"Output data dimensions: {cleaned_data.shape[0]} rows × {cleaned_data.shape[1]} columns")
    print(f"Number of NULLs before cleaning: {data.isna().sum().sum():,}")
    print(f"Number of NULLs after cleaning: {cleaned_data.isna().sum().sum():,}")
    print(f"Number of columns with 'Skipped': {(cleaned_data == 'skipped').sum().sum():,}")

def transform_data(va_data:str):
    """
        Transform VA data to InterVA format based on detected WHO VA questionnaire version.
    """
    print(f"""Loading : {va_data}""")
    data = pd.read_csv(va_data, low_memory=False)
    version = vman.detectwhoqn(data)
    print(f"Detected WHO VA questionnaire version: {version}")

    if version == "who2016":
        input_format = "2016WHOv151"
        output_format = "InterVA5"
    elif version == "who2022":
        input_format = "2022WHOv0101"
        output_format = "InterVA_2022"
    else:
        print("Unknown data format. Terminating function.")
        return None
    
    ccva_df = vman.pycrossva(
        input_data=data,
        input_format=input_format,
        output_format=output_format,
        raw_data_id="instanceID",
        lower=True,
        verbose=3
    )
    return ccva_df
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='vman3 test unit')
    parser.add_argument("--input", type=str, required=False, help="Path to VA data CSV file")
    parser.add_argument("--model", required=True,type=int, default=1, help="select the model, 1. InterVA5, 2.InterVA6, 3.ML")
    args = parser.parse_args()
    # # interva5
    # va_data = files("vman3.data").joinpath("whova2016.csv")
    # ivout = run_interva5(va_data)
    # #top10 = ivout.get_csmf(top=10)
    # #print(top10)

    # python3 -m tests.test --model 1
    if args.model == 1: # interva5
        va_data = files("vman3.data").joinpath("whova2016.csv")
        ccva_data = transform_data(va_data)
        #ccva_data.to_csv(files("vman3.data").joinpath("ccva_2016.csv"),index=False,encoding='utf-8',quoting=csv.QUOTE_NONNUMERIC)
        itva5out = vman.interva5(ccva_data)
        va_cod = pd.DataFrame(itva5out['COD'])
        va_causetext = itva5out['CauseText']
        #print(va_cod.iloc[:, :10].head())
        #print(va_cod[['ID','CAUSE1','LIK1']].head())
        #va_cod.to_csv(files("vman3.data").joinpath("iv5_cod.csv"),index=False,encoding='utf-8',quoting=csv.QUOTE_NONNUMERIC)
        print(va_causetext)
        vman.get_csmf(va_cod, top=10)
    
    # python3 -m tests.test --model 2
    if args.model == 2: # interva6
        print("test 2")
        va_data = files("vman3.data").joinpath("whova2022.csv")
        ccva_data = transform_data(va_data)
        #ccva_data.to_csv(files("vman3.data").joinpath("ccva_2022.csv"),index=False,encoding='utf-8',quoting=csv.QUOTE_NONNUMERIC)
        itva6out = vman.interva6(ccva_data)
        va_causetext = itva6out['CauseText']
        va_cod = pd.DataFrame(itva6out['COD'])
        #print(va_cod.iloc[:, :10].head())
        #print(va_cod[['ID','CAUSE1','LIK1']].head())
        #va_cod.to_csv(files("vman3.data").joinpath("iv6_cod.csv"),index=False,encoding='utf-8',quoting=csv.QUOTE_NONNUMERIC)
        print(va_causetext)
        vman.get_csmf(va_cod, top=10)

    if args.model == 3: # ml
        print("test 3")

    #python3 -m tests.test --model 4 --input whova2022.csv
    if args.model == 4: # detect questinnaire version
        va_data = pd.read_csv(files("vman3.data").joinpath(args.input),low_memory = False)
        version = vman.detectwhoqn(va_data)
        print(f"Detected WHO VA questionnaire version: {version}")