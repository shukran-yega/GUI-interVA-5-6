import pandas as pd
import numpy as np
import os
from typing import Union, List, Dict, Any, Optional
import warnings
from importlib.resources import files

class InterVA6:
    """
    Python implementation of InterVA2022 algorithm for verbal autopsy analysis.
    
    This class implements the InterVA algorithm specifically for data
    collected using the 2022 WHO verbal autopsy instrument.
    """
    
    def __init__(self, input_data: Union[str, pd.DataFrame], 
            hiv: str = "h", malaria: str = "h", covid: str = "h",
            write: bool = True, directory: Optional[str] = None,
            filename: str = "VA2022_result", output: str = "classic",
            append: bool = False, groupcode: bool = False):
        
        self.input_data = input_data
        self.hiv = hiv
        self.malaria = malaria
        self.covid = covid
        self.write = write
        self.directory = directory
        self.filename = filename
        self.output = output
        self.append = append
        self.groupcode = groupcode
        # Initialize probbase and causetext (would be loaded from data files)
        self.probbase = None
        self.causetext = None
        self.valabels = None
        
    def _load_probbase(self):
        """Load the probability base matrix"""
        data_path = files("interva.data").joinpath("probbase2022.csv")
        probbase = pd.read_csv(data_path)
        return probbase
    
    def _load_causetext(self):
        """Load cause text descriptions"""
        data_path = files("interva.data").joinpath("causetext2022.csv")
        causetext = pd.read_csv(data_path)
        return causetext
    
    def _recode_probbase(self, probbase: pd.DataFrame) -> pd.DataFrame:
        """Recode probability base values from letters to numeric probabilities"""
        probbase = probbase.copy()
        
        # Recode prior probabilities (row 1, columns 4+)
        prior_mapping = {
            'I': 1.0, 'A+': 0.8, 'A': 0.5, 'A-': 0.2,
            'B+': 0.1, 'B': 0.05, 'B-': 0.02, 'C+': 0.01,
            'C': 0.005, 'C-': 0.002, 'D+': 0.001, 'D': 0.0005,
            'D-': 0.0001, 'E': 0.00001, 'N': 0, '': 0
        }
        
        # Recode pregnancy indicators (all rows, columns 4-6)
        preg_mapping = {
            'I': 1.0, 'A+': 0.8, 'A': 0.5, 'A-': 0.2,
            'B+': 0.1, 'B': 0.05, 'B-': 0.02, 'C+': 0.01,
            'C': 0.005, 'C-': 0.002, 'D+': 0.001, 'D': 0.0005,
            'D-': 0.0001, 'E': 0.00001, 'N': 0, '': 0
        }
        
        # Recode symptom probabilities (rows 2+, columns 7+)
        symptom_mapping = {
            'I': 1.0, 'A': 0.8, 'B': 0.5, 'C': 0.1,
            'D': 0.01, 'E': 0.001, 'F': 0.0001, 'G': 0.00001,
            'H': 0.000001, 'N': 0
        }
        
        # Apply mappings
        for col_idx in range(4, len(probbase.columns)):
            probbase.iloc[0, col_idx] = prior_mapping.get(str(probbase.iloc[0, col_idx]), 0)

        for col_idx in range(4, 7):
            for row in range(len(probbase)):
                probbase.iloc[row, col_idx] = preg_mapping.get(str(probbase.iloc[row, col_idx]), 0)
        
        for col_idx in range(7, len(probbase.columns)):
            for row in range(1, len(probbase)):
                probbase.iloc[row, col_idx] = symptom_mapping.get(str(probbase.iloc[row, col_idx]), 0)
        
        probbase.iloc[0, 0:4] = 0  # Set first 4 columns of first row to 0
        
        return probbase
    
    def _adjust_priors(self, probbase: pd.DataFrame, hiv: str, malaria: str, covid: str) -> pd.DataFrame:
        """Adjust prior probabilities based on disease prevalence levels"""
        probbase = probbase.copy()
        
        # HIV adjustments (column 9)
        hiv_mapping = {'h': 0.05, 'l': 0.005, 'v': 0.00001}
        probbase.iloc[0, 9] = hiv_mapping.get(hiv.lower(), 0.005)
        
        # Malaria adjustments (columns 11 and 33)
        malaria_mapping = {
            'h': (0.05, 0.05),
            'l': (0.005, 0.00001),
            'v': (0.00001, 0.00001)
        }
        malaria_vals = malaria_mapping.get(malaria.lower(), (0.005, 0.00001))
        probbase.iloc[0, 11] = malaria_vals[0]  # Column 11
        probbase.iloc[0, 33] = malaria_vals[1]  # Column 33
        
        # COVID adjustments (column 19)
        covid_mapping = {'h': 0.05, 'l': 0.005, 'v': 0.00001}
        probbase.iloc[0, 19] = covid_mapping.get(covid.lower(), 0.005)
        
        return probbase
    
    def _check_input_format(self, input_data: pd.DataFrame):
        """Validate input data format"""
        if len(input_data) < 1:
            raise ValueError("error: no data input")
        
        if len(input_data.columns) != 343:
            raise ValueError("error: invalid data input format. Number of values incorrect")
        
        if input_data.columns[-1].lower() != 'i446o':
            raise ValueError("error: the last variable should be 'i446o'")
    
    def _process_individual(self, row: pd.Series, probbase: pd.DataFrame, 
                        hiv: str, malaria: str, covid: str, causetext: pd.DataFrame):
        """Process an individual VA record (patched: correct row mapping and per-symptom normalization)"""
        # Extract ID and input values
        record_id = str(row.iloc[0])
        input_values = row.values

        # Convert Y/N to 1/0
        input_values = np.where(input_values == 'y', '1', input_values)
        input_values = np.where(input_values == 'n', '0', input_values)
        input_values = np.where(~np.isin(input_values, ['0', '1']), np.nan, input_values)

        # Convert to numeric
        input_numeric = pd.to_numeric(input_values, errors='coerce')
        input_numeric[0] = 0  # Set ID to 0

        # Check for completeness
        if np.nansum(input_numeric[4:11]) < 1:
            return None, f"{record_id} Error in age indicator: Not Specified"

        if np.sum(np.isnan(input_numeric[2:4])) == 2:
            return None, f"{record_id} Error in sex indicator: Not Specified"

        if np.nansum(input_numeric[18:343]) < 1:
            return None, f"{record_id} Error in indicators: No symptoms specified"

        # Replace NaN with 0 and create binary input
        input_numeric = np.nan_to_num(input_numeric, nan=0)
        binary_input = (input_numeric == 1).astype(int)

        # Check reproductive age
        reproductive_age = 0
        if (binary_input[2] == 1 or binary_input[3] == 1) and \
        (binary_input[15] == 1 or binary_input[16] == 1 or binary_input[17] == 1):
            reproductive_age = 1

        # Prepare system prior vector (Sys_Prior equivalent)
        sys_prior = probbase.iloc[0, 4:].values.astype(float)
        prob = sys_prior.copy()

        # Number of pregnancy entries and total prob length (keeps slices dynamic)
        num_preg = 3
        total_prob_len = prob.shape[0]
        # In R: prob_A = prob[1:3]; prob_B = prob[4:67] -> zero-based: prob[:3], prob[3:67]
        # compute end index for causes using available length
        num_causes = total_prob_len - num_preg
        cause_slice_end = num_preg + num_causes  # equals total_prob_len

        # Find symptoms present
        # binary_input[1:] corresponds to Input positions 2..end (zero-based symptom indices)
        symptoms_present = np.where(binary_input[1:] == 1)[0]

        # For each symptom, map to probbase row and multiply then normalize (matching R)
        for symptom_idx in symptoms_present:
            probbase_row = symptom_idx + 1  # integer index into probbase (zero-based iloc)
            if probbase_row < len(probbase):
                # columns 4: in R are 5:pb_ncol; zero-based iloc uses start index 4
                symptom_probs = probbase.iloc[probbase_row, 4:].values.astype(float)
                prob *= symptom_probs

                # normalize pregnancy block and cause block after each symptom (R behaviour)
                if np.sum(prob[:num_preg]) > 0:
                    prob[:num_preg] = prob[:num_preg] / np.sum(prob[:num_preg])
                if np.sum(prob[num_preg:cause_slice_end]) > 0:
                    prob[num_preg:cause_slice_end] = prob[num_preg:cause_slice_end] / np.sum(prob[num_preg:cause_slice_end])

        # Final split
        prob_a = prob[:num_preg]                  # pregnancy probabilities (length 3)
        prob_b = prob[num_preg:cause_slice_end]   # cause probabilities

        # Determine pregnancy status
        preg_state = "n/a"
        lik_preg = " "

        if reproductive_age == 1:
            if np.sum(prob_a) == 0:
                preg_state = "indeterminate"
            else:
                max_prob_idx = np.argmax(prob_a)
                max_prob_val = prob_a[max_prob_idx]

                if max_prob_val >= 0.1:
                    preg_state_options = [
                        "Not pregnant or recently delivered",
                        "Pregnancy ended within 6 weeks of death",
                        "Pregnant at death"
                    ]
                    preg_state = preg_state_options[max_prob_idx]
                    lik_preg = round((max_prob_val / np.sum(prob_a)) * 100)

        # Determine causes of death using patched function below
        cause1, lik1, cause2, lik2, cause3, lik3, indet = self._determine_causes(prob_b, causetext)

        # Prepare result
        result = {
            'ID': record_id,
            'MALPREV': malaria.upper(),
            'HIVPREV': hiv.upper(),
            'COVIDPREV': covid.upper(),
            'PREGSTAT': preg_state,
            'PREGLIK': lik_preg,
            'CAUSE1': cause1,
            'LIK1': lik1,
            'CAUSE2': cause2,
            'LIK2': lik2,
            'CAUSE3': cause3,
            'LIK3': lik3,
            'INDET': indet,
            'wholeprob': np.concatenate([prob_a, prob_b])
        }

        return result, None

    
    def _determine_causes(self, prob_b: np.ndarray, causetext: pd.DataFrame):
        """Determine top causes from probability distribution (patched to match R thresholds and formatting)"""
        # If maximum cause probability is less than 0.4 -> blank causes and indet 100 (R behavior)
        max_prob_B = np.max(prob_b) if prob_b.size > 0 else 0.0
        if max_prob_B < 0.4:
            return " ", " ", " ", " ", " ", " ", 100

        # Get sorted indices (descending) and top three
        sorted_indices = np.argsort(prob_b)[::-1]
        top_indices = sorted_indices[:3]
        top_probs = prob_b[top_indices]
        top_causes = causetext.iloc[top_indices]['description'].values
        #top_causes = causetext.iloc[top_indices, 1].values


        # Primary cause
        lik1 = round(top_probs[0] * 100)
        cause1 = top_causes[0]

        # Secondary and tertiary only if >= 0.5 * max(orig_prob_B)
        lik2 = round(top_probs[1] * 100) if top_probs.size > 1 and top_probs[1] >= 0.5 * max_prob_B else " "
        cause2 = top_causes[1] if top_probs.size > 1 and top_probs[1] >= 0.5 * max_prob_B else " "

        lik3 = round(top_probs[2] * 100) if top_probs.size > 2 and top_probs[2] >= 0.5 * max_prob_B else " "
        cause3 = top_causes[2] if top_probs.size > 2 and top_probs[2] >= 0.5 * max_prob_B else " "

        # indeterminacy: 100 - sum of numeric likelihoods (ignore blanks)
        numeric_liks = [x for x in (lik1, lik2, lik3) if isinstance(x, (int, float, np.integer, np.floating))]
        indet = round(100 - sum(numeric_liks))

        return cause1, lik1, cause2, lik2, cause3, lik3, indet

    
    def _save_results(self, results: List[Dict], filename: str, output: str, causetext: pd.DataFrame):
        """Save results to CSV file"""
        if output == "classic":
            columns = ['ID', 'MALPREV', 'HIVPREV', 'COVIDPREV', 'PREGSTAT', 
                      'PREGLIK', 'CAUSE1', 'LIK1', 'CAUSE2', 'LIK2', 
                      'CAUSE3', 'LIK3', 'INDET']
        else:  # extended
            columns = ['ID', 'MALPREV', 'HIVPREV', 'COVIDPREV', 'PREGSTAT', 
                      'PREGLIK', 'CAUSE1', 'LIK1', 'CAUSE2', 'LIK2', 
                      'CAUSE3', 'LIK3', 'INDET'] + list(causetext['description'])
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(filename, index=False)
    
    def run(self)-> None:
        """
        Main function to perform InterVA6 analysis.
        
        Parameters:
        input_data: Path to CSV file or DataFrame with VA data
        hiv: HIV prevalence level ("h", "l", or "v")
        malaria: Malaria prevalence level ("h", "l", or "v")
        covid: COVID prevalence level ("h", "l", or "v")
        write: Whether to save output to file
        directory: Directory to save output
        filename: Output filename
        output: Output format ("classic" or "extended")
        append: Whether to append to existing file
        groupcode: Whether to include group codes
        
        Returns:
        Dictionary with analysis results
        """
        
        # Validate input parameters
        if  self.hiv.lower() not in ['h', 'l', 'v'] or \
            self.malaria.lower() not in ['h', 'l', 'v'] or \
            self.covid.lower() not in ['h', 'l', 'v']:
            raise ValueError("HIV, Malaria, and Covid indicators should be 'h', 'l', or 'v'")
        
        if self.write and self.directory is None:
            raise ValueError("Please provide a directory when write=True")
        
        # Load data
        if isinstance(self.input_data, str):
            input_df = pd.read_csv(self.input_data)
        else:
            input_df = self.input_data.copy()
        
        # Load probability base and cause text
        self.probbase = self._load_probbase()
        self.causetext = self._load_causetext()

         ## checking 
        # after loading:
        print("probbase.shape:", self.probbase.shape)  # expect (343, 71)
        # print("probbase columns sample:", list(self.probbase.columns[:10]))
        # print("first probbase row (priors) slice len:", len(self.probbase.iloc[0,4:]))
        # print("first 6 prior values:", self.probbase.iloc[0,4:10].values)

        # check causetext
        print("causetext.shape:", self.causetext.shape)      # expect (67, something)
        # print("causetext columns:", list(self.causetext.columns))
        # print("first 6 causetext descriptions:", self.causetext['description'].head(6).tolist())

        # self.probbase = self.probbase.loc[:, ~self.probbase.columns.str.contains('^Unnamed')]
        # self.causetext = self.causetext.loc[:, ~self.causetext.columns.str.contains('^Unnamed')]

        
        # Recode probbase
        self.probbase = self._recode_probbase(self.probbase)
        
        # Adjust priors based on disease prevalence
        self.probbase = self._adjust_priors(self.probbase, self.hiv, self.malaria, self.covid)
        
        # Check input format
        self._check_input_format(input_df)
        
        # Process each record
        results = []
        errors = []
        total = len(input_df)
        progress_marks = set([int(total * i / 10) for i in range(1, 11)])  # 10%, 20%, ..., 100%
        
        for idx, row in input_df.iterrows():
            result, error = self._process_individual(row, self.probbase, self.hiv, self.malaria, self.covid, self.causetext)
            
            if result:
                results.append(result)
            if error:
                errors.append(error)
                
            if (idx + 1) in progress_marks:
                percent = int((idx + 1) / total * 100)
                print(f"Progress: {percent}%")

        # Save results if requested
        if self.write and self.directory:
            os.makedirs(self.directory, exist_ok=True)
            output_path = os.path.join(self.directory, f"{self.filename}.csv")
            self._save_results(results, output_path, self.output, self.causetext)
        
        # Save errors to log file
        if self.write and errors:
            log_path = os.path.join(self.directory, "errorlog2022.txt")
            with open(log_path, 'w') as f:
                f.write("Error & warning log built for InterVA6\n")
                for error in errors:
                    f.write(f"{error}\n")
        
        return {
            'results': results,
            'errors': errors,
            'settings': {
                'hiv': self.hiv,
                'malaria': self.malaria,
                'covid': self.covid,
                'num_processed': len(input_df),
                'num_successful': len(results)
            }
        }

# Example usage
if __name__ == "__main__":
    # Initialize the analyzer
    interva6 = InterVA6()
    
    # Example analysis
    try:
        results = interva6.run(
            input_data="va_data.csv",
            hiv="h",
            malaria="l",
            covid="v",
            write=True,
            directory="./results",
            filename="va_analysis",
            output="extended"
        )
        print(f"Analysis completed. Processed {results['settings']['num_processed']} records.")
        print(f"Successfully analyzed {results['settings']['num_successful']} records.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")