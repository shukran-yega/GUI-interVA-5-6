#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
InterVA2022: Python Implementation matching R InterVA2022 package
This implementation closely follows the R script logic with proper indexing adjustments
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Union, Dict, List, Optional, Tuple, Any
import csv
import datetime
import os
import sys
import warnings
from IPython.display import display

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, desc="Processing", **kwargs):
        return iterable


class InterVA2022:
    """
    Python implementation of InterVA2022 algorithm matching the R package behavior.
    """
    
    def __init__(self):
        self.results = {}
        self.errors = []
        self.probbase_version = None
        self.probbase_matrix = None
        self.causetext = None
        self.probbase_raw = None
    
    def _find_data_file(self, filename: str) -> Optional[str]:
        """Search for data files in common locations."""
        search_paths = [
            filename,  # Direct path
            os.path.join(os.path.dirname(__file__), filename),
            os.path.join(os.path.dirname(__file__), "data", filename),
            os.path.join(os.getcwd(), filename),
            os.path.join(os.getcwd(), "data", filename),
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                print(f"Found {filename} at: {path}")
                return path
        return None
    
    def _load_probbase(self, probbase_file: Optional[str] = None) -> pd.DataFrame:
        """Load probability base matrix."""
        if probbase_file is None:
            probbase_file = self._find_data_file("probbase2022.csv")
            if probbase_file is None:
                probbase_file = self._find_data_file("ccva_2022.csv")
        
        if probbase_file is None or not os.path.exists(probbase_file):
            raise FileNotFoundError(
                "Could not find probbase file. Please provide path to probbase2022.csv"
            )
        
        try:
            # Load with appropriate encoding
            try:
                probbase = pd.read_csv(probbase_file)
            except UnicodeDecodeError:
                probbase = pd.read_csv(probbase_file, encoding='latin-1')
            
            print(f"Loaded probbase with shape: {probbase.shape}")
            
            # Store raw probbase for reference
            self.probbase_raw = probbase.copy()
            
            # Extract version from position [0, 2] (R: [1, 3])
            if probbase.shape[0] > 0 and probbase.shape[1] > 2:
                self.probbase_version = str(probbase.iloc[0, 2])
                print(f"Using Probbase version: {self.probbase_version}")
            else:
                self.probbase_version = "Unknown"
            
            return probbase
            
        except Exception as e:
            raise FileNotFoundError(f"Could not load probbase file: {e}")
    
    def _recode_probbase(self, probbase: pd.DataFrame) -> np.ndarray:
        """
        Recode probbase matrix following R script logic exactly.
        R uses 1-based indexing, Python uses 0-based.
        """
        # Convert to numpy array for easier manipulation
        pb_matrix = probbase.values.copy()
        pb_nrow, pb_ncol = pb_matrix.shape
        
        print(f"Recoding probbase matrix of shape: {pb_matrix.shape}")
        
        # Initialize numeric matrix
        numeric_matrix = np.zeros((pb_nrow, pb_ncol), dtype=float)
        
        # Copy the string values first (for reference)
        for i in range(pb_nrow):
            for j in range(pb_ncol):
                if isinstance(pb_matrix[i, j], str):
                    pb_matrix[i, j] = pb_matrix[i, j].strip()
        
        # 1. Recode prior probabilities (row 1, columns 5-71 in R = row 0, columns 4-70 in Python)
        prior_mapping = {
            'I': 1.0, 'A+': 0.8, 'A': 0.5, 'A-': 0.2, 'B+': 0.1,
            'B': 0.05, 'B-': 0.02, 'B -': 0.02, 'C+': 0.01, 'C': 0.005,
            'C-': 0.002, 'D+': 0.001, 'D': 0.0005, 'D-': 0.0001,
            'E': 0.00001, 'N': 0.0, '': 0.0, ' ': 0.0
        }
        
        # Process priors (first row, columns 5+ in R = columns 4+ in Python)
        for j in range(4, min(pb_ncol, 71)):
            val = str(pb_matrix[0, j]).strip() if not pd.isna(pb_matrix[0, j]) else ''
            numeric_matrix[0, j] = prior_mapping.get(val, 0.0)
        
        # 2. Recode pregnancy indicators (all rows, columns 5-7 in R = columns 4-6 in Python)
        for i in range(pb_nrow):
            for j in range(4, min(7, pb_ncol)):
                val = str(pb_matrix[i, j]).strip() if not pd.isna(pb_matrix[i, j]) else ''
                numeric_matrix[i, j] = prior_mapping.get(val, 0.0)
        
        # 3. Recode conditional probabilities Pr(S|C) (rows 2+, columns 8+ in R = rows 1+, columns 7+ in Python)
        conditional_mapping = {
            'I': 1.0, 'A': 0.8, 'B': 0.5, 'C': 0.1, 'D': 0.01,
            'E': 0.001, 'F': 0.0001, 'G': 0.00001, 'H': 0.000001, 
            'N': 0.0, '': 0.0, ' ': 0.0
        }
        
        for i in range(1, pb_nrow):
            for j in range(7, min(pb_ncol, 71)):
                val = str(pb_matrix[i, j]).strip() if not pd.isna(pb_matrix[i, j]) else ''
                numeric_matrix[i, j] = conditional_mapping.get(val, 0.0)
        
        # 4. Set first 4 columns of first row to 0 (as in R script)
        numeric_matrix[0, :4] = 0.0
        
        print("Probbase recoding completed")
        return numeric_matrix
    
    def _load_causetext(self, groupcode: bool = False) -> pd.DataFrame:
        """Load cause text descriptions."""
        causetext_file = self._find_data_file("causetext2022.csv")
        
        if causetext_file and os.path.exists(causetext_file):
            try:
                causetext = pd.read_csv(causetext_file)
                print(f"Loaded causetext with shape: {causetext.shape}")
                
                # Select columns based on groupcode parameter
                # Assuming causetext has columns like: index, description, code
                if groupcode and causetext.shape[1] >= 3:
                    # Remove description column (column 2 in R = column 1 in Python)
                    causetext = causetext.drop(causetext.columns[1], axis=1)
                elif not groupcode and causetext.shape[1] >= 3:
                    # Remove code column (column 3 in R = column 2 in Python)
                    causetext = causetext.drop(causetext.columns[2], axis=1)
                    
            except Exception as e:
                print(f"Warning: Could not load causetext: {e}")
                causetext = self._create_default_causetext()
        else:
            print("Warning: causetext2022.csv not found, using default causes")
            causetext = self._create_default_causetext()
        
        return causetext
    
    def _create_default_causetext(self) -> pd.DataFrame:
        """Create default cause descriptions matching InterVA2022 causes."""
        causes_data = []
        # First 3 are pregnancy-related
        causes_data.extend([
            ['PREG1', 'Not pregnant or recently delivered'],
            ['PREG2', 'Pregnancy ended within 6 weeks of death'],
            ['PREG3', 'Pregnant at death'],
        ])
        
        # Add 64 actual causes (total should be 67)
        default_causes = [
            'HIV/AIDS related death', 'Tuberculosis', 'Malaria', 'COVID-19',
            'Diarrhoeal diseases', 'Meningitis and encephalitis', 'Sepsis',
            'Stroke', 'Ischaemic heart disease', 'COPD', 'Asthma',
            'Diabetes mellitus', 'Road traffic accident', 'Drowning',
            'Falls', 'Poisoning', 'Burns', 'Suicide', 'Homicide'
        ]
        
        for i, cause in enumerate(default_causes):
            causes_data.append([f'Cause{i+4}', cause])
        
        # Fill remaining causes
        for i in range(len(causes_data), 67):
            causes_data.append([f'Cause{i+1}', f'Cause {i+1}'])
        
        return pd.DataFrame(causes_data, columns=['code', 'description'])
    
    def _setup_logging(self, directory: str, write: bool) -> Optional[logging.Logger]:
        """Setup error logging."""
        if not write:
            return None
        
        logger = logging.getLogger('interva2022')
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        log_file = os.path.join(directory, "errorlog2022.txt")
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Error & warning log built for InterVA2022 {timestamp}\n")
        logger.info("\n\nthe following records are incomplete and excluded from further processing:\n\n")
        
        return logger
    
    def _validate_input_data(self, input_data: Union[pd.DataFrame, str]) -> pd.DataFrame:
      """Validate and preprocess input data."""
      if isinstance(input_data, str):
        try:
            input_df = pd.read_csv(input_data)
            print(f"Loaded input data from file with shape: {input_df.shape}")
        except Exception as e:
            raise ValueError(f"Could not load input data: {e}")
      elif isinstance(input_data, pd.DataFrame):
        # Already a DataFrame - just copy it
        input_df = input_data.copy()
        print(f"Using provided DataFrame with shape: {input_df.shape}")
      else:
        raise ValueError("input_data must be either a file path (str) or pandas DataFrame")
    
      if len(input_df) < 1:
        raise ValueError("error: no data input")
    
    # Check last column
      if len(input_df.columns) > 0:
        last_col = input_df.columns[-1]
        if last_col.lower() != "i446o":
            print(f"Warning: Last column is '{last_col}', expected 'i446o'")
            # Don't raise error, just warn
    
    # Check number of columns matches probbase rows
      if self.probbase_matrix is not None:
        if input_df.shape[1] != self.probbase_matrix.shape[0]:
            print(f"Warning: Column count mismatch. Expected {self.probbase_matrix.shape[0]}, got {input_df.shape[1]}")
            # Don't raise error in production - be lenient
    
      return input_df
    


    def _process_record(self, record: pd.Series, record_id: str, 
                       logger: Optional[logging.Logger]) -> Optional[np.ndarray]:
        """Process individual record following R script logic."""
        # Convert to array
        input_current = record.values.copy()
        
        # Replace Y/N with 1/0
        for i in range(len(input_current)):
            val = str(input_current[i]).upper() if not pd.isna(input_current[i]) else ''
            if val == 'Y':
                input_current[i] = 1
            elif val == 'N':
                input_current[i] = 0
            else:
                try:
                    input_current[i] = float(input_current[i])
                    if pd.isna(input_current[i]):
                        input_current[i] = 0
                except:
                    input_current[i] = 0
        
        # Convert to numeric
        input_current = np.array(input_current, dtype=float)
        
        # Set ID (first column) to 0
        input_current[0] = 0
        
        # Validate age indicators (columns 5-11 in R = indices 4-10 in Python)
        if np.sum(input_current[4:11]) < 1:
            if logger:
                logger.info(f"{record_id} Error in age indicator: Not Specified")
            return None
        
        # Validate sex indicators (columns 3-4 in R = indices 2-3 in Python)
        if np.isnan(input_current[2]) and np.isnan(input_current[3]):
            if logger:
                logger.info(f"{record_id} Error in sex indicator: Not Specified")
            return None
        
        # Validate symptoms (columns 19-343 in R = indices 18-342 in Python)
        if np.sum(input_current[18:343]) < 1:
            if logger:
                logger.info(f"{record_id} Error in indicators: No symptoms specified")
            return None
        
        # Create new input array with 0/1 values
        new_input = np.zeros(len(input_current))
        for i in range(1, len(input_current)):  # Skip ID column
            if not np.isnan(input_current[i]) and input_current[i] == 1:
                new_input[i] = 1
        
        return new_input
    
    def _setup_system_priors(self, hiv: str, malaria: str, covid: str) -> np.ndarray:
        """Setup system prior probabilities."""
        if self.probbase_matrix is None:
            raise ValueError("Probbase matrix not loaded")
        
        sys_prior = self.probbase_matrix[0, :].copy()
        
        # Validate inputs
        hiv = hiv.lower()
        malaria = malaria.lower()
        covid = covid.lower()
        
        if hiv not in ['h', 'l', 'v'] or malaria not in ['h', 'l', 'v'] or covid not in ['h', 'l', 'v']:
            raise ValueError("error: the HIV, Malaria, and Covid indicators should be one of the three: 'h', 'l', and 'v'")
        
        # Set HIV prevalence (column 10 in R = index 9 in Python)
        if hiv == 'h':
            sys_prior[9] = 0.05
        elif hiv == 'l':
            sys_prior[9] = 0.005
        elif hiv == 'v':
            sys_prior[9] = 1e-05
        
        # Set Malaria prevalence (columns 12 and 34 in R = indices 11 and 33 in Python)
        if malaria == 'h':
            sys_prior[11] = 0.05
            sys_prior[33] = 0.05
        elif malaria == 'l':
            sys_prior[11] = 0.005
            sys_prior[33] = 1e-05
        elif malaria == 'v':
            sys_prior[11] = 1e-05
            sys_prior[33] = 1e-05
        
        # Set Covid prevalence (column 20 in R = index 19 in Python)
        if covid == 'h':
            sys_prior[19] = 0.05
        elif covid == 'l':
            sys_prior[19] = 0.005
        elif covid == 'v':
            sys_prior[19] = 1e-05
        
        return sys_prior
    
    def _calculate_probabilities(self, new_input: np.ndarray, sys_prior: np.ndarray) -> np.ndarray:
        """Calculate Bayesian probabilities following R script logic."""
        D = len(sys_prior)
        
        # Initialize prob with system priors (columns 5 to D in R = indices 4 to D-1 in Python)
        prob = sys_prior[4:D].copy()
        
        # Find which symptoms are present (indices 2 onwards in R = indices 1 onwards in Python)
        temp = np.where(new_input[1:] == 1)[0]
        
        # Bayesian updating for each symptom
        for symptom_idx in temp:
            # Row in probbase (symptom_idx + 1 because of 0-indexing, then +1 for header row)
            probbase_row = symptom_idx + 1
            
            # Update probabilities (columns 5 to D in R = indices 4 to D-1 in Python)
            for j in range(4, D):
                if probbase_row < self.probbase_matrix.shape[0] and j < self.probbase_matrix.shape[1]:
                    prob[j - 4] *= self.probbase_matrix[probbase_row, j]
            
            # Normalize pregnancy probabilities (first 3 causes)
            if np.sum(prob[:3]) > 0:
                prob[:3] = prob[:3] / np.sum(prob[:3])
            
            # Normalize cause probabilities (causes 4-67)
            if np.sum(prob[3:67]) > 0:
                prob[3:67] = prob[3:67] / np.sum(prob[3:67])
        
        return prob
    
    def _determine_pregnancy_status(self, prob: np.ndarray, new_input: np.ndarray) -> Tuple[str, Any]:
        """Determine pregnancy status following R script logic."""
        reproductive_age = 0
        
        # Check reproductive age (columns match R script)
        # Female (columns 3-4 in R = indices 2-3 in Python)
        # Age 12-50 (columns 16-18 in R = indices 15-17 in Python)
        if ((new_input[2] == 1 or new_input[3] == 1) and 
            (new_input[15] == 1 or new_input[16] == 1 or new_input[17] == 1)):
            reproductive_age = 1
        
        prob_A = prob[:3]  # Pregnancy probabilities
        
        if np.sum(prob_A) == 0 or reproductive_age == 0:
            return "n/a", " "
        
        if np.max(prob_A) < 0.1 and reproductive_age == 1:
            return "indeterminate", " "
        
        max_idx = np.argmax(prob_A)
        max_prob = prob_A[max_idx]
        
        if max_idx == 0 and max_prob >= 0.1 and reproductive_age == 1:
            lik_preg = int(round(prob_A[0] / np.sum(prob_A) * 100))
            return "Not pregnant or recently delivered", lik_preg
        
        if max_idx == 1 and max_prob >= 0.1 and reproductive_age == 1:
            lik_preg = int(round(prob_A[1] / np.sum(prob_A) * 100))
            return "Pregnancy ended within 6 weeks of death", lik_preg
        
        if max_idx == 2 and max_prob >= 0.1 and reproductive_age == 1:
            lik_preg = int(round(prob_A[2] / np.sum(prob_A) * 100))
            return "Pregnant at death", lik_preg
        
        return "indeterminate", " "
    
    def _determine_causes(self, prob: np.ndarray) -> Dict[str, Any]:
        """Determine causes of death following R script logic."""
        prob_B = prob[3:67]  # Cause probabilities (excluding pregnancy)
        
        # Initialize results
        cause1 = lik1 = cause2 = lik2 = cause3 = lik3 = " "
        indet = 100
        
        if np.max(prob_B) < 0.4:
            return {
                'cause1': cause1, 'lik1': lik1,
                'cause2': cause2, 'lik2': lik2,
                'cause3': cause3, 'lik3': lik3,
                'indet': indet
            }
        
        # Find top cause
        max_prob = np.max(prob_B)
        max_idx = np.argmax(prob_B)
        lik1 = int(round(max_prob * 100))
        cause1 = self._get_cause_name(max_idx + 3)  # +3 because first 3 are pregnancy
        
        # Find second cause
        prob_temp = prob_B.copy()
        prob_temp[max_idx] = -1  # Mark as used
        
        if np.max(prob_temp) >= 0.5 * max_prob:
            max_idx2 = np.argmax(prob_temp)
            lik2 = int(round(prob_temp[max_idx2] * 100))
            cause2 = self._get_cause_name(max_idx2 + 3)
            
            # Find third cause
            prob_temp[max_idx2] = -1  # Mark as used
            
            if np.max(prob_temp) >= 0.5 * max_prob:
                max_idx3 = np.argmax(prob_temp)
                lik3 = int(round(prob_temp[max_idx3] * 100))
                cause3 = self._get_cause_name(max_idx3 + 3)
        
        # Calculate indeterminate
        top3 = []
        if lik1 != " ":
            top3.append(lik1)
        if lik2 != " ":
            top3.append(lik2)
        if lik3 != " ":
            top3.append(lik3)
        
        indet = int(round(100 - sum(top3)))
        
        return {
            'cause1': cause1, 'lik1': lik1,
            'cause2': cause2, 'lik2': lik2,
            'cause3': cause3, 'lik3': lik3,
            'indet': indet
        }
    
    def _get_cause_name(self, cause_index: int) -> str:
        """Get cause name from causetext."""
        if self.causetext is not None and cause_index < len(self.causetext):
            # Assuming causetext has description in second column
            if 'description' in self.causetext.columns:
                return str(self.causetext.iloc[cause_index]['description'])
            elif len(self.causetext.columns) > 1:
                return str(self.causetext.iloc[cause_index, 1])
        return f"Cause_{cause_index + 1}"
    
    def analyze(self, 
                input_data: Union[pd.DataFrame, str],
                hiv: str = "h", 
                malaria: str = "l", 
                covid: str = "v",
                write: bool = True,
                directory: Optional[str] = None,
                filename: str = "VA2022_result",
                output: str = "classic",
                append: bool = False,
                groupcode: bool = False,
                probbase_file: Optional[str] = None,
                sci: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Main analysis function matching R InterVA2022 function.
        
        Parameters match the R function parameters.
        """
        print("Starting InterVA2022 analysis...")
        
        # Handle directory
        if write and directory is None:
            raise ValueError("error: please provide a directory (required when write = TRUE)")
        
        if directory is None:
            directory = os.getcwd()
        
        # Create directory if needed
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Store original directory and change to working directory
        original_dir = os.getcwd()
        os.chdir(directory)
        
        try:
            # Setup logging
            logger = self._setup_logging(".", write)
            
            # Load probbase (either from sci parameter or default)
            if sci is not None:
                # Validate sci dimensions
                if not isinstance(sci, (pd.DataFrame, np.ndarray)):
                    raise ValueError("error: invalid sci (must be data frame or matrix)")
                if isinstance(sci, pd.DataFrame):
                    if sci.shape != (343, 71):
                        raise ValueError("error: invalid sci (must be data frame or matrix with 343 rows and 71 columns)")
                    self.probbase_raw = sci
                    self.probbase_matrix = self._recode_probbase(sci)
                else:
                    if sci.shape != (343, 71):
                        raise ValueError("error: invalid sci (must be data frame or matrix with 343 rows and 71 columns)")
                    self.probbase_matrix = sci
            else:
                # Load default probbase
                print("Loading probbase...")
                probbase_df = self._load_probbase(probbase_file)
                self.probbase_matrix = self._recode_probbase(probbase_df)
            
            # Load causetext
            print("Loading causetext...")
            self.causetext = self._load_causetext(groupcode)
            
            # Load and validate input data
            print("Loading and validating input data...")
            input_df = self._validate_input_data(input_data)
            
            # Setup system priors
            print("Setting up system priors...")
            sys_prior = self._setup_system_priors(hiv.lower(), malaria.lower(), covid.lower())
            
            # Initialize results
            ID_list = []
            VAresult = []
            rows = []
            
            # Write header if not appending
            if write and not append:
                #output_file = f"{filename}.csv"
                #with open(output_file, 'w', newline='') as f:
                #   writer = csv.writer(f)
                    header = ["ID", "MALPREV", "HIVPREV", "COVIDPREV", "PREGSTAT", "PREGLIK","CAUSE1", "LIK1", "CAUSE2", "LIK2", "CAUSE3", "LIK3", "INDET"]
                    if output == "extended" and self.causetext is not None:
                        # Add cause names to header
                        cause_names = []
                        for i in range(len(self.causetext)):
                            if 'description' in self.causetext.columns:
                                cause_names.append(self.causetext.iloc[i]['description'])
                            elif len(self.causetext.columns) > 1:
                                cause_names.append(self.causetext.iloc[i, 1])
                            else:
                                cause_names.append(f"Cause_{i+1}")
                        header.extend(cause_names[:67])
                   # writer.writerow(header)
            
            # Process each record
            N = len(input_df)
            print(f"Processing {N} records...")
            
            # Progress indicators
            nd = max(1, round(N / 100))
            np_indicator = max(1, round(N / 10))
            
            for i in range(N):
                # Progress display
                if (i + 1) % nd == 0:
                    print(".", end="", flush=True)
                if (i + 1) % np_indicator == 0:
                    print(f" {round((i + 1) / N * 100)}% completed")
                if i == N - 1:
                    print(" 100% completed")
                
                # Get record ID
                record_id = str(input_df.iloc[i, 0])
                
                # Process record
                processed = self._process_record(input_df.iloc[i], record_id, logger)
                
                if processed is None:
                    continue  # Skip invalid record
                
                # Calculate probabilities
                prob = self._calculate_probabilities(processed, sys_prior)
                
                # Determine pregnancy status
                preg_state, lik_preg = self._determine_pregnancy_status(prob, processed)
                
                # Determine causes
                causes = self._determine_causes(prob)
                
                # Store results
                ID_list.append(record_id)
                
                result = {
                    'ID': record_id,
                    'MALPREV': malaria.lower(),
                    'HIVPREV': hiv.lower(),
                    'COVIDPREV': covid.lower(),
                    'PREGSTAT': preg_state,
                    'PREGLIK': lik_preg,
                    'CAUSE1': causes['cause1'],
                    'LIK1': causes['lik1'],
                    'CAUSE2': causes['cause2'],
                    'LIK2': causes['lik2'],
                    'CAUSE3': causes['cause3'],
                    'LIK3': causes['lik3'],
                    'INDET': causes['indet'],
                    'wholeprob': prob.tolist()
                }
                
                VAresult.append(result)
                
                # Write to file if requested
                #if write:
                 #   output_file = f"{filename}.csv"
                  #  with open(output_file, 'a', newline='') as f:
                   #     writer = csv.writer(f)
                        #if output == "classic":
                        #   row = [
                        #      result['ID'], result['MALPREV'], result['HIVPREV'], result['COVIDPREV'],
                        #    result['CAUSE2'], result['LIK2'], result['CAUSE3'], result['LIK3'],
                        #    result['INDET']
                        # ]
                        #else:  
                            # extended
                row = [
                    result['ID'], result['MALPREV'], result['HIVPREV'], result['COVIDPREV'],
                    result['PREGSTAT'], result['PREGLIK'], result['CAUSE1'], result['LIK1'],
                    result['CAUSE2'], result['LIK2'], result['CAUSE3'], result['LIK3'],
                    result['INDET']
                    ] + result['wholeprob'][:67]  # Add probability distribution
                rows.append(row)
                       # writer.writerow(row)
            
            print(f"\nSuccessfully processed {len(ID_list)} out of {N} records")
            
            # Prepare final output
            #self.results = {
            #    'ID': ID_list,
            #    'VA2022': VAresult,
            #   'Malaria': malaria.lower(),
            #    'HIV': hiv.lower(),
            #    'Covid': covid.lower()
            # }
            columns = ["ID", "MALPREV", "HIVPREV", "COVIDPREV", "PREGSTAT", 
                       "PREGLIK", "CAUSE1", "LIK1", "CAUSE2", "LIK2", "CAUSE3", "LIK3", "INDET"]
        
            if output == "extended" and self.causetext is not None:
            # Add cause names to columns (same as CSV header)
             for i in range(len(self.causetext)):
                if 'description' in self.causetext.columns:
                    columns.append(self.causetext.iloc[i]['description'])
                elif len(self.causetext.columns) > 1:
                    columns.append(self.causetext.iloc[i, 1])
                else:
                    columns.append(f"Cause_{i+1}")
            columns = columns[:13 + 67] # safety check to ensure dataframe has 80 columns exactly 
            
            resultdf = pd.DataFrame(rows, columns=columns) # dataframe used to calculate csmf

            return resultdf
        
        finally:
            # Return to original directory
            os.chdir(original_dir)
            print("Analysis completed!")

# calcuate the csmf for population 

def getCsmf(df, top):    
    """
    This method is used to calculate the cause specific mortality rate for a given population,
    it is dependent on the inter va 6 algorithm and requires path and number for top csmf to
    it returns a tuple 
    """
    print(f"loading dataframe to get csmf")    
    
    # Ensuring its a data-frame 
    df  
    #print(df)
    
    #df.to_csv(r"C:\Users\Shukurani\Desktop\interva6\output\df.csv")
    # Check number of columns (should be 80)
    print("Checking the number of columns:", df.shape[1])

    if df.shape[1] != 80:
        print(f"Expected 80 columns but got {df.shape[1]}!")
        # Return empty tuple if columns don't match
        return ()
    
    # --- Step 1: Compute totals for columns 13 onward ---
    totals = {}
    for col in df.columns[13:]:
        total = pd.to_numeric(df[col], errors='coerce').sum()
        totals[col] = total

    # Add a label in the first column for the totals row
    totals[df.columns[0]] = "Total"

    # --- Step 2: Append the totals as a new row ---
    df_totals = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

    # --- Step 3: Compute the grand total (sum of all totals from column 13 onward) ---
    grand_total_value = sum(v for k, v in totals.items() if isinstance(v, (int, float)))

    # Check if grand_total_value is zero to avoid division by zero
    if grand_total_value == 0:
        print("Warning: Grand total is zero. Cannot compute CSMF.")
        return ()

    # Create a dictionary for the grand total row
    grand_total = {col: "" for col in df.columns}
    grand_total[df.columns[0]] = "Grand Total"

    # Fill each numeric column with the grand total of totals (optional)
    for col in df.columns[13:]:
        grand_total[col] = grand_total_value

    # Append the grand total row
    df_totals = pd.concat([df_totals, pd.DataFrame([grand_total])], ignore_index=True)

    # --- Step 4: Normalize totals (divide each total by the grand total) ---
    normalized = {
        col: (val / grand_total_value) if isinstance(val, (int, float)) else 0
        for col, val in totals.items() if col != df.columns[0]  # Skip the ID column
    }

    # --- Step 5: Get top columns with highest normalized values ---
    topcsmf = sorted(normalized.items(), key=lambda x: x[1], reverse=True)[:top]
    csmf_df = pd.DataFrame(topcsmf) # data frame creation

    print(f"\nTop {top} columns by normalized total:")
    for name, val in topcsmf:
        print(f"{name}: {val:.4f}")

    # --- Step 6: Save the final CSV ---
    #output_path = r"C:\Users\Shukurani\Desktop\interva6\output\result_with_totals.csv"
    #try:
     #   df_totals.to_csv(output_path, index=False)
     #   print(f"\nCSV with totals and grand total saved to:\n{output_path}")
    #except Exception as e:
    #  print(f"Warning: Could not save CSV file: {e}")

    print(f"\nTop {top} diseases:\n{csmf_df}")
    

    #  Identify the first 3 pregnancy-related COD columns (cols[13:16])
    #  Exclude them from the main totals / normalization
    #  Compute the sum of these excluded columns → assign to Undetermined
    #  Normalize all remaining totals including “Undetermined”
    #  Sort by normalized values and pick top N
    #  Ensure top includes "Undetermined" if it ranks high
    


    return csmf_df # csmf data frame


def main():
    """Main function for testing."""
    # Initialize analyzer
    analyzer = InterVA2022()
    
    # Example usage
    try:
        
        results = analyzer.analyze(
            input_data=r"C:\Users\Shukurani\Desktop\interva6\ccva_2022.csv",
            probbase_file=r"C:\Users\Shukurani\Desktop\interva6\probbase2022.csv",
            hiv="h",
            malaria="l", 
            covid="v",
            write=True,
            directory="output",
            filename="VA2022_result",
            output="extended" # use this to ensure extended result is provided 
        )

        #getCsmf(results, 3) # get the csmf for the population

        print(f"\nAnalysis complete! Processed {len(results['ID'])} records")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()