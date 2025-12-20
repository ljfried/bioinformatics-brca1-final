# How to Run

## Required Dependencies

Python 3.8+
Packages: numpy, pandas, scikit-learn, hmmlearn

## Required Files

final.py

Either:
Given files "clinvar_result_benign.txt" and "clinvar_result_pathogenic.txt"
Your own custom files with a custom name and/or location.

## Running Program

If your files are named "clinvar_result_benign.txt" and "clinvar_result_pathogenic.txt", and are in the same directory as final.py, you can simply run: 

python final.py

in your terminal. 

### Custom Files

python final.py --benign "your_benign_file_path" --path "your_pathogenic_file_path"

Custom files must be tab-separated and include the following columns:
    
    Canonical SPDI 
        Format: 
            SequenceID:Position:Deleted:Inserted
    
    Germline Classification 
        Accepted values: 
            Benign, 
            Likely benign, 
            Pathogenic, 
            Likely pathogenic, 
            Pathogenic/Likely pathogenic. 
        Other values will be ignored.

### Custom modes

Use the --mode flag to select which model(s) to execute.

    type — mutation-type HMM only
    location - mutation-location HMM only
    both - both models (default)