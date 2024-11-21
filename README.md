![Static Badge](https://img.shields.io/badge/Python-8A2BE2)
![Static Badge](https://img.shields.io/badge/AI-8A2BE2)
![Static Badge](https://img.shields.io/badge/PET%20/%20MRI-4CAF50)

# PET-MRI-data-feature-selection-for-suppport-vector-machine

This project contains code to perform statistical analysis and data visualization for comparing the features of subjects diagnosed with Alzheimer's Disease (AD), Mild Cognitive Impairment (MCI), and Normal Controls (NC). It includes image and medical features from brain scans, such as SUVR (Standardized Uptake Value Ratio) and volume measures for various brain regions of interest (ROIs). The groups will not be classified as AD vs. MCI and MCI vs. NC using the support vector machine (SVM) model.

## Requirements

- Python 3.x
- Pandas
- NumPy
- SciPy
- Matplotlib

You can install the required libraries by running:

```bash
pip install pandas numpy scipy matplotlib
```

## Files

- **Features-2.xlsx**: Excel file containing different feature sets for AD, MCI, and NC groups. The file has multiple sheets with image and medical data for the subjects:
  - `AD`, `MCI`, `NC`: Image features for the respective groups.
  - `AD-ref`, `MCI-ref`, `NC-ref`: Reference SUVR values for each group.

## Data Processing

The code performs the following steps:

1. **Import Data**: Loads features from the `Features-2.xlsx` Excel file.
2. **SUVR Calculation**: Normalizes image features by dividing by the reference SUVR values.
3. **Outlier Handling**: Replaces outliers (values outside the interquartile range) with the median value for each feature.
4. **Statistical Analysis**: 
   - **Two-sample t-tests**: Compares AD vs. MCI and MCI vs. NC for both SUVR and volume measures.
   - **Levene’s Test**: Assesses equality of variance before performing t-tests.

## Statistical Results

For each group comparison (AD vs. MCI, MCI vs. NC) for both SUVR and volume, the script stores the results in the following data frames:

- `results_suvr_all`: Contains the results for the SUVR comparisons.
- `results_volume_all`: Contains the results for the volume comparisons.

The comparison results are added as columns:
- `suvr_AD_vs_MCI`: Results for the AD vs. MCI SUVR comparison.
- `suvr_MCI_vs_NC`: Results for the MCI vs. NC SUVR comparison.
- `volume_AD_vs_MCI`: Results for the AD vs. MCI volume comparison.
- `volume_MCI_vs_NC`: Results for the MCI vs. NC volume comparison.

A correction for multiple comparisons is applied using the Bonferroni method (adjusted significance level).

## Plotting

The script generates two plots:

1. **SUVR Comparison**: Displays the average SUVR for each group (AD, MCI, NC) across the brain ROIs.
2. **Volume Comparison**: Displays the average volume (in mm³) for each group (AD, MCI, NC) across the same ROIs.

Both plots use different markers to represent the groups and ensure that the cerebral cortex regions are excluded for visualization purposes.

## How to Run

1. Place the `Features-2.xlsx` file in the same directory as the script.
2. Run the script using a Python environment:

```bash
python analysis.py
```

3. The script will output the statistical results and display the plots.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
