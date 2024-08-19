# ei-common-bean

## 50-fold-crossvalidation-IBM-tool.py
- Function to be used in AutoXAI4Omics/src/plotting/plot_both.py
- group_size should be changed to 20% of the data analysed or data.shape[\0\] * 0.2
- nsplits can be changed to any number of cross validation rounds required

## hapmap-to-numeric.py
- Converts HapMap to numeric format
- Time complexity: O(n×m), where n is the number of rows and m is the number of columns from the 12th onwards
- Starts from 12th column since columns before are 'headers' (can be changed to any number required)
- 0: Major allele
- 1: NN, heterozygous
- 2: Minor allele

## plots.py
- Function plots a box plot and violin plot from the cross validation results by IBM tool to compare all of the models used
