import pandas as pd


def hapmap_to_numeric(df):
    
    # Create a new dictionary for each row since same alleles can repeat
    for row in range(len(df)):
        dictionary = {
            "NN": 1
        }
        # Whole row values for each SNP
        values = df.iloc[row, 11:]

        # Creating dictionary for storing allele conversion values
        alleles = df.iloc[row, 1].split("/")
        dictionary[str(alleles[0]+alleles[0])] = 0
        dictionary[str(alleles[1]+alleles[1])] = 2
        dictionary[str(alleles[0]+alleles[1])] = 1
        dictionary[str(alleles[1]+alleles[0])] = 1

        # Change the row values based on the dictionary values
        for index, value in enumerate(values):
            df.iloc[row, (11 + index)] = dictionary[value]

    return df

# Remove sep= in case csv file
df = pd.read_csv("filename.txt", sep='\t+')
result = hapmap_to_numeric(df)
# Delete columns that are not needed (column 2 to 11)
for i in range(1, 11):
    result = result.drop(df.columns[i], axis=1)

# Save file
csv_filename = 'converted.csv'
result.to_csv(csv_filename, header=True, index=False)
