import pandas as pd
from preprocess import TextPreprocessor

df = pd.read_csv('../../dataset/comprehensive_emotions.csv')
preprocessor = TextPreprocessor()

# Test preprocessing on a few samples
print('Original samples:')
for i in range(5):
    print(f'{i}: {df.iloc[i]["text"]}')

print('\nPreprocessed samples:')
df['cleaned_text'] = df['text'].apply(preprocessor.clean_text)
for i in range(5):
    print(f'{i}: {df.iloc[i]["cleaned_text"]}')

print(f'\nAny NaN values: {df["cleaned_text"].isna().any()}')
print(f'Empty strings: {(df["cleaned_text"] == "").sum()}')
print(f'Whitespace only: {(df["cleaned_text"].str.strip() == "").sum()}')
