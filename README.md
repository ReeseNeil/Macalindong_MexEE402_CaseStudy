# Macalindong_MexEE402_CaseStudy

# MexEE 402: Data Preprocessing Case Study

MexEE Elective 2: Data Science and Machine Learning
Batangas State University, Alangilan Campus
1st Semester, AY 2026-2027

## Member

| Name | Student Number | Section |
|---|---|---|
| Macalindong, Aris Neil C. | 23-06911 | MEXE-4101 |

## Notebook links

| Chapter | Member 1 |
|---|---|
| Ch1_2_3 | [link](https://colab.research.google.com/drive/1TZdJuudcMxbkQTZg2ylyWfC0F09WTBrx?usp=sharing) |
| Ch4 | [link](https://colab.research.google.com/drive/1Ke3_lDXun6bhcCKbVWhTOx-KS5xLNI4u?usp=sharing) |
| Ch5 | [link](https://colab.research.google.com/drive/1RV_4__UulWq3OdzqrhB2_SAQwHSvFRQp?usp=sharing) |
| Ch6 | [link](https://colab.research.google.com/drive/1-7zePx99y_IHFRj64RS7bybKG2KVMOvf?usp=sharing) |
| Ch7 | [link](https://colab.research.google.com/drive/1ku2iYbiSNSesZCGiwxIvxm36Q_PE-Izu?usp=sharing) |
| Ch8 | [link](https://colab.research.google.com/drive/1AlTIgpwK1jDXkL49vbZFDp3lJrHTLFv1?usp=sharing) |
| Ch9 | [link](https://colab.research.google.com/drive/1PEDIgwrPCQ5Jjvug9cApcTJhrD-lJURw?usp=sharing) |

## What I learned
One short paragraph per chapter, Ch1_2_3 to Ch9. Say what the chapter taught
you and what surprised you. Not what the library does, but what you understood.

### Chapter 1_2_3
### Chapter 4
### Chapter 5
### Chapter 6
### Chapter 7
### Chapter 8
### Chapter 9



## Errors I found
> List any mistake you found in the original notebooks, and the correct version.
There are real ones in there. Finding them earns points.

### Chapter 1_2_3
**Mistake 1**: Mean imputation gives an impossible Year

**⚠️ Problem:** Year is a whole number, but the mean fills it with 2006.406, which is not a real release year. Mean also gets pulled by old games (1980 to 2020).

✖️ Incorrect Code:
```
df['Year'].fillna(df['Year'].mean(), inplace=True)
```
✔️ Correct Code:
```
df['Year'] = df['Year'].fillna(df['Year'].median())
```

**Mistake 2**: Deletion step does nothing

**⚠️ Problem**: Publisher was already filled with the mode, so `notna()` finds no missing rows to drop.

✖️ Incorrect Code:
```
df['Publisher'].fillna(df['Publisher'].mode()[0], inplace=True)
df = df[df['Publisher'].notna()]
```

✔️ Correct Code:
```
# choose one method per column
df = df[df['Publisher'].notna()]   # deletion (only 58 rows)
```

### Chapter 4

**Mistake 1**: Bins leave the "very hot" label unused

**⚠️ Problem**: With bins [70, 75, 85, 95, 100] the highest temperature (95) falls in hot, so very hot never appears. Also, 75 lands in cool because intervals are right-closed.


✖️ Incorrect Code:
```
bins = [70, 75, 85, 95, 100]
```

✔️ Correct Code:
```
bins = [70, 75, 85, 90, 100] 
```

**Mistake 2**: Ordinal encoding does not match the markdown

**⚠️ Problem**: The markdown says Little = 1, Medium = 2, Lots = 3, but the output is 0.0, 1.0, 2.0.


✖️ Incorrect Code:
```
df_3['Ice_encoded'] = ord_enc.fit_transform(df_3[['Ice']])
```

✔️ Correct Code:
```
df_3['Ice_encoded'] = ord_enc.fit_transform(df_3[['Ice']]) + 1   # now 1, 2, 3
```

**Mistake 3**: One-hot output is True/False, not 1/0

**⚠️ Problem**: Newer pandas returns booleans, but the markdown shows 1 and 0.

✖️ Incorrect Code:
```
df_encoded = pd.get_dummies(df_2, columns=['Weather'])
```

✔️ Correct Code:
```
df_encoded = pd.get_dummies(df_2, columns=['Weather'], dtype=int)
```

### Chapter 6
**Mistake 1**: Z-score threshold is too high

**⚠️ Problem**: With 8 values, the largest possible Z-score is about 2.65, so a cutoff of 3 doesn't do anything.

✖️ Incorrect Code:
```
outliers = data[np.abs(z_scores) > 3]
```

✔️ Correct Code:
```
outliers = data[np.abs(z_scores) > 2]
```

### Chapter 7
**Mistake 1**: The filter keeps the target and ignores negative correlations

**⚠️ Problem**: `final grade` correlates 1.0 with itself, so it stays in the "relevant features". Also `> 0.5` would throw away a significant feature with negative correlation like -0.9.

✖️ Incorrect Code:
```
relevant_features = correlations[correlations > 0.5]
```

✔️ Correct Code:
```
correlations = df_2.corr()['final grade'].drop('final grade')
relevant_features = correlations[correlations.abs() > 0.5]
```

### Chapter 8
**Mistake 1**: Every other column is dropped

**⚠️ Problem**: `ColumnTransformer` drops unlisted columns by default. The output data contains only `Age` and `Fare`. `Sex`, `Pclass`, and the rest are gone.

✖️ Incorrect Code:
```
preprocessor = ColumnTransformer(transformers=[
('age_fare', pipeline, ['Age', 'Fare'])
])
```

✔️ Correct Code:
```
preprocessor = ColumnTransformer(transformers=[
('age_fare', pipeline, ['Age', 'Fare'])
], remainder='passthrough')
```

### Chapter 9
**Mistake 1**: Discretization overwrites the original Age column

**⚠️ Problem**:  `pd.cut` replaces the numeric Age with text labels. So the original data are lost,and and "before" and "after" can't be compared. Also, 50 is a young cutoff for `Elderly`.

✖️ Incorrect Code:
```
bins = [0, 12, 50, 200]
labels = ['Child', 'Adult', 'Elderly']
data['Age'] = pd.cut(data['Age'], bins=bins, labels=labels)
```

✔️ Correct Code:
```
bins = [0, 12, 60, 120]
labels = ['Child', 'Adult', 'Senior']
data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=labels)
```

**Mistake 2**: Incorrect "before" and "after" plots

**⚠️ Problem**:  The "before" cell plots Age after it was already converted (bars 581, 64, 69 are group counts). The "after" cell plots `titanic_preprocessed[:, 2]`, the `Embarked_C` one-hot column, not age.

✖️ Incorrect Code:
```
plt.hist(data['Age'].dropna(), alpha=0.5, label='Before discretization')
plt.hist(titanic_preprocessed[:,2], alpha=0.5, label='After discretization')
```

✔️ Correct Code:
```
plt.hist(data['Age'].dropna(), bins=20, alpha=0.7, label='Before discretization')
data['Age_Group'].value_counts().reindex(labels).plot(kind='bar', alpha=0.7, label='After discretization')
plt.legend()
plt.show()
```

## Note on AI tools
I used Gemini Flash 3.6 Extended solely for proofreading (grammar checking and enhancing sentences) my chapter answers. The conversation can be viewed here [Gemini Conversation](https://share.gemini.google/HBPBJWAGfoqF)

## References

- McKinney, W. (2021). Python for Data Analysis, 3rd ed. O'Reilly.
- VanderPlas, J. Python Data Science Handbook.
- GeeksforGeeks, (n.d.). Feature engineering: Scaling, normalization and standardization.
- GeeksforGeeks, (2025, July 23). Feature selection using SelectFromModel and LassoCV in Scikit Learn.
- GeeksforGeeks. (2025, November 29). Discretization.
