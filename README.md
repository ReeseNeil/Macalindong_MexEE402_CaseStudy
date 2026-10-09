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

| Chapter | Macalindong |
|---|---|
| Ch1_2_3 | [link](https://colab.research.google.com/drive/1TZdJuudcMxbkQTZg2ylyWfC0F09WTBrx?usp=sharing) |
| Ch4 | [link](https://colab.research.google.com/drive/1Ke3_lDXun6bhcCKbVWhTOx-KS5xLNI4u?usp=sharing) |
| Ch5 | [link](https://colab.research.google.com/drive/1RV_4__UulWq3OdzqrhB2_SAQwHSvFRQp?usp=sharing) |
| Ch6 | [link](https://colab.research.google.com/drive/1-7zePx99y_IHFRj64RS7bybKG2KVMOvf?usp=sharing) |
| Ch7 | [link](https://colab.research.google.com/drive/1ku2iYbiSNSesZCGiwxIvxm36Q_PE-Izu?usp=sharing) |
| Ch8 | [link](https://colab.research.google.com/drive/1AlTIgpwK1jDXkL49vbZFDp3lJrHTLFv1?usp=sharing) |
| Ch9 | [link](https://colab.research.google.com/drive/1PEDIgwrPCQ5Jjvug9cApcTJhrD-lJURw?usp=sharing) |

## What I learned
> One short paragraph per chapter, Ch1_2_3 to Ch9. Say what the chapter taught you and what surprised you. Not what the library does, but what you understood.

### 📂 Chapter 1_2_3: Introduction to preprocessing, exploring, and cleaning data.

I learned from this chapter that raw data is usually messy and that analyzing and preprocessing it is important to identify and fix missing values, inconsistencies, and irrelevant columns prior to training. Using inspection tools like `head()`, `info()`, and `describe()` makes it easy to spot specific issues such as missing values in `Year` and `Publisher` within the imported `vgsales` dataset. Columns may also seem fine and organized at the surface, such as `Rank`, but closer inspection reveals that it only introduces noise in machine learning since it's just a positional index without any patterns.  


### 📂 Chapter 4: Transformation, Feature Engineering, and Encoding

This chapter showed how to uncover hidden relations by using ratios and correct category encoding, which deepens the context available for machine learning. An example of this is creating interaction features like `Lemonade per Degree` or binning continuous data into labels like `cool` and `hot`. Another thing that surprised me is the damaging effects of incorrect encoding, such as using ordinal encoding for nominal categories such as weather, wherein the model is forced to assume incorrect hierarchy and mathematical sequence, thus leading to wrong logic and bias.


### 📂 Chapter 5: Scaling and Normalization

I understood that scaling tames the scope of numerical data so that they are normalized and equal to every other feature in model calculations. The most fascinating aspect of this chapter is that an algorithm's absence of inherent understanding of scale introduces bias, as it will lead the model to lean heavily to higher numerical data (an example of this is comparing `Grades` and `Study Hours`). Without scaling, the higher numbers will overshadow other numerical data, thus obscuring the true underlying patterns. 


### 📂 Chapter 6: Outlier Detection

I learned what mathematical processes such as Z-scores and IQR to use to identify outliers or data points that is extremely separate and different from the normal data distribution. One thing that piqued my attention is the thought that outliers are not always "wrong data" that has to be removed; and that removing them thoughtlessly may remove remarkable patterns that actually exist in real life. Hence, it is sometimes more optimal to cap or transform those values rather than directly removing them.


### 📂 Chapter 7: Feature Selection

I learned from this chapter that processes, like the most relevant IQR, are set among all the features in the dataset and are crucial to speed up training and minimize model complexity. This shattered my initial belief that more data or features leads to better models. When in reality, adding redundant features removed accuracy. It was also fascinating how methods exist, such as `LassoCV`, for reducing the gravity of trivial variables to zero. 


### 📂 Chapter 8: Constructing a Preprocessing Pipeline

I learned that preprocessing pipelines are simply stick-together processes in cleaning data to make the workflow automatic, clean, and more organized. By chaining steps like `SimpleImputer` and `StandardScaler` into a single pipeline with `ColumnTransformer`, preprocessing now becomes more modular.


### 📂 Chapter 9: Full Pipeline and Visualization

In this last chapter, a full end-to-end pipeline for both numerical and categorical data was stitched together along with its visual representations (via `matplotlib.pylot` and `seaborn`) based on the Titanic dataset (`train.csv`). Another useful thing that I learned here is the importance of making visual plots after preprocessing not just for formality and aesthetic purposes but rather to validate how processes such as age discretization change what the data tells us.

## Errors I found
> List any mistake you found in the original notebooks, and the correct version.
There are real ones in there. Finding them earns points.

### Chapter 1_2_3
**Mistake 1**: Mean imputation gives an impossible Year

**⚠️ Problem:** Year is a whole number, but the mean fills it with 2006.406, which is not a real release year. Mean also gets pulled by old games (1980 to 2020).

✖️ Incorrect Code:
```python
df['Year'].fillna(df['Year'].mean(), inplace=True)
```
✔️ Correct Code:
```python
df['Year'] = df['Year'].fillna(df['Year'].median())
```

**Mistake 2**: Deletion step does nothing

**⚠️ Problem**: Publisher was already filled with the mode, so `notna()` finds no missing rows to drop.

✖️ Incorrect Code:
```python
df['Publisher'].fillna(df['Publisher'].mode()[0], inplace=True)
df = df[df['Publisher'].notna()]
```

✔️ Correct Code:
```python
# choose one method per column
df = df[df['Publisher'].notna()]   # deletion (only 58 rows)
```

### Chapter 4

**Mistake 1**: Bins leave the "very hot" label unused

**⚠️ Problem**: With bins [70, 75, 85, 95, 100] the highest temperature (95) falls in hot, so very hot never appears. Also, 75 lands in cool because intervals are right-closed.


✖️ Incorrect Code:
```python
bins = [70, 75, 85, 95, 100]
```

✔️ Correct Code:
```python
bins = [70, 75, 85, 90, 100] 
```

**Mistake 2**: Ordinal encoding does not match the markdown

**⚠️ Problem**: The markdown says Little = 1, Medium = 2, Lots = 3, but the output is 0.0, 1.0, 2.0.


✖️ Incorrect Code:
```python
df_3['Ice_encoded'] = ord_enc.fit_transform(df_3[['Ice']])
```

✔️ Correct Code:
```python
df_3['Ice_encoded'] = ord_enc.fit_transform(df_3[['Ice']]) + 1   # now 1, 2, 3
```

**Mistake 3**: One-hot output is True/False, not 1/0

**⚠️ Problem**: Newer pandas returns booleans, but the markdown shows 1 and 0.

✖️ Incorrect Code:
```python
df_encoded = pd.get_dummies(df_2, columns=['Weather'])
```

✔️ Correct Code:
```python
df_encoded = pd.get_dummies(df_2, columns=['Weather'], dtype=int)
```

### Chapter 6
**Mistake 1**: Z-score threshold is too high

**⚠️ Problem**: With 8 values, the largest possible Z-score is about 2.65, so a cutoff of 3 doesn't do anything.

✖️ Incorrect Code:
```python
outliers = data[np.abs(z_scores) > 3]
```

✔️ Correct Code:
```python
outliers = data[np.abs(z_scores) > 2]
```

### Chapter 7
**Mistake 1**: The filter keeps the target and ignores negative correlations

**⚠️ Problem**: `final grade` correlates 1.0 with itself, so it stays in the "relevant features". Also `> 0.5` would throw away a significant feature with negative correlation like -0.9.

✖️ Incorrect Code:
```python
relevant_features = correlations[correlations > 0.5]
```

✔️ Correct Code:
```python
correlations = df_2.corr()['final grade'].drop('final grade')
relevant_features = correlations[correlations.abs() > 0.5]
```

### Chapter 8
**Mistake 1**: Every other column is dropped

**⚠️ Problem**: `ColumnTransformer` drops unlisted columns by default. The output data contains only `Age` and `Fare`. `Sex`, `Pclass`, and the rest are gone.

✖️ Incorrect Code:
```python
preprocessor = ColumnTransformer(transformers=[
('age_fare', pipeline, ['Age', 'Fare'])
])
```

✔️ Correct Code:
```python
preprocessor = ColumnTransformer(transformers=[
('age_fare', pipeline, ['Age', 'Fare'])
], remainder='passthrough')
```

### Chapter 9
**Mistake 1**: Discretization overwrites the original Age column

**⚠️ Problem**:  `pd.cut` replaces the numeric Age with text labels. So the original data are lost,and and "before" and "after" can't be compared. Also, 50 is a young cutoff for `Elderly`.

✖️ Incorrect Code:
```python
bins = [0, 12, 50, 200]
labels = ['Child', 'Adult', 'Elderly']
data['Age'] = pd.cut(data['Age'], bins=bins, labels=labels)
```

✔️ Correct Code:
```python
bins = [0, 12, 60, 120]
labels = ['Child', 'Adult', 'Senior']
data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=labels)
```

**Mistake 2**: Incorrect "before" and "after" plots

**⚠️ Problem**:  The "before" cell plots Age after it was already converted (bars 581, 64, 69 are group counts). The "after" cell plots `titanic_preprocessed[:, 2]`, the `Embarked_C` one-hot column, not age.

✖️ Incorrect Code:
```python
plt.hist(data['Age'].dropna(), alpha=0.5, label='Before discretization')
plt.hist(titanic_preprocessed[:,2], alpha=0.5, label='After discretization')
```

✔️ Correct Code:
```python
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
- Pandas (n.d.). User Guide — pandas 3.0.6 documentation
