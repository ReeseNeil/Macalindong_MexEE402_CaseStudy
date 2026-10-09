# Macalindong_MexEE402_CaseStudy

# MexEE 402: Data Preprocessing Case Study

MexEE Elective 2: Data Science and Machine Learning
Batangas State University, Alangilan Campus
1st Semester, AY 2026-2027

## Member

| Name | Student Number | Section |
|---|---|---|
| Macalindong, Aris Neil C. | 6 | MEXE-4101 |

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
List any mistake you found in the original notebooks, and the correct version.
There are real ones in there. Finding them earns points.

### Chapter 1_2_3
**Error 1**: Filling missing values (Year / Publisher) on the whole dataset before performing the train-test split leaks summary statistics (like median or mean) from test data into training data.

Incorrect Code:
```
# Imputing on the entire dataframe BEFORE splitting causes Data Leakage
df['Year'].fillna(df['Year'].median(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
Correct Code:
```
# Split the dataset FIRST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Compute median ONLY on X_train and apply to both train and test
median_year = X_train['Year'].median()
X_train['Year'].fillna(median_year, inplace=True)
X_test['Year'].fillna(median_year, inplace=True)
```
### Chapter 4
**Error 1**: Using OrdinalEncoder() without specifying explicit category order defaults to alphabetical sorting. This incorrectly assigns Little (0), Lots (1), Medium (2) instead of the natural scale Little (0), Medium (1), Lots (2).

Incorrect Code:
```
# Default encoder sorts alphabetically: ['Little', 'Lots', 'Medium']
encoder = OrdinalEncoder()
df['Amount_Encoded'] = encoder.fit_transform(df[['Amount']])
```

Correct Code:
```
# Pass explicit category ordering
encoder = OrdinalEncoder(categories=[['Little', 'Medium', 'Lots']])
df['Amount_Encoded'] = encoder.fit_transform(df[['Amount']])
```

**Error 2**: Creating interaction features like Lemonade per Degree without checking if Temperature is 0 causes division-by-zero errors or infinite values (inf).

Incorrect Code:
```
df['Lemonade per Degree'] = df['Lemonade Sold'] / df['Temperature']
```

Correct Code:
```
import numpy as np

df['Lemonade per Degree'] = df['Lemonade Sold'] / df['Temperature'].replace(
    0, np.nan
)
```

### Chapter 5
**Error 1**: Calling fit_transform() on the test dataset re-fits the scaler's parameters ($\mu$ and $\sigma$) using test set metrics, leading to target leakage and inconsistent scaling.

Incorrect Code:
```
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(
    X_test
)  # INCORRECT: Re-estimates mean and std on test set!
```

Correct Code:
```
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(
    X_test
)  # CORRECT: Uses parameters fitted from X_train
```

### Chapter 6
**Error 1**: Filtering outliers using boolean indexing creates a pandas dataframe view. Mutating it directly afterwards raises a SettingWithCopyWarning.

Incorrect Code:
```
# Clean dataframe is a view, not a copy
clean_df = df[df['z_score'].abs() < 3]
clean_df['scaled_val'] = clean_df['val'] / 10  # Triggers warning
```

Correct Code:
```
# Use .copy() to decouple from original dataframe
clean_df = df[df['z_score'].abs() < 3].copy()
clean_df['scaled_val'] = clean_df['val'] / 10
```

### Chapter 7
**Error 1**: LassoCV penalizes parameters based on their absolute magnitudes. If features are unscaled, variables with larger natural numeric scales (e.g., Fare vs Age) will be unfairly penalized or favored.

Incorrect Code:
```
from sklearn.linear_model import LassoCV

# Fitting Lasso directly on unscaled X_train biases feature selection
lasso = LassoCV().fit(X_train, y_train)
```

Correct Code:
```
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Scale features inside a pipeline before applying LassoCV
pipeline = make_pipeline(StandardScaler(), LassoCV())
pipeline.fit(X_train, y_train)
```

### Chapter 8
**Error 1**: When OneHotEncoder encounters unseen category levels in test/production data, it throws a ValueError unless explicitly configured to ignore them.

Incorrect Code:
```
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    (
        'encoder',
        OneHotEncoder(),
    ),  # Will break on test data if unknown categories exist
])
```

Correct Code:
```
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])
```

### Chapter 9
**Error 1**: When using pd.cut() with right=True (default), the lowest bin edge (e.g., age 0) is excluded by default, turning boundary values into NaN.

Incorrect Code:
```
# Age 0 becomes NaN because right=True excludes lower bound
df['Age_Group'] = pd.cut(
    df['Age'], bins=[0, 12, 50, 200], labels=['Child', 'Adult', 'Elderly']
)
```

Correct Code:
```
# Set include_lowest=True to include 0 in the 'Child' bin
df['Age_Group'] = pd.cut(
    df['Age'],
    bins=[0, 12, 50, 200],
    labels=['Child', 'Adult', 'Elderly'],
    include_lowest=True,
)
```



## Note on AI tools

I used Gemini Flash 3.6 Extended solely for proofreading (grammar checking and enhancing sentences) my chapter answers. The conversation can be viewed here [Gemini Conversation](https://share.gemini.google/HBPBJWAGfoqF)


## References

McKinney, W. (2021). Python for Data Analysis, 3rd ed. O'Reilly.
VanderPlas, J. Python Data Science Handbook.
