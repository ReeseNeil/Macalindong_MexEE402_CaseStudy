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
**Problem:** Year is a whole number, but the mean fills it with 2006.406, which is not a real release year. Mean also gets pulled by old games (1980 to 2020).

Incorrect Code:
```
df['Year'].fillna(df['Year'].mean(), inplace=True)
```
Correct Code:
```
df['Year'] = df['Year'].fillna(df['Year'].median())
```

**Mistake 2**: Deletion step does nothing
**Problem**: Publisher was already filled with the mode, so `notna()` finds no missing rows to drop. The notebook presents two methods but only one actually ran.

Incorrect Code:
```
df['Publisher'].fillna(df['Publisher'].mode()[0], inplace=True)
df = df[df['Publisher'].notna()]
```

Correct Code:
```
# choose one method per column
df = df[df['Publisher'].notna()]   # deletion (only 58 rows)
```

**Mistake 3**: Outlier filter removes the two best-selling games
**Problem**: `Global_Sales <= 40` deletes Wii Sports (82.74) and Super Mario Bros. (40.24). These are real hits, not errors, so removing them is the wrong fix. 

Incorrect Code:
```
df = df[df['Global_Sales'] <= 40]
```

Correct Code:
```
# keep real values; cap instead of delete
df['Global_Sales'] = df['Global_Sales'].clip(upper=40)
```

### Chapter 4

**Mistake 1**: Bins leave the "very hot" label unused
**Problem**: With bins [70, 75, 85, 95, 100] the highest temperature (95) falls in hot, so very hot never appears. Also, 75 lands in cool because intervals are right-closed.


Incorrect Code:
```
bins = [70, 75, 85, 95, 100]
```

Correct Code:
```
bins = [70, 75, 85, 90, 100] 
```

**Mistake 2**: Ordinal encoding does not match the markdown
**Problem**: The markdown says Little = 1, Medium = 2, Lots = 3, but the output is 0.0, 1.0, 2.0.


Incorrect Code:
```
df_3['Ice_encoded'] = ord_enc.fit_transform(df_3[['Ice']])
```

Correct Code:
```
df_3['Ice_encoded'] = ord_enc.fit_transform(df_3[['Ice']]) + 1   # now 1, 2, 3
```

**Mistake 3**: One-hot output is True/False, not 1/0
**Problem**: Newer pandas returns booleans, but the markdown shows [1,0,0]. Models need numbers.

Incorrect Code:
```
df_encoded = pd.get_dummies(df_2, columns=['Weather'])
```

Correct Code:
```
df_encoded = pd.get_dummies(df_2, columns=['Weather'], dtype=int)
```

### Chapter 5
**Mistake 1**: 

Incorrect Code:
```
```

Correct Code:
```
```

### Chapter 6
**Mistake 1**: 

Incorrect Code:
```
```

Correct Code:
```
```

### Chapter 7
**Mistake 1**:
Incorrect Code:
```
```

Correct Code:
```
```

### Chapter 8
**Mistake 1**:

Incorrect Code:
```
```

Correct Code:
```
```

### Chapter 9
**Mistake 1**: 

Incorrect Code:
```
```

Correct Code:
```
```



## Note on AI tools

I used Gemini Flash 3.6 Extended solely for proofreading (grammar checking and enhancing sentences) my chapter answers. The conversation can be viewed here [Gemini Conversation](https://share.gemini.google/HBPBJWAGfoqF)


## References

McKinney, W. (2021). Python for Data Analysis, 3rd ed. O'Reilly.
VanderPlas, J. Python Data Science Handbook.
GeeksforGeeks, (n.d.). Feature engineering: Scaling, normalization and standardization.
GeeksforGeeks, (2025, July 23). Feature selection using SelectFromModel and LassoCV in Scikit Learn.
GeeksforGeeks. (2025, November 29). Discretization.
