import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv(r'D:\Project\Bone_Gender_Age_Project\dataset\Age and Gender Detection.csv')

print("Original shape:", df.shape)
print("\nAge unique values before cleaning:")
print(df['Age'].unique())

# Age standardize කිරීමේ function එක
def standardize_age(age):
    if pd.isna(age):
        return np.nan
    
    age = str(age).strip()
    
    # Exact age numbers handle කිරීම
    try:
        exact_age = float(age)
        if 18 <= exact_age < 25:
            return '18-25'
        elif 26 <= exact_age < 35:
            return '26-35'
        elif 36 <= exact_age < 45:
            return '36-45'
        elif 46 <= exact_age < 55:
            return '46-55'
        else:
            return '55+'
    except:
        pass
    
    # Range values handle කිරීම
    age_mapping = {
        '18-20': '18-25', '18-22': '18-25', '20-22': '18-25',
        '20-24': '18-25', '20-25': '18-25', '22-25': '18-25',
        '25-30': '25-35', '25+': '25-35',
        '30-35': '25-35',
        '30-40': '35-45',
        '35-45': '35-45',
        '40-50': '45-55', '40+': '45-55',
        '50+': '55+', '50-60': '55+',
        '20-30': np.nan, '30-50': np.nan, '30+': np.nan
    }
    
    return age_mapping.get(age, np.nan)

# Age column standardize කරන්න
df['Age_Group'] = df['Age'].apply(standardize_age)

print("\nAge groups after standardization:")
print(df['Age_Group'].value_counts())

print("\nMissing values in Age_Group:", df['Age_Group'].isna().sum())

# Missing Age_Group rows drop කරන්න
df = df.dropna(subset=['Age_Group'])

print("\nShape after dropping missing age rows:", df.shape)

# Measurement columns identify කරන්න
bone_columns = [col for col in df.columns if col not in 
                ['Sex', 'Age', 'Age_Group', 'LHUM', 'RHUM', 'LRAD', 
                 'RRAD', 'LFEM', 'RFEM', 'LTIB', 'RTIB', 'OSCX']]

print("\nMissing values in measurement columns:")
print(df[bone_columns].isnull().sum()[df[bone_columns].isnull().sum() > 0])

# Missing values median imputation වලින් fill කරන්න
df[bone_columns] = df[bone_columns].fillna(df[bone_columns].median())

print("\nMissing values after imputation:", df[bone_columns].isnull().sum().sum())

# Average features හදන්න (Left + Right average)
df['AVG_HML'] = df[['LHML', 'RHML']].mean(axis=1)
df['AVG_HEB'] = df[['LHEB', 'RHEB']].mean(axis=1)
df['AVG_HHD'] = df[['LHHD', 'RHHD']].mean(axis=1)
df['AVG_HMLD'] = df[['LHMLD', 'RHMLD']].mean(axis=1)
df['AVG_HAPD'] = df[['LHAPD', 'RHAPD']].mean(axis=1)
df['AVG_RML'] = df[['LRML', 'RRML']].mean(axis=1)
df['AVG_RMLD'] = df[['LRMLD', 'RRMLD']].mean(axis=1)
df['AVG_RAPD'] = df[['LRAPD', 'RRAPD']].mean(axis=1)
df['AVG_FML'] = df[['LFML', 'RFML']].mean(axis=1)
df['AVG_FBL'] = df[['LFBL', 'RFBL']].mean(axis=1)
df['AVG_FEB'] = df[['LFEB', 'RFEB']].mean(axis=1)
df['AVG_FAB'] = df[['LFAB', 'RFAB']].mean(axis=1)
df['AVG_FHD'] = df[['LFHD', 'RFHD']].mean(axis=1)
df['AVG_FMLD'] = df[['LFMLD', 'RFMLD']].mean(axis=1)
df['AVG_FAPD'] = df[['LFAPD', 'RFAPD']].mean(axis=1)
df['AVG_TML'] = df[['LTML', 'RTML']].mean(axis=1)
df['AVG_TPB'] = df[['LTPB', 'RTPB']].mean(axis=1)
df['AVG_TMLD'] = df[['LTMLD', 'RTMLD']].mean(axis=1)
df['AVG_TAPD'] = df[['LTAPD', 'RTAPD']].mean(axis=1)
df['AVG_IBL'] = df[['LIBL', 'RIBL']].mean(axis=1)
df['AVG_AcH'] = df[['LAcH', 'RAcH']].mean(axis=1)

print("\nNew features added! Total columns:", df.shape[1])

# Clean dataset save කරන්න
df.to_csv(r'D:\Project\Bone_Gender_Age_Project\dataset\cleaned_data.csv', index=False)
print("\nCleaned data saved!")