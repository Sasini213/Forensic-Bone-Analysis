import pandas as pd
import numpy as np

df = pd.read_csv(r'D:\Project\Bone_Gender_Age_Project\dataset\cleaned_data.csv')

print("Original shape:", df.shape)
print("Original Age distribution:\n", df['Age_Group'].value_counts())
print("Original Sex distribution:\n", df['Sex'].value_counts())

np.random.seed(42)

def generate_samples(group_df, n_samples):
    synthetic = pd.DataFrame()
    for col in group_df.columns:
        if col in ['Sex', 'Age', 'Age_Group']:
            synthetic[col] = group_df[col].sample(n_samples, replace=True).values
        else:
            mean = group_df[col].mean()
            std = max(group_df[col].std(), 0.01)
            values = np.random.normal(mean, std * 0.15, n_samples)
            synthetic[col] = np.clip(values, 0, None)
    return synthetic

all_synthetic = []

# හැම age group + sex combination එකකටම generate කරන්න
for age_group in df['Age_Group'].unique():
    for sex in df['Sex'].unique():
        group = df[(df['Age_Group'] == age_group) & (df['Sex'] == sex)]
        if len(group) > 5:
            synthetic = generate_samples(group, 300)
            all_synthetic.append(synthetic)
            print(f"Generated 300 samples for Age:{age_group} Sex:{sex}")

df_synthetic = pd.concat(all_synthetic, ignore_index=True)
df_combined = pd.concat([df, df_synthetic], ignore_index=True)

print("\nNew shape:", df_combined.shape)
print("New Age distribution:\n", df_combined['Age_Group'].value_counts())
print("New Sex distribution:\n", df_combined['Sex'].value_counts())

df_combined.to_csv(
    r'D:\Project\Bone_Gender_Age_Project\dataset\cleaned_data_augmented.csv',
    index=False
)
print("\nDone! New dataset saved!")