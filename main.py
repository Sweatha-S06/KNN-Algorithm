import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load Data
df = pd.read_csv('Iris.csv')
print("Original Data Loaded:", df.shape)

# Data Cleaning
df = df.drop_duplicates()
df = df.fillna(df.median(numeric_only=True))
df = df.fillna(df.mode().iloc[0])
print("After Cleaning:", df.shape)

# Drop ID column if present
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

# Convert Species (String → Number)
for c in df.select_dtypes(include='object'):
    df[c] = LabelEncoder().fit_transform(df[c])

print("After Encoding:")
print(df.head())

# Feature Selection
corr = df.corr()['Species'].abs()
features = corr[corr > 0.1].index.drop('Species')

print("Correlations:\n", corr)
print("Selected Features:", list(features))

# Split Data
x = df[features]
y = df['Species']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Train & Evaluate Model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Accuracy: {:.2f}%".format(
    accuracy_score(y_test, y_pred) * 100
))

# Save Model
joblib.dump(model, 'iris_model.pkl')
print("iris_model.pkl file created successfully!")