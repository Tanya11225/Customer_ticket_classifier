# train.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import pandas as pd

# Import preprocessing function
from preprocess import load_and_clean_data


# Use the new dataset path
DATA_PATH = r'C:\Users\HP\Downloads\archive (2)\customer_support_tickets.csv'
df = pd.read_csv(DATA_PATH)

# Keep only 'text' and 'category' columns
if 'text' in df.columns and 'category' in df.columns:
	df = df[['text', 'category']]
else:
	raise ValueError('CSV must contain text and category columns')

# Preprocess if preprocess_text is available
try:
	from preprocess import preprocess_text
	df['cleaned_text'] = df['text'].apply(preprocess_text)
	X = df['cleaned_text']
except ImportError:
	X = df['text']
	print('preprocess_text not found, using raw text.')
y = df['category']


print("🧠 Training the model on all data with tuned TfidfVectorizer...")
# Tune TfidfVectorizer: add n-grams, limit max_features

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=3000)
model = make_pipeline(vectorizer, LogisticRegression(max_iter=200, class_weight='balanced'))
model.fit(X, y)

# Print classification report for diagnostics
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, 'ticket_classifier_model.pkl')

print("\n✅ Model Training Complete!")
print("🎉 Model trained on all available data.")
