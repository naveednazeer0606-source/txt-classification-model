import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- STEP 1: EXPANDED DATASET (4 CATEGORIES) ---
# We now have: Spam, Social, News, Sentiment
data = {
    'text': [
        # SPAM
        "Free money!!! Click here now to claim your prize.",
        "URGENT! You have won a guaranteed $1000 cash. Call now!",
        "Win a free iPhone 14 Pro! Reply WIN to this message.",
        "Limited time offer! Buy 1 get 1 free.",
        "You have won a lottery! Claim immediately.",
        
        # SOCIAL (Chat/Ham)
        "Meeting at 3pm today, please don't be late.",
        "Can we reschedule our dinner to tomorrow?",
        "Hey, how are you doing? Long time no see.",
        "Don't forget to buy milk on your way home.",
        "Let's catch up for coffee this weekend.",
        
        # NEWS (Factual/Headlines)
        "The stock market hit a record high today amid tech rally.",
        "Government announces new budget for infrastructure projects.",
        "Scientists discover water on a distant planet.",
        "Local elections are scheduled for next month.",
        "Sports update: The team won the championship last night.",
        
        # SENTIMENT (Emotional/Feedback)
        "I absolutely love this product, it is amazing!",
        "I am feeling very sad and lonely today.",
        "This service is terrible, I am never coming back.",
        "What a beautiful day, I feel so happy!",
        "I am angry about the delay in my delivery."
    ],
    'label': [
        "spam", "spam", "spam", "spam", "spam",
        "social", "social", "social", "social", "social",
        "news", "news", "news", "news", "news",
        "sentiment", "sentiment", "sentiment", "sentiment", "sentiment"
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

print("--- Week 2: Multi-Class Text Classifier ---")
print(f"Categories: {df['label'].unique()}")
print(f"Dataset Size: {len(df)} examples\n")

# --- STEP 2: PREPROCESSING (TF-IDF) ---
print("1. Vectorizing text...")
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['text'])
y = df['label']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- STEP 3: TRAIN MODEL ---
print("2. Training Multi-Class Naive Bayes...")
model = MultinomialNB()
model.fit(X_train, y_train)

# --- STEP 4: EVALUATION ---
y_pred = model.predict(X_test)

print("\n--- MODEL PERFORMANCE ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# --- VISUALIZATION ---
print("3. Generating Confusion Matrix...")
unique_labels = sorted(df['label'].unique()) # Sort alphabetically: news, sentiment, social, spam
cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
            xticklabels=unique_labels, yticklabels=unique_labels)
plt.title('Confusion Matrix: 4-Class Classification')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig("confusion_matrix.png")
plt.show()

# --- STEP 5: INTERACTIVE TEST ---
print("\n--- INTERACTIVE MODE ---")
print("Try typing: 'The president spoke today' OR 'I hate this service'")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("Enter text: ")
    if user_input.lower() == 'exit':
        break
    
    user_vector = tfidf.transform([user_input])
    prediction = model.predict(user_vector)[0]
    
    print(f">>> Classified as: [{prediction.upper()}]")