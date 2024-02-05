import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Wczytanie danych
try:
    dane_dot = pd.read_excel('C:\\Users\\michu\\Desktop\\WAT\\SEMESTR-7\\MED\\Lab3\\dane_dot.xlsx', header=None)
    dane_nie_dot = pd.read_excel('C:\\Users\\michu\\Desktop\\WAT\\SEMESTR-7\\MED\\Lab3\\dane_niedot.xlsx', header=None)
except FileNotFoundError:
    print("Nie można odnaleźć plików Excel. Sprawdź, czy ścieżki są poprawne.")

def preprocess(text, lang='en'):
    if text is not None:
        return text.lower()
    return ""

# Oznaczanie klas
dane_dot['label'] = 1
dane_nie_dot['label'] = 0

# Łączenie danych
dane = pd.concat([dane_dot, dane_nie_dot])
dane[0] = dane[0].apply(preprocess)

# Podział na zbiory
X_train, X_val, y_train, y_val = train_test_split(dane[0], dane['label'], test_size=0.3, random_state=42)

# Usuwanie słów stopu
stop_words = 'english'  # Zmiana na łańcuch znakowy
vectorizer = CountVectorizer(stop_words=stop_words,lowercase = True)
X_train_counts = vectorizer.fit_transform(X_train)
X_val_counts = vectorizer.transform(X_val)


# Tworzenie i trening modelu
model = MultinomialNB()
model.fit(X_train_counts, y_train)

# Walidacja modelu
predictions = model.predict(X_val_counts)
accuracy = accuracy_score(y_val, predictions)
print(f'Dokładność modelu: {accuracy}')
print(dane)