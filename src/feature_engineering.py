# src/feature_engineering.py

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def apply_count_vectorizer(text_series, ngram_range=(1, 1)):
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    X = vectorizer.fit_transform(text_series)
    print(f"✅ CountVectorizer shape: {X.shape}")
    return X, vectorizer

def apply_tfidf_vectorizer(text_series, ngram_range=(1, 2), max_features=5000):
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
    X = vectorizer.fit_transform(text_series)
    print(f"✅ TF-IDF shape: {X.shape}")
    return X, vectorizer

def reduce_with_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X.toarray())
    return X_reduced

def reduce_with_tsne(X, n_components=2, perplexity=30, random_state=42):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    X_reduced = tsne.fit_transform(X.toarray())
    return X_reduced
