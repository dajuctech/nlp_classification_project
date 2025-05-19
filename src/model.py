# src/model.py

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def get_models():
    models = {
        "SVM": SVC(kernel="rbf", C=10, probability=True, random_state=100),
        "Naive Bayes": MultinomialNB(alpha=1.0),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=100),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=100),
        "XGBoost": XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', use_label_encoder=False, random_state=100),
        "Voting": VotingClassifier(estimators=[
            ('svm', SVC(probability=True, kernel='linear', random_state=100)),
            ('rf', RandomForestClassifier(n_estimators=200, random_state=100)),
            ('nb', MultinomialNB()),
            ('lr', LogisticRegression(max_iter=1000, random_state=100))
        ], voting='soft')
    }
    return models
