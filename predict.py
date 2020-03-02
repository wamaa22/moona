from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
nb_run = 3

models = [
    SoftmaxClassifier(), # le modele que vous avez implémenté plus haut
    LogisticRegression(),
    KNeighborsClassifier(),
    GaussianNB(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier()
]

scoring = ['neg_log_loss', 'precision_macro','recall_macro','f1_macro']

compare(models ,X_train_preprocess ,y_train ,nb_run, scoring)