import pandas as pd

# Charger les deux datasets
df_seed = pd.read_csv('Dataset_project_RS_random.csv')
df_full = pd.read_csv('Dataset_project_RS.csv')

# Vérification que les deux datasets ont des colonnes communes
common_columns = df_seed.columns.intersection(df_full.columns)

# Aligner les colonnes des deux datasets
df_seed_aligned = df_seed[common_columns]
df_full_aligned = df_full[common_columns]


# Sauvegarder les données manquantes dans un fichier CSV
df_full_aligned.to_csv('datatest.csv', index=False)

testdf = pd.read_csv('datatest.csv')
print(testdf.shape)
print(df.shape)


testdf = pd.read_csv('datatest.csv')
print(testdf.shape)

# 2. Suppression des colonnes spécifiques
testdf.drop(columns=["urgent", "num_outbound_cmds"], inplace=True)
testdf.drop(columns=zero_columns, inplace=True)

# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

testdf = pd.read_csv('datatest.csv')
print(testdf.shape)
print(testdf.columns)

# 2. Suppression des colonnes spécifiques
testdf.drop(columns=["urgent", "num_outbound_cmds"], inplace=True)


# 3. Suppression des colonnes avec plus de 80% de zéros
# Assure-toi que 'zero_columns' a été défini lors du traitement de 'df'
testdf.drop(columns=zero_columns, inplace=True)

# 4. Encodage des colonnes catégorielles avec le même OneHotEncoder
categorical_columns = ['protocol_type', 'service']
encoded_columns_test = one_hot_encoder.transform(testdf[categorical_columns])

# Transformer en dataframe et ajouter au dataframe testdf
encoded_df_test = pd.DataFrame(encoded_columns_test, columns=one_hot_encoder.get_feature_names_out(categorical_columns))
testdf = pd.concat([testdf.drop(columns=categorical_columns), encoded_df_test], axis=1)

# 5. Imputation des valeurs manquantes avec le même SimpleImputer
numeric_columns = testdf.select_dtypes(include=[np.number]).columns
testdf[numeric_columns] = imputer_num.transform(testdf[numeric_columns])

# 6. Suppression des lignes dupliquées
testdf.drop_duplicates(inplace=True)

# 7. Sélection des features avec le même VarianceThreshold
target_column = 'outcome'
X_test = testdf.drop(columns=[target_column])
y_test_actual = testdf[target_column]

# Application du même selector sur X_test
X_test_reduced = selector.transform(X_test)

# 8. Mise à l'échelle avec le même StandardScaler
X_test_scaled = scaler.transform(X_test_reduced)

# 9. Prédictions et évaluations des modèles sur testdf

# Régression Logistique
y_pred_lr_test = lr_model.predict(X_test_scaled)
print("=== Régression Logistique sur testdf ===")
print(f"Accuracy: {accuracy_score(y_test_actual, y_pred_lr_test):.4f}")
print(f"Classification Report:\n{classification_report(y_test_actual, y_pred_lr_test)}")

# K-Nearest Neighbors (KNN)
y_pred_knn_test = knn_model.predict(X_test_scaled)
print("=== KNN sur testdf ===")
print(f"Accuracy: {accuracy_score(y_test_actual, y_pred_knn_test):.4f}")
print(f"Classification Report:\n{classification_report(y_test_actual, y_pred_knn_test)}")

# Multi-Layer Perceptron (MLP)
y_pred_mlp_test = mlp_model.predict(X_test_scaled)
print("=== MLP sur testdf ===")
print(f"Accuracy: {accuracy_score(y_test_actual, y_pred_mlp_test):.4f}")
print(f"Classification Report:\n{classification_report(y_test_actual, y_pred_mlp_test)}")

# Decision Tree
y_pred_dt_test = dt_model.predict(X_test_scaled)
print("=== Decision Tree sur testdf ===")
print(f"Accuracy: {accuracy_score(y_test_actual, y_pred_dt_test):.4f}")
print(f"Classification Report:\n{classification_report(y_test_actual, y_pred_dt_test)}")

# Random Forest
y_pred_rf_test = rf_model.predict(X_test_scaled)
print("=== Random Forest sur testdf ===")
print(f"Accuracy: {accuracy_score(y_test_actual, y_pred_rf_test):.4f}")
print(f"Classification Report:\n{classification_report(y_test_actual, y_pred_rf_test)}")

# Gradient Boosting
y_pred_gb_test = gb_model.predict(X_test_scaled)
print("=== Gradient Boosting sur testdf ===")
print(f"Accuracy: {accuracy_score(y_test_actual, y_pred_gb_test):.4f}")
print(f"Classification Report:\n{classification_report(y_test_actual, y_pred_gb_test)}")

# Support Vector Machine (SVM)
y_pred_svm_test = svm_model.predict(X_test_scaled)
print("=== SVM sur testdf ===")
print(f"Accuracy: {accuracy_score(y_test_actual, y_pred_svm_test):.4f}")
print(f"Classification Report:\n{classification_report(y_test_actual, y_pred_svm_test)}")

# 10. Comparaison des modèles sur testdf
print("=== Comparaison des Modèles sur testdf ===")
print(f"Logistic Regression Accuracy: {accuracy_score(y_test_actual, y_pred_lr_test):.4f}")
print(f"KNN Accuracy: {accuracy_score(y_test_actual, y_pred_knn_test):.4f}")
print(f"MLP Accuracy: {accuracy_score(y_test_actual, y_pred_mlp_test):.4f}")
print(f"Decision Tree Accuracy: {accuracy_score(y_test_actual, y_pred_dt_test):.4f}")
print(f"Random Forest Accuracy: {accuracy_score(y_test_actual, y_pred_rf_test):.4f}")
print(f"Gradient Boosting Accuracy: {accuracy_score(y_test_actual, y_pred_gb_test):.4f}")
print(f"SVM Accuracy: {accuracy_score(y_test_actual, y_pred_svm_test):.4f}")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Préparation des données
X = df.drop(columns=['outcome'])  # Remplacer par le nom de ta colonne cible
y = df['outcome']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normaliser les données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialiser les modèles
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'SVM': SVC(),
    'MLP Classifier': MLPClassifier(max_iter=600, random_state=42)
}

# Évaluer chaque modèle avec validation croisée
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"{name}: Moyenne accuracy {cv_scores.mean():.4f}, Écart-type {cv_scores.std():.4f}")

# Entraîner et évaluer chaque modèle sur le jeu de test
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name}: Accuracy sur le jeu de test: {acc:.4f}")
