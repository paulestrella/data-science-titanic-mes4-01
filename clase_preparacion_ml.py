import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

pd.options.future.infer_string = False

df = pd.read_csv("titanic_clean.csv")

df['Pclass'] = df['Pclass'].astype('category')

orden_edad = ['Niño', 'Adolescentes', 'Adulto joven', 'Adulto', 'Mayor']
df['AgeGroup'] = pd.Categorical(df['AgeGroup'], categories=orden_edad, ordered=True)

print("Dataset cargado:", df.shape)
print("Pclass dtype:", df['Pclass'].dtype)
print("AgeGroup dtype:", df['AgeGroup'].dtype)

# Columnas utiles

columnas_a_descartar = ['Name', 'PassengerId', 'Ticket']

df_ml = df.drop(columns=columnas_a_descartar)
print("Columnas que mantuvimos para el ML")
print(list(df_ml.columns))
print("\nNueva Forma:", df_ml.shape)

# Varibles "X" e "y"
X = df_ml.drop(columns=['Survived'])
y = df_ml['Survived']

print("La variable 'X' son los features que el modelo usara para aprender")
print("   shape:", X.shape)
print("   Columnas:", list(X.columns))
print()
print("La varible 'y' es la variable que el modelo intentara predecir")
print("   shape:", y.shape)
print("   valores unicos:", y.unique())
print("   Distribucion:")
print(y.value_counts())

# train/test Split "PARTE MAS IMPORTANTE"

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Tamaños del split:")
print(f" X_train: {X_train.shape} - informacion con la que el modelo aprende")
print(f" X_test:  {X_test.shape} - informacion con la que se evalua el modelo")
print(f" y_train: {y_train.shape}")
print(f" y_test:  {y_test.shape}")
print()
print("Proporcion de supervivientes en train", round(y_train.mean(), 4))
print("Proporcion de supervivientes en test", round(y_test.mean(), 4))

# Encoding de variables categoricas
columnas_categoricas = ['Pclass', 'Sex', 'AgeGroup', 'Embarked', 'Title']

print("columnas a codificar:", columnas_categoricas)
print()

X_train_enc = pd.get_dummies(X_train, columns=columnas_categoricas, drop_first=False)
X_test_enc = pd.get_dummies(X_test, columns=columnas_categoricas, drop_first=False)


# alinear columnas
X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

# COMPARACION
print("columnas ANTES del encoding:", X_train.shape[1])
print("columnas DESPUES del encoding:", X_train_enc.shape[1])
print()

print("Nueva Info de las columnas:")
print(list(X_train.columns))
print()
print(list(X_train_enc.columns))

# escalar variables numericas

columnas_numericas = ['Age', 'SibSp', 'Parch', 'Fare', 'Fare', 'FamilySize', 'IsAlone']

scaler = StandardScaler()

X_train_enc[columnas_numericas] = scaler.fit_transform(X_train_enc[columnas_numericas])

X_test_enc[columnas_numericas] = scaler.transform(X_test_enc[columnas_numericas])

print("Estadisticas de columna numerica X_train:")
print(X_train_enc[columnas_numericas].describe().round(3))
print()
print("media de age en X_train", round(X_train_enc['Age'].mean(), 4))
print("Std de age en x_train", round(X_train_enc['Age'].std(), 4))
print()
print("media de age en X_test:", round(X_test_enc['Age'].mean(), 4))

#columnas booleanas a int
bool_cols = X_train_enc.select_dtypes(bool).columns
X_train_enc[bool_cols] = X_train_enc[bool_cols].astype(int)
X_test_enc[bool_cols] = X_test_enc[bool_cols].astype(int)

print("X_train:", X_train_enc.shape)
print("X_test:", X_test_enc.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print(f"supervivientes en y_train: {y_train.sum()} de {len(y_train)}")
print(f"supervivientes en y_test: {y_test.sum()} de {len(y_test)}")

print("Nulos en X_train:", X_train_enc.isnull().sum().sum())
print("Nulos en X_test:", X_test_enc.isnull().sum().sum())

print("tipos de datos numeros en X_train")
print(X_train_enc.dtypes.value_counts())
print(list(X_train_enc.columns))

X_train_enc.to_csv("X_train.csv", index=False)
X_test_enc.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

