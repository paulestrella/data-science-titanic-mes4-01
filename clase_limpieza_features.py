import pandas as pd


df = pd.read_csv("titanic.csv")

print("Forma del dataset:", df.shape)
print("\nPrimelas filas:")
print(df.head())
print("\nTipos de datos:")
print(df.dtypes)

# diagnosticar los valores nulos

# crear variables para ver nulos en valor absoluto y en porcentaje
nulos = df.isnull().sum()
porcentaje_nulos = (df.isnull().sum() / len(df) * 100).round(2)

print("\nValores Nulos:")
print(nulos)
print("\nPorcentaje de  datos nulos")
print(porcentaje_nulos)

# age
df['Age'] = df['Age'].fillna(df['Age'].median())

#embarked
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

#cabin
df = df.drop(columns=['Cabin'])
print("Cantidad de Nulos:", df.isnull().sum().sum())

# Eliminacion de valores nulos
print("Filas antes de verificar:", len(df))
df = df.drop_duplicates()
print("Filas despues de verificar:", len(df))

# cambiar tipo de datos
df['Pclass'] = df['Pclass'].astype('category')

print(df.dtypes)
print(df.isnull().sum())

# feature engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

print(df[['SibSp', 'Parch', 'FamilySize']].head())

# variables binarias 
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

print(df['IsAlone'].value_counts())

# extraer el titulo del nombre
df['Title'] = df['Name'].str.extract(r',\s*([A-Za-z]+)\.?\s')

print(df['Title'].value_counts())

# Unificar categorias raras
df['Title'] = df['Title'].replace({
    'Mlle': 'Miss',
    'Ms': 'Miss',
    'Mme': 'Mrs',
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
    'Capt': 'Rare', 'Sir': 'Rare', 'Lady': 'Rare',
    'the': 'Rare', 'Jonkheer': 'Rare', 'Don': 'Rare'
})

print(df['Title'].value_counts())

df['AgeGroup'] = pd.cut(
    df['Age'],
    bins=[0,12,18,35,60,100],
    labels=['Niño', 'Adolescentes', 'Adulto joven', 'Adulto', 'Mayor']
)

print(df[['Age', 'AgeGroup']].head(10))

print("DATASET FINAL - LIMPIO FEATURES NUEVOS")
print(f"\nForma Final:  {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"\nColumnas actuales: \n{list(df.columns)}")
print(f"\Valores nulos restantes: {df.isnull().sum().sum()}")
print(f"\nVista previa del dataset final:")
print(df.head())

#guardar el resultado en un archivo 
df.to_csv("titanic_clean.csv", index=False)
