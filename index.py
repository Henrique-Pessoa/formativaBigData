import pandas as pd
import numpy as np  
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, precision_score,f1_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

data = pd.read_csv("./dados_produtos.csv")

print(data.isna())
print(data.isnull())

plt.figure(figsize=(10, 6))
sns.boxplot(x='product_name', y='rating', data=data)
plt.xlabel('Produto')
plt.ylabel('Rating')
plt.show()



trainData, testeData = train_test_split(data, test_size=0.2, random_state=42)

produtos = data['product_name'].unique()

for produto in produtos:
    dados_produto = trainData[trainData['product_name'] == produto]
    
    X = dados_produto[['rating', 'rating_count']]
    y = dados_produto['purchased']
    
    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X, y)
    
    dados_teste_produto = testeData[testeData['product_name'] == produto]
    
    X_teste = dados_teste_produto[['rating', 'rating_count']]
    y_teste = dados_teste_produto['purchased']
    
    y_pred = modelo.predict(X_teste)
    
    precision = precision_score(y_teste, y_pred)
    recall = recall_score(y_teste, y_pred)
    f1 = f1_score(y_teste, y_pred)
    print(f"PRODUCT: {produto}")
    print("Precision: %f" % precision)
    print("recall: %f" % recall)
    print("F1: %f" % f1)

    plt.figure(figsize=(12, 8))
    plot_tree(modelo, filled=True, feature_names=['rating', 'rating_count'], class_names=['Not Purchased', 'Purchased'])
    plt.title(f'Árvore de Decisão para {produto}')
    plt.show()


#>>>>> Conclusão <<<<<
#


