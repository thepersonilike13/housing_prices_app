**<h1 align='center'> Housing Prices Prediction App </h1>**

<p align="center">
<img src="https://user-images.githubusercontent.com/69912320/234723379-21eab3af-67c2-45e7-97d1-19613de4cfc0.gif" alt="animated" />
</p>

## Web app: https://housingprice.streamlit.app/

## 🔍 Sobre o projeto
 
Neste projeto, vamos utilizar técnicas de regressão linear com a biblioteca **scikit-learn** do **Python** para prever os preços de habitações na Califórnia. O objetivo é entender quais características de um imóvel residencial (como número de quartos, localização, renda da localidade etc.) influenciam o seu valor de mercado.

O conjunto de dados utilizado foi [retirado do Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices) e é referente ao **censo de 1990**. Utilizaremos técnicas de limpeza e pré-processamento dos dados para que eles estejam prontos para a análise.

Em seguida, vamos utilizar a **regressão linear** para modelar a relação entre as características da casa e o seu preço de mercado. Para avaliar a qualidade do modelo, vamos utilizar métricas como **R²** e **RMSE**.

Por fim, foi criado um aplicativo web, utilizando **Streamlit**, em que é possível prever o valor de um imóvel de acordo com dados fornecidos pelo usuário. Vamos analisar os resultados e entender a visão de negócio do projeto, podendo ser útil para auxiliar profissionais do mercado imobiliário, investidores e até mesmo para pessoas que estão em busca de uma casa para comprar.

## Descrição dos dados

O *dataset* possui as seguintes variáveis:

- **`longitude`**: longitude de um determinado conjunto de casas.
- **`latitude`**: latitude de um determinado conjunto de casas.
- **`housing_median_age`**: idade mediana das casas em um quarteirão.
- **`total_rooms`**: total de quartos em um quarteirão de casas.
- **`total_bedrooms`**: total de quartos para dormir em um quarteirão.
- **`population`**: população da localidade em um quarteirão.
- **`households`**: número total de famílias, grupos de pessoas residindo em uma unidades domiciliar, por um quarteirão.
- **`median_income`**: renda mediana em um quarteirão.
- **`ocean_proximity`**: Proximidade com o oceano (menos de uma hora para chegar no oceano; terrestre; perto do oceano; perto de uma baía; em uma ilha).

## Etapas do projeto

1. Importar os dados e as bibliotecas
2. Entender os dados e seus tipos
3. Análise Exploratória
4. *Feature Engineering*
5. Modelagem
6. *Deploy*
