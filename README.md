# **California Housing Prices Prediction App
### Web App: https://housing-prices-prediction.streamlit.app/

## 🔍 Sobre o projeto
 
Neste projeto, vamos utilizar técnicas de regressão linear com a biblioteca **scikit-learn** do **Python** para prever os preços de casas na Califórnia. 
O objetivo é entender quais características de uma casa (como número de quartos, área construída, localização etc.) influenciam o seu valor de mercado e em seguida, utilizar a **regressão linear** para modelar a relação entre as características da casa e o seu preço de mercado. Para avaliar a qualidade do modelo, vamos utilizar métricas como **R²** e **RMSE**.

Por fim, foi criado um aplicativo web, utilizando **Streamlit**, em que é possível prever o valor de um imóvel de acordo com dados fornecidos pelo usuário. Isso pode ser útil para profissionais do mercado imobiliário, investidores e até mesmo para pessoas que estão em busca de uma casa para comprar.

## Descrição dos dados

O *dataset* possui as seguintes variáveis:

- **Longitude**: longitude de um determinado conjunto de casas.
- **Latitude**: latitude de um determinado conjunto de casas.
- **Housing median age**: idade mediana das casas.
- **Total rooms**: total de quartos.
- **Total bedrooms**: total de quartos para dormir.
- **Population**: população da localidade.
- **Households**: número total de famílias, grupos de pessoas residindo em uma unidades domiciliar, por um quarteirão.
- **Median income**: renda mediana da localidade.
- **Ocean proximity**: Proximidade com o oceano (menos de uma hora para chegar no oceano; terrestre; perto do oceano; perto de uma baía; em uma ilha).
