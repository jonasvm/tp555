# Previsão de Probabilidade de Diabetes em Estágio Inicial usando Técnicas de Mineração de Dados

Este projeto aborda a previsão de probabilidade de diabetes em estágio inicial utilizando técnicas de mineração de dados e diferentes algoritmos de aprendizado de máquina. Ele inclui tanto os códigos para reproduzir os resultados do trabalho "Likelihood Prediction of Diabetes at Early Stage Using Data Mining Techniques" apresentados em sala de aula, quanto a aplicação desenvolvida para demonstração.

## Estrutura do Projeto

### 1. Apresentação e Relatório

- Apresentação Trabalho Final TP555.pdf: Apresentação realizada em sala de aula.
- Previsao_de_probabilidade_de_diabetes_em_estagio_inicial_usando_tecnicas_de_mineracao_de_dados-Relatorio.pdf: Relatório do trabalho, em formato de artigo.

### 2. Diretório codigos

Contém os arquivos da API desenvolvida para a demonstração prática:

    diabetes_rf_model.pkl: Modelo treinado utilizando Random Forests.
    diabetes_rf_model_neuralnetfastai.pkl: Modelo treinado utilizando Redes Neurais com a biblioteca FastAI.
    index.html: Interface gráfica do usuário (frontend).
    styles.css: Arquivo de estilos para a interface gráfica.
    predict_app.py: Lógica do backend que utiliza os modelos treinados (Random Forests e Neural Net FastAI) para fazer previsões.
    teste_predict_api.py: Código backend para testar o modelo treinado com Random Forests.

#### Como executar a API

    Certifique-se de ter o Python 3.8 instalado no ambiente.
    Coloque todos os arquivos do diretório codigos no mesmo local.
    Execute o arquivo predict_app.py:

    python predict_app.py

    Abra o arquivo index.html no navegador para acessar a interface gráfica.

### 3. Diretório reproducao_artigo

Inclui os códigos utilizados para explorar modelos e reproduzir os resultados da pesquisa:

    Diabetes Risk Prediction.ipynb: Código original utilizado pelos autores do artigo que inspirou este trabalho.
    Encontrando o Melhor algoritmo com AutoGluon.ipynb: Exploração de diferentes modelos com AutoGluon para identificar o melhor desempenho em comparação ao Random Forest.
    Treinando o modelo para usar no java.ipynb: Código usado para treinar o modelo aplicado no backend da API.
    ModeloComNeural.ipynb: Código para treinar o modelo com Redes Neurais utilizando a biblioteca FastAI.

### Objetivos do Projeto

    Explorar diferentes algoritmos de aprendizado de máquina para prever a probabilidade de diabetes em estágio inicial.
    Desenvolver uma API funcional, com frontend e backend, para realizar previsões em tempo real.
    Reproduzir os resultados do trabalho que inspirou esta pesquisa, garantindo reprodutibilidade e validação científica.

### Requisitos

    Python 3.8 (para execução do backend)
    Bibliotecas necessárias: Listadas em cada notebook ou arquivo de código, incluindo:
        FastAI
        AutoGluon
        Flask (para o backend da API)
