Apresentação e Artigo

- O arquivo Apresentação Trabalho Final TP555.pdf tem a apresentação feita em sala de aula.
- Relatório da pesquisa (artigo): Previsao_de_probabilidade_de_diabetes_em_estagio_inicial_usando_tecnicas_de_mineracao_de_dados-Relatorio.pdf

Resultados

O diretório codigos tem os códigos da API apresentada em sala:

- diabetes_rf_model.pkl arquivo com o modelo treinado com Random Forests
- diabetes_rf_model_neuralnetfastai.pkl arquivo com o modelo treinado com o Neural Net Fast AI
- index.html (frontend) interface gráfica (formulário)
- styles.css arquivo de estilos utilizado pela interface gráfica
- predict_app.py (backend) arquivo que contém a lógica para utilizar tanto o modelo treinado com Random Forest quanto com Neural Net Fast AI.
- teste_predict_api.py arquivo (backend) que contém a lógica para utilizar o modelo treinado com Random Forest.

Para rodar a API, basta executar o código predict_app.py em um ambiente python 3.8 (versão utilizada no meu experimento). Depois basta abrir o arquivo index.html para utilizar o aplicativo. Para tudo funcionar corretamente, todos os arquivos devem estar no mesmo diretório.

O diretório reproducao_artigo tem os códigos utilizados pela equipe que desenvolveu o trabalho que inspirou a pesquisa apresentada em sala (Diabetes Risk Prediction.ipynb). Além disso, neste diretório estão os códigos que utilizei com o autoGluon para explorar diversos modelos a fim de encontrar um melhor que a Floresta Aleatória (Encontrando o Melhor algoritmo com AutoGluon.ipynb). Além disso, é neste diretório que está o código que utilizei para treinar o modelo que eu uso no backend da API (Treinando o modelo para usar no java.ipynb). Por fim, este diretório também contém o arquivo que utilizei para treinar o modelo de Neural Net Fast AI (ModeloComNeural.ipynb).
