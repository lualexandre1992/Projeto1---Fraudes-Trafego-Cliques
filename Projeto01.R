
## Projeto com Feedback 1 - Detecção de Fraudes no Tráfego de Cliques em Propagandas de Aplicações Mobile


# Configurando o diretório de trabalho
setwd("F:/Users/Luana/OneDrive/Documentos/DSA/BigDataRAzure/ProjetoFeedback01")
getwd()

### Etapa 0 - Carregando os pacotes

library(readr)           # [readr] para leitura de grandes arquivos de dados
library(Amelia)          # Pacote para análise de dados faltantes
library(corrplot)        # Pacote para análise de correlação
library(caret)           # Pacote para preprocessamento dos dados
library(ROSE)            # Pacote para balanceamento das classes
library(e1071)           # Pacote para aplicação do Naive Bayes
library(ROCR)            # Pacote para avaliação da curva ROC
library(caTools)
library(dplyr)       

### Etapa 1 - Coletando os Dados

dados_treino <- read_csv("train.csv", n_max = 2000000)
amostra.df <- read_csv("train_sample.csv")

### Etapa 2 - Analisando informações do dataset 

head(dados_treino)

#is.na(dados_treino)

# Tipos de Dados

summary(dados_treino)
str(dados_treino)


# Contagem de valores únicos por coluna 

for (n in colnames(dados_treino)){
  print(paste(n, ': ', n_distinct(dados_treino[n])))
}


n_distinct(as.Date(dados_treino$click_time))
unique(as.Date(dados_treino$click_time)) # Os dados correspondem aos cliques realizados entre os dias 06/11/2017 e 09/11/2017

# Análise descritiva da base (entender cada atributo e ver distribuição dos domínios, identificar variável target e avaliar desbalanceamento)

# Frequencia de cada atributo

prop.table(table(dados_treino$is_attributed)) # Está com forte desbalanceamento
prop.table(table(dados_treino$channel))
prop.table(table(dados_treino$is_attributed))
prop.table(table(dados_treino$os))
prop.table(table(dados_treino$device))
prop.table(table(dados_treino$app))
prop.table(table(dados_treino$dia_semana))


# Verificando o desbalanceamento na variável dependente por meio de um plot de barras
plot(as.factor(dados_treino$is_attributed))

# Fazendo o boxplot das variaveis independentes X variável v

boxplot(data = dados_treino,   ip ~ is_attributed)      # Indício da presença de outliers quando is_attributed = 0
boxplot(data = dados_treino,   app ~ is_attributed)     # Indício da presença de outliers em ambos os casos
boxplot(data = dados_treino,   device ~ is_attributed)  # Indício da presença de outliers em ambos os casos
boxplot(data = dados_treino,   os ~ is_attributed)      # Indício da presença de outliers em ambos os casos
boxplot(data = dados_treino,   channel ~ is_attributed)           # Sem indício de outliers
boxplot(data = dados_treino,   click_time ~ is_attributed)        # Sem indício de outliers e equilibrio entre 0 e 1
boxplot(data = dados_treino,   attributed_time ~ is_attributed)   # Como esperado, somente apresenta valores quando a varíavel dependente é igual a 1


### Analisando a Correlação entre as variáveis

# Obtendo apenas as colunas numéricas
colunas_numericas <- sapply(dados_treino, is.numeric)
colunas_numericas

# Filtrando as colunas numéricas para correlação
data_cor <- cor(dados_treino[,colunas_numericas])

head(data_cor)

corrplot(data_cor, method = 'color')
#existe forte correlação entre as variáveis [device], [os] e [app]


### Etapa 3 - Preparação dos dados

## Convertendo as variáveis para o tipo categórico

dados_treino$ip <- as.factor(dados_treino$ip)
dados_treino$app <- as.factor(dados_treino$app)
dados_treino$device <- as.factor(dados_treino$device)
dados_treino$os <- as.factor(dados_treino$os)
dados_treino$channel <- as.factor(dados_treino$channel)
dados_treino$is_attributed <- as.factor(dados_treino$is_attributed)


# Transformando a variável click_time  e attributed_time em data

dados_treino$click_day <- as.Date(dados_treino$click_time, format="%Y-%m-%d %H:%M:%S")
dados_treino$dia_semana_click <- weekdays(dados_treino$click_day)

# Excluindo colunas

dados_treino$click_time <- NULL

#criar amostras de forma randômica
# Criando dados de treino e dados de teste
divisao <- sample.split(dados_treino$app, SplitRatio = 0.70) #cria indice
div_treino <- subset(dados_treino, divisao == TRUE) #70%
div_teste <- subset(dados_treino, divisao == FALSE) #30%


# Fazendo o balanceamento das classes
treino_over <- ovun.sample(is_attributed ~ ip + app + device + os + channel, method = "both", data = div_treino)$data


# Verificando o resultado do balanceamento na variável dependente
plot(treino_over$is_attributed)

## Etapa 4 - Treinando o modelo 

# Criando modelos preditivos baseados em Naive Bayes
modeloV1 <- naiveBayes(is_attributed ~ ., 
                       data = div_treino)

# Criando modelos preditivos baseados em Naive Bayes
modeloV2 <- naiveBayes(is_attributed ~ ip + app + channel + click_day, 
                       data = div_treino)

# Criando modelo com os dados balanceados
modelo_balance <- naiveBayes(is_attributed ~ ., 
                             data = treino_over)

# Criando vetor para segundo teste
p2 <- c('ip',  'app' , 'channel' , 'click_day')

# Realizando as previsões com os modelos
previ1 <- predict(modeloV1, div_treino)
previ2 <- predict(modeloV2, div_treino[,p2])
previ_balance <- predict(modelo_balance, div_treino)

# Criando a Confusion Matrix com as previsões
table(pred = previ1, true = div_treino$is_attributed)
table(pred = previ2, true = div_treino$is_attributed)
table(pred = previ_balance, true = div_treino$is_attributed)


# Verificando a média de acertos entre as previsões e o grupo de treino
mean(previ1 == div_treino$is_attributed)
mean(previ2 == div_treino$is_attributed)
mean(previ_balance == div_treino$is_attributed)


## O modeloV1 apresentou melhor previsão de resultados para a classe positiva (0)
## Esse alto % de assertividade pode gerar um modelo com overfitting, isso se deve ao desbalanceamento das classes
## Contudo, esse modelo ainda é o melhor dentro o 3 testados, pois tem maior assertividade da classe negativa (1) maior % de Specificity


## Etapa 5 - Avaliando e Interpretando o Modelo

# Realizando a previsão com base no sample de testes
previsao <- predict(modeloV1, div_teste)

# Criando a Confusion Matrix
cM <- confusionMatrix(div_teste$is_attributed, previsao)
cM
cM$byClass["Precision"]   # True Positive / (True Positive + False Positive)
cM$byClass["Recall"]      # True Positive / (True Positve + False Negative )
cM$byClass["F1"]          # 2 * (Precision * Recall) / (Precision + Recall)

# Gerando uma curva ROC

pred <- prediction(as.numeric(previsao), div_teste$is_attributed)
perf <- performance(pred, "tpr","fpr")


plot(perf, col = rainbow(10), main = "Curva ROC")
abline(a=0, b=1)


# Interpretando o resultado
# O modelo previu corretamente 593281 vezes que o clique sem download
# O modelo previu corretamente 774 vezes o clique seguido de download
# O modelo previu erradamente 228 vezes que um clique teria download
# O modelo previu erradamente 5709 vezes que um clique não teria download

# Score
#Acurácia - Número total de previsões corretas comparado com o total da amostra:
 # 99.01%

#Recall - Número de acertos da classe positiva (0):
#  99.96%

#Precisão - Total de previsões corretas para classe positiva (0):
 # 99.04%

#Specificidade - Total de previsões corretas para classe negativa (1):
#  11.94%

#Taxa de Falso Positivo - Número de previsões incorretas para a classe positiva (0) dividido pelo total de membros da classe negativa (0):
 # 88.06%

#F1 Score - Média armônica entre a Precisão e a Acurácia:
#  99.50%

#Conclusão
#O modelo apresentou alta pontuação ao prever cliques sem download (Classe 0): 99,01% de precisão

#O mesmo não se pode dizer na performance de previsão para cliques com download (Classe 1): 11,94% de especificidade

#Além disso, apresentou uma alta taxa de falso positivo: 88,06%

#Tal comportamento é provavelmente fruto do desbalanceamento entre as classes: 99,83% dos dados de treinos são da Classe 0,

