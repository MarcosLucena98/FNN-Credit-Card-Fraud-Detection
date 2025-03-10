# Pacotes necessários
library(dplyr)
library(ggplot2)
library(caret)
library(readr)
library(keras)
library(tensorflow)

# Leitura dos dados
data <- read.csv(file = "creditcard.csv", header = TRUE)

# Verificar estrutura
str(data)
summary(data)


# Gerar um boxplot para todas as variáveis numéricas
library(reshape2)
data_long <- melt(data, id.vars = c("Class", "Time", "Amount"))

# Criando o boxplot aprimorado com ggplot2
jpeg("Distribuição_Variavel.jpg", width = 8, height = 6, units = "in", res = 600)
ggplot(data_long, aes(x = variable, y = value, fill = variable)) +
  geom_boxplot(outlier.colour = "red", outlier.size = 2, alpha = 0.7) +
  labs(
    title = "Distribuição das Variáveis Numéricas",
    x = "Variáveis",
    y = "Valores"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),  # Rotaciona os rótulos do eixo X
    legend.position = "none"  # Remove a legenda
  )
dev.off()

# Calculando a média de Amount
mean_amount <- mean(data$Amount, na.rm = TRUE)

# Exemplo de gráfico de dispersão entre Time e Amount
jpeg("Distribuição_Time_Amount_mean.jpg", width = 8, height = 6, units = "in", res = 600)
ggplot(data, aes(x = Time, y = Amount)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_hline(yintercept = mean_amount, linetype = "dashed", color = "red", size = 1) +
  labs(
    title = "Dispersão entre Tempo e Valor das Transações",
    x = "Tempo (segundos desde a primeira transação)",
    y = "Valor da Transação (Amount)"
  ) +
  annotate("text", x = max(data$Time), y = mean_amount, label = paste("Média =", round(mean_amount, 2)), 
           hjust = 1.1, vjust = -0.5, color = "red") +
  theme_minimal()
dev.off()

# Gráfico de linha para visualizar a variação de Amount ao longo do Time
jpeg("TRansação_Ao_Tempo.jpg", width = 8, height = 6, units = "in", res = 600)
ggplot(data, aes(x = Time, y = Amount)) +
  geom_line(alpha = 0.7, color = "darkgreen") +
  labs(
    title = "Variação do Valor das Transações ao Longo do Tempo",
    x = "Tempo (segundos desde a primeira transação)",
    y = "Valor da Transação (Amount)"
  ) +
  theme_minimal()
dev.off()


# Filtrando apenas as transações com fraude (Class = 1)
fraude_data <- subset(data, Class == 1)

# Calculando a média do valor das transações fraudulentas
mean_fraude_amount <- mean(fraude_data$Amount, na.rm = TRUE)

# Visualizando um resumo das transações fraudulentas
summary(fraude_data$Amount)

jpeg("Dispersão_Fraude.jpg", width = 8, height = 6, units = "in", res = 600)
# Criando o gráfico de dispersão com a linha de média
ggplot(fraude_data, aes(x = Time, y = Amount)) +
  geom_point(alpha = 0.5, color = "red") +
  geom_hline(yintercept = mean_fraude_amount, linetype = "dashed", color = "blue", size = 1) +
  labs(
    title = "Dispersão de Transações Fraudulentas com Linha de Média",
    x = "Tempo (segundos desde a primeira transação)",
    y = "Valor da Transação (Amount)"
  ) +
  annotate(
    "text", 
    x = max(fraude_data$Time) * 0.8, y = mean_fraude_amount + 50, 
    label = paste("Média =", round(mean_fraude_amount, 2)), 
    hjust = 0, vjust = -0.5, color = "blue", size = 5, fontface = "bold"
  ) +
  theme_minimal()
dev.off()

# Normalização dos dados retirando a target Class
data_normalized <- data %>% 
  mutate_at(vars(-Class), scale)

# Definir semente para reprodutibilidade
set.seed(123)

# Embaralhamento dos dados
data_shuffled <- data_normalized %>% sample_frac(1)

# Dividir em treino (80%) e teste (20%)
train_index <- 1:round(0.8 * nrow(data_shuffled))
train_data <- data_shuffled[train_index, ]
test_data <- data_shuffled[-train_index, ]

# Separação de características e variável alvo
train_x <- as.matrix(train_data %>% select(-Class))      
train_y <- as.matrix(train_data$Class)

test_x <- as.matrix(test_data %>% select(-Class))
test_y <- as.matrix(test_data$Class)

# Definir a Rede Neural com melhorias
model <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = 'relu', input_shape = ncol(train_x),kernel_regularizer = regularizer_l2(0.001)) %>%  # Camada oculta com 32 neurônios
  layer_dropout(0.3) %>%  # Adiciona dropout para evitar overfitting
  layer_dense(units = 16, activation = 'relu', kernel_regularizer = regularizer_l2(0.001)) %>%  # Segunda camada oculta
  layer_dropout(0.3) %>%  # Dropout na segunda camada
  layer_dense(units = 1, activation = 'sigmoid')   # Camada de saída (binária)

# Resumo do modelo
summary(model)

# Definir o callback de Early Stopping para evitar overfitting
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 3, restore_best_weights = TRUE)

# Durante a compilação, remova o 'class_weight'
model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.0001),
  loss = 'binary_crossentropy',
  metrics = c('accuracy')
)

# Defina os pesos das classes para o treinamento, por exemplo, 10 para fraudes (1) e 1 para legítimas (0)
class_weights <- list("0" = 1, "1" = 10)

# Treinamento com class_weight dentro do fit()
model %>% fit(
  train_x, train_y,
  epochs = 50,
  batch_size = 32,
  validation_data = list(test_x, test_y),
  class_weight = class_weights,  # Aqui é o lugar correto para o uso de class_weight
  callbacks = list(early_stop)
)


# Validação cruzada com K-Folds
k_folds <- 5
folds <- createFolds(train_data$Class, k = k_folds)

results <- lapply(folds, function(indices) {
  train_fold <- train_data[-indices, ]
  val_fold <- train_data[indices, ]
  
  train_x_fold <- as.matrix(train_fold %>% select(-Class))
  train_y_fold <- as.matrix(train_fold$Class)
  
  val_x_fold <- as.matrix(val_fold %>% select(-Class))
  val_y_fold <- as.matrix(val_fold$Class)
  
  history <- model %>% fit(
    train_x_fold, train_y_fold,
    epochs = 50,
    batch_size = 32,
    validation_data = list(val_x_fold, val_y_fold),
    callbacks = list(early_stop),  # Parar o treino cedo se a perda de validação não melhorar
    verbose = 0  # Modo silencioso para todas as iterações
  )
  
  list(history = history, val_data = val_fold)
})

# Extração dos resultados e gráfico
df_histories <- do.call(rbind, lapply(results, function(res) {
  history <- res$history
  data.frame(
    epoch = 1:length(history$metrics$loss),
    loss = history$metrics$loss,
    val_loss = history$metrics$val_loss,
    accuracy = history$metrics$accuracy,
    val_accuracy = history$metrics$val_accuracy
  )
}))

jpeg("perda_Treinamento_K_Folds.jpg", width = 8, height = 6, units = "in", res = 600)

# Gráficos de perdas e acurácia aprimorados
ggplot(df_histories, aes(x = epoch)) +
  geom_line(aes(y = loss, color = "Treinamento")) +
  geom_line(aes(y = val_loss, color = "Validação")) +
  labs(title = "Perda durante o Treinamento com K-Folds",
       x = "Época",
       y = "Perda") +
  scale_color_manual(name = "Conjunto de Dados", values = c("Treinamento" = "blue", "Validação" = "orange")) +
  theme_minimal()

dev.off()

getwd()

jpeg("acuracia_Treinamento_K_Folds.jpg", width = 8, height = 6, units = "in", res = 600)

ggplot(df_histories, aes(x = epoch)) +
  geom_line(aes(y = accuracy, color = "Treinamento")) +
  geom_line(aes(y = val_accuracy, color = "Validação")) +
  labs(title = "Acurácia durante o Treinamento com K-Folds",
       x = "Época",
       y = "Acurácia") +
  scale_color_manual(name = "Conjunto de Dados", values = c("Treinamento" = "blue", "Validação" = "orange")) +
  theme_minimal()

dev.off()

# Avaliação do modelo no conjunto de teste
score <- model %>% evaluate(test_x, test_y)
cat('Perda no conjunto de teste:', score[1], "\n")
cat('Acurácia no conjunto de teste:', score[2], "\n")

# Previsões no conjunto de teste
predictions <- model %>% predict(test_x)
predicted_classes <- ifelse(predictions > 0.3, 1, 0)

# Matriz de confusão
conf_matrix <- confusionMatrix(factor(predicted_classes), factor(test_data$Class))
print(conf_matrix)

# Transformando a matriz de confusão em dataframe
conf_matrix_df <- as.data.frame(conf_matrix$table)
colnames(conf_matrix_df) <- c("Previsto", "Real", "Freq")

# Plotando a matriz de confusão com ggplot2
jpeg("Matriz_de_Confusão.jpg", width = 8, height = 6, units = "in", res = 600)

ggplot(data = conf_matrix_df, aes(x = Previsto, y = Real, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), vjust = 1) +
  scale_fill_gradient(low = "lightblue", high = "blue") +
  labs(title = "Matriz de Confusão", x = "Valores Preditos", y = "Valores Reais") +
  theme_minimal()

dev.off()

# Métricas adicionais
accuracy <- conf_matrix$overall['Accuracy']
sensitivity <- conf_matrix$byClass['Sensitivity']
specificity <- conf_matrix$byClass['Specificity']

# Resultados
cat("Acurácia:", accuracy, "\n")
cat("Sensibilidade (Taxa de detecção de fraudes):", sensitivity, "\n")
cat("Especificidade (Taxa de verdadeiros negativos):", specificity, "\n")

library(knitr)

# Criação da tabela com as métricas
metrics <- data.frame(
  Métrica = c("Acurácia", "Sensibilidade", "Especificidade"),
  Valor = c(accuracy, sensitivity, specificity)
)

# Visualizando a tabela
kable(metrics, col.names = c("Métrica", "Valor"), caption = "Métricas de Desempenho do Modelo")

# Plotando o gráfico de barras
jpeg("Desempenho do Modelo.jpg", width = 8, height = 6, units = "in", res = 600)

ggplot(metrics, aes(x = Métrica, y = Valor, fill = Métrica)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_text(aes(label = round(Valor, 4)), vjust = -0.5) +
  labs(title = "Desempenho do Modelo", y = "Valor", x = "Métrica") +
  theme_minimal() +
  scale_fill_manual(values = c("Acurácia" = "steelblue", "Sensibilidade" = "green", "Especificidade" = "red"))

dev.off()
################################################################################################################
############# CURVA ROC e a AUC ###############################

# Calcular as previsões como probabilidades
pred_probs <- model %>% predict(test_x)

# Corrigindo a variável de previsão para ser um vetor
pred_probs <- as.vector(model %>% predict(test_x))

############################################
#Gráfico Curva ROC e AUC

# Gerar e plotar a curva ROC
library(pROC)

# Criando a curva ROC com o vetor corrigido
roc_curve <- roc(test_data$Class, pred_probs)

# Ajustar o tamanho da janela gráfica
par(mar = c(4, 4, 2, 2))  # Reduz as margens da figura

# Plotar a curva ROC
jpeg("Curva ROC.jpg", width = 8, height = 6, units = "in", res = 600)
plot(roc_curve, col = "blue", main = "Curva ROC")
dev.off()

# Imprimir o valor da AUC
cat("AUC:", auc(roc_curve), "\n")

###############################################
#PRECISION-RECALL CURVE

# Carregar o pacote PRROC
library(PRROC)

# Gerar a curva Precision-Recall com o argumento curve = TRUE
pr_curve <- pr.curve(scores.class0 = pred_probs, weights.class0 = test_data$Class, curve = TRUE)

# Plotar a curva Precision-Recall
jpeg("PR_CURVE.jpg", width = 8, height = 6, units = "in", res = 600)
plot(pr_curve)
dev.off()


#############################################################
# Criar um data frame com as probabilidades previstas e as classes reais
df_probs <- data.frame(Class = test_data$Class, Pred_Prob = pred_probs)

# Boxplot com melhorias
jpeg("Distribuição_Probabilidades_Classe.jpg", width = 8, height = 6, units = "in", res = 600)

ggplot(df_probs, aes(x = factor(Class), y = Pred_Prob, fill = factor(Class))) +
  geom_boxplot(notch = TRUE) +  # Adicionar notches
  stat_summary(fun = mean, geom = "point", shape = 20, size = 4, color = "black", 
               position = position_dodge(0.75)) +  # Adicionar a média
  labs(title = "Distribuição das Probabilidades por Classe", 
       x = "Classe Real", 
       y = "Probabilidade Prevista") +
  scale_fill_manual(values = c("lightgreen", "lightcoral")) +  # Cores distintas
  theme_minimal() +  # Tema minimalista
  theme(legend.position = "none")  # Remover legenda, se necessário

dev.off()

################################################################

# Contagem das classes
class_counts <- table(test_data$Class)

# Gráfico de barras com melhorias
jpeg("Fraudes_Transacoes.jpg", width = 8, height = 6, units = "in", res = 600)
ggplot(data.frame(Class = names(class_counts), Count = as.numeric(class_counts)), 
       aes(x = Class, y = Count, fill = Class)) +
  geom_bar(stat = "identity", color = "black") +
  geom_text(aes(label = Count), vjust = -0.5) +  # Rótulos com contagem
  labs(title = "Distribuição de Fraudes vs Transações Legítimas", 
       x = "Classe", 
       y = "Contagem") +
  scale_x_discrete(labels = c("0" = "Legítima", "1" = "Fraude")) +
  scale_fill_manual(values = c("lightgreen", "lightcoral")) +  # Cores distintas
  theme_minimal() +  # Tema minimalista
  theme(legend.position = "none")  # Remover legenda

dev.off()
#################################################################

# Carregar pacotes
library(pheatmap)

# Calcular a matriz de correlação
cor_matrix <- cor(data_normalized %>% select(-Class))

# Criar um heatmap
jpeg("heat_map.jpg", width = 8, height = 6, units = "in", res = 600)
pheatmap(cor_matrix,
         clustering_distance_rows = "euclidean",  # Distância para agrupamento das linhas
         clustering_distance_cols = "euclidean",  # Distância para agrupamento das colunas
         clustering_method = "complete",            # Método de agrupamento
         display_numbers = TRUE,                    # Exibir os números
         number_color = "black",                    # Cor dos números
         fontsize_number = 8,                      # Tamanho da fonte dos números
         number_format = "%.2f",
         color = colorRampPalette(c("blue", "white", "red"))(50),  # Paleta de cores
         main = "Heatmap de Correlação",           # Título do gráfico
         legend = TRUE,                             # Mostrar a legenda
         border_color = NA)                         # Remove bordas

dev.off()

# Calcular Precision, Recall e F1-Score
precision <- conf_matrix$byClass['Pos Pred Value']
recall <- conf_matrix$byClass['Sensitivity']
f1_score <- 2 * ((precision * recall) / (precision + recall))

# Criar um dataframe para plotar
f1_data <- data.frame(
  Métrica = c("Precisão", "Recall", "F1-Score"),
  Valor = c(precision, recall, f1_score)
)

# Plotando o gráfico de F1-Score
jpeg("PRF1.jpg", width = 8, height = 6, units = "in", res = 600)
ggplot(f1_data, aes(x = Métrica, y = Valor, fill = Métrica)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_text(aes(label = round(Valor, 4)), vjust = -0.5) +
  labs(title = "Precisão, Recall e F1-Score", y = "Valor", x = "Métrica") +
  theme_minimal() +
  scale_fill_manual(values = c("Precisão" = "blue", "Recall" = "orange", "F1-Score" = "green"))
dev.off()

# Usar o ggplot2 para plotar a distribuição de densidade das probabilidades
jpeg("PRF1.jpg", width = 8, height = 6, units = "in", res = 600)
ggplot(df_probs, aes(x = Pred_Prob, fill = factor(Class))) +
  geom_density(alpha = 0.5) +
  labs(title = "Distribuição de Densidade das Probabilidades Previstas", 
       x = "Probabilidade Prevista", 
       y = "Densidade", 
       fill = "Classe Real") +
  scale_fill_manual(values = c("lightgreen", "lightcoral")) +
  theme_minimal()
dev.off()

# Matriz de confusão normalizada
conf_matrix_normalized <- prop.table(conf_matrix$table, margin = 1)
conf_matrix_normalized

# Transformando a matriz de confusão normalizada em dataframe
conf_matrix_normalized_df <- as.data.frame(conf_matrix_normalized)
colnames(conf_matrix_normalized_df) <- c("Previsto", "Real", "Freq")

# Plotando a matriz de confusão normalizada
jpeg("Confusão_Normalizada.jpg", width = 8, height = 6, units = "in", res = 600)
ggplot(data = conf_matrix_normalized_df, aes(x = Previsto, y = Real, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(Freq, 2)), vjust = 1) +
  scale_fill_gradient(low = "lightblue", high = "blue") +
  labs(title = "Matriz de Confusão Normalizada", x = "Valores Preditos", y = "Valores Reais") +
  theme_minimal()
dev.off()

