# 1 Comprendre les mécanismes d'attention

import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding

# Vocabulaire
source_vocab = {"<pad»": 0, "J'": 1, "aime": 2, "l'IA": 3}
target_vocab = {"<pad»": 0, "I": 1, "love": 2, "AI": 3}
source_sentence = ["J'", "aime", "l'IA"]
target_sentence = ["I", "love", "AI"]

# Embedding

embedding_dim = 256

source_embedding_layer = Embedding(input_dim=len(source_vocab), output_dim=embedding_dim)
target_embedding_layer = Embedding(input_dim=len(target_vocab), output_dim=embedding_dim)
encoder_input = tf.constant([[source_vocab[w] for w in source_sentence]])
decoder_input = tf.constant([[target_vocab[w] for w in target_sentence]])
encoder_embedded = source_embedding_layer(encoder_input)
decoder_embedded = target_embedding_layer(decoder_input)
print(encoder_embedded)
# Cela permet de créer une couche d'embedding pour les mots de la source et de la cible.
# En clair cela va transformer les mots en vecteurs de dimension embedding_dim.
# On peut alors les utiliser pour les entrées du modèle.

# Transformation en clé, valeur et requête (Key, Value, Query)
Q = Dense(256, name="query")(decoder_embedded)
K = Dense(256, name="key")(encoder_embedded)
V = Dense(256, name="value")(encoder_embedded)
print("Dimensions de Q:", Q.shape)
print("\nDimensions de K:", K.shape)
print("\nDimensions de V:", V.shape)

## Calcul de l'Attention, cette formule ci dessous a été expliquée dans l'article sur les transformers sorti par google
QK = tf.matmul(Q, K, transpose_b=True)
QK_normalized = QK / math.sqrt (256)
softmax = tf.nn.softmax(QK_normalized)
attention_output = tf.matmul (softmax, V)
print("Dimension de l'attention:", attention_output.shape)
# Cela permet de donné une attention à chaque mot de la source en fonction de chaque mot de la cible.
# Cela permet de donner plus de poids aux mots qui sont importants pour la traduction.
