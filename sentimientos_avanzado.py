import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import string

# Descargar recursos necesarios (solo la primera vez)
nltk.download('vader_lexicon')

print("🧠 Analizador de Sentimientos IA\n")

texto = input("Escribí una frase o reseña: ")

# Limpiar puntuación
texto_limpio = texto.translate(str.maketrans('', '', string.punctuation))

# Inicializar analizador
sia = SentimentIntensityAnalyzer()

resultado = sia.polarity_scores(texto_limpio)

print("\n📊 Resultado del análisis:")
print(resultado)

# Interpretación
if resultado["compound"] >= 0.05:
    print("\nSentimiento: POSITIVO 😊")
elif resultado["compound"] <= -0.05:
    print("\nSentimiento: NEGATIVO 😡")
else:
    print("\nSentimiento: NEUTRO 😐")