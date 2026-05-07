import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "A inteligência artificial generativa está revolucionando a criação de conteúdo digital e artes visuais.",
    "A SpaceX lançou mais uma leva de satélites Starlink para expandir a cobertura de internet global.",
    "Novos estudos indicam que a inteligência artificial pode auxiliar médicos no diagnóstico precoce de câncer.",
    "O telescópio James Webb capturou imagens inéditas de galáxias formadas logo após o Big Bang.",
    "O governo anunciou novas medidas econômicas para conter a inflação e estimular o consumo interno."
]

query = "inteligência artificial na medicina"

stop_words_pt = ['a', 'o', 'as', 'os', 'de', 'do', 'da', 'em', 'um', 'uma', 'e', 'para', 'no', 'na', 'com']

vectorizer = TfidfVectorizer(stop_words=stop_words_pt)

tfidf_matrix = vectorizer.fit_transform(documents)

query_vector = vectorizer.transform([query])

scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

df_resultados = pd.DataFrame({
    'Documento': documents,
    'Score': scores
}).sort_values(by='Score', ascending=False)

print(f"Consulta do Usuário: '{query}'\n")
print(df_resultados.to_string(index=False))