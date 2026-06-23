# mini_buscador_jogos.py
# trabalho: mini buscador web com tema de plataforma de jogos (estilo Steam)
# mini-web com 10 paginas, 2 consultas, pagerank, hits, normalizacao e logica propria de melhoria

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# 1) PAGINAS DA MINI-WEB (tema: plataforma de jogos estilo Steam)
# =============================================================================

docs = [
    {
        "id": "A",
        "titulo": "GameStore - Portal Principal",
        "texto": "portal principal de jogos digitais loja online jogos para pc console indie aaa multiplayer",
        "links": ["B", "C", "D", "E", "F"]
    },
    {
        "id": "B",
        "titulo": "Jogos Multiplayer Online",
        "texto": "jogos multiplayer online para jogar com amigos competitivo cooperativo partidas ranked online",
        "links": ["C", "G", "H"]
    },
    {
        "id": "C",
        "titulo": "Counter-Strike 2 - Página do Jogo",
        "texto": "counter-strike 2 shooter multiplayer online competitivo ranked partidas online gratuito free to play",
        "links": ["B", "G"]
    },
    {
        "id": "D",
        "titulo": "Jogos de RPG",
        "texto": "jogos de rpg aventura historia personagens gratuito free to play rpg online mundo aberto",
        "links": ["E", "I"]
    },
    {
        "id": "E",
        "titulo": "Genshin Impact - Página do Jogo",
        "texto": "genshin impact rpg gratuito free to play aventura multiplayer online mundo aberto personagens",
        "links": ["D", "I"]
    },
    {
        "id": "F",
        "titulo": "Jogos Indie",
        "texto": "jogos indie criados por desenvolvedores independentes aventura puzzle plataforma historia unico",
        "links": ["A", "D"]
    },
    {
        "id": "G",
        "titulo": "Análises e Reviews de Jogos",
        "texto": "analises reviews jogos multiplayer online shooter rpg notas recomendacoes da comunidade",
        "links": ["B", "C", "D", "E"]
    },
    {
        "id": "H",
        "titulo": "Comunidade e Fórum de Jogadores",
        "texto": "forum comunidade jogadores dicas multiplayer online partidas grupos guilds eventos torneios",
        "links": ["B", "C", "G"]
    },
    {
        "id": "I",
        "titulo": "Jogos Gratuitos - Free to Play",
        "texto": "jogos gratuitos free to play rpg multiplayer online sem custo acessivel para todos os jogadores",
        "links": ["D", "E", "C"]
    },
    {
        "id": "J",
        "titulo": "Pagina Suspeita - Spam de Jogos",
        "texto": "melhor jogo melhor jogo gratis gratis multiplayer multiplayer clique agora promocao imperdivel jogo jogo jogo",
        "links": ["B", "C", "I"]
    }
]

df_paginas = pd.DataFrame(docs)

print("=" * 70)
print("MINI BUSCADOR WEB — TEMA: PLATAFORMA DE JOGOS (ESTILO STEAM)")
print("=" * 70)

print("\n1) PAGINAS DA MINI-WEB")
print(df_paginas[["id", "titulo", "texto", "links"]].to_string(index=False))


# =============================================================================
# CONSULTA 1: "jogos multiplayer online"
# =============================================================================

print("\n" + "=" * 70)
print("CONSULTA 1: jogos multiplayer online")
print("=" * 70)

consulta1 = "jogos multiplayer online"
relevantes1 = {"B", "C", "H"}  # paginas que realmente tratam de multiplayer online

print("\n2) CONSULTA USADA")
print(consulta1)

print("\n3) PAGINAS RELEVANTES PARA ESSA CONSULTA")
print(sorted(relevantes1))


# indice invertido

indice = {}
for doc in docs:
    palavras = doc["texto"].lower().split()
    for palavra in palavras:
        indice.setdefault(palavra, set()).add(doc["id"])

print("\n4) INDICE INVERTIDO — palavras da consulta")
for palavra in ["jogos", "multiplayer", "online"]:
    print(f"  '{palavra}': {sorted(indice.get(palavra, []))}")


# ranking textual tf-idf consulta 1

textos = [doc["texto"] for doc in docs]
vectorizer = TfidfVectorizer(lowercase=True)
X = vectorizer.fit_transform(textos)

q1 = vectorizer.transform([consulta1])
scores_texto1 = cosine_similarity(q1, X).flatten()

ranking_texto1 = pd.DataFrame({
    "id": [doc["id"] for doc in docs],
    "titulo": [doc["titulo"] for doc in docs],
    "score_texto": scores_texto1
})
ranking_texto1 = ranking_texto1.sort_values("score_texto", ascending=False).reset_index(drop=True)
ranking_texto1["posicao_texto"] = ranking_texto1.index + 1
ranking_texto1["top3_texto"] = ranking_texto1["posicao_texto"] <= 3
ranking_texto1["relevante"] = ranking_texto1["id"].isin(relevantes1)

print("\n5) RANKING TEXTUAL — Consulta 1")
print(ranking_texto1[["posicao_texto", "id", "titulo", "score_texto", "top3_texto", "relevante"]].round(6).to_string(index=False))


# grafo da mini-web

G = nx.DiGraph()
for doc in docs:
    G.add_node(doc["id"], titulo=doc["titulo"])
    for destino in doc["links"]:
        G.add_edge(doc["id"], destino)

df_links = pd.DataFrame(list(G.edges()), columns=["origem", "destino"])

print("\n6) LINKS DA MINI-WEB")
print(df_links.to_string(index=False))


# figura do grafo

print("\n   --> Gerando figura do grafo...")

pos = nx.spring_layout(G, seed=42, k=2.5)

node_colors = []
for node in G.nodes():
    if node == "J":
        node_colors.append("#e74c3c")   # vermelho = suspeito
    elif node in relevantes1:
        node_colors.append("#2ecc71")   # verde = relevante
    else:
        node_colors.append("#3498db")   # azul = normal

fig, ax = plt.subplots(figsize=(13, 9))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#7f8c8d", arrows=True,
                       arrowsize=20, width=1.5, connectionstyle="arc3,rad=0.1")
nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=1800, alpha=0.95)
nx.draw_networkx_labels(G, pos, ax=ax, font_color="white", font_size=11, font_weight="bold")

titulos_curtos = {doc["id"]: doc["titulo"].replace(" - ", "\n").replace("GameStore - ", "") for doc in docs}
label_offset = {node: (x, y - 0.18) for node, (x, y) in pos.items()}
nx.draw_networkx_labels(G, label_offset, labels=titulos_curtos, ax=ax,
                        font_color="#bdc3c7", font_size=6.5)

legenda = [
    mpatches.Patch(color="#2ecc71", label="Página Relevante (Consulta 1)"),
    mpatches.Patch(color="#3498db", label="Página Normal"),
    mpatches.Patch(color="#e74c3c", label="Página Suspeita (Spam)"),
]
ax.legend(handles=legenda, loc="upper left", facecolor="#2c2c54", labelcolor="white", fontsize=9)

plt.title("Mini-Web: Plataforma de Jogos (estilo Steam)\nGrafo de Links entre Páginas",
          color="white", fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("grafo_mini_web_jogos.png", dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
plt.close()
print("   --> Grafo salvo como 'grafo_mini_web_jogos.png'")


# links recebidos

df_links_recebidos = pd.DataFrame({
    "id": list(dict(G.in_degree()).keys()),
    "links_recebidos": list(dict(G.in_degree()).values())
})
df_links_recebidos = df_links_recebidos.merge(df_paginas[["id", "titulo"]], on="id")
df_links_recebidos = df_links_recebidos[["id", "titulo", "links_recebidos"]]
df_links_recebidos = df_links_recebidos.sort_values("links_recebidos", ascending=False).reset_index(drop=True)

print("\n7) LINKS RECEBIDOS POR PAGINA")
print(df_links_recebidos.to_string(index=False))


# pagerank

pagerank = nx.pagerank(G, alpha=0.85)

df_pr = pd.DataFrame({
    "id": list(pagerank.keys()),
    "pagerank": list(pagerank.values())
})
df_pr = df_pr.merge(df_paginas[["id", "titulo"]], on="id")
df_pr = df_pr[["id", "titulo", "pagerank"]]
df_pr = df_pr.sort_values("pagerank", ascending=False).reset_index(drop=True)
df_pr["posicao_pagerank"] = df_pr.index + 1

print("\n8) RANKING POR PAGERANK")
print(df_pr[["posicao_pagerank", "id", "titulo", "pagerank"]].round(6).to_string(index=False))


# hits

hubs, authorities = nx.hits(G, max_iter=1000, normalized=True)

df_hits = pd.DataFrame({
    "id": [doc["id"] for doc in docs],
    "titulo": [doc["titulo"] for doc in docs],
    "hub": [hubs[doc["id"]] for doc in docs],
    "authority": [authorities[doc["id"]] for doc in docs]
})

for col in ["hub", "authority"]:
    df_hits[col] = df_hits[col].apply(lambda x: 0 if abs(x) < 0.0000001 else x)

ranking_authorities = df_hits.sort_values("authority", ascending=False).reset_index(drop=True)
ranking_authorities["posicao_authority"] = ranking_authorities.index + 1

ranking_hubs = df_hits.sort_values("hub", ascending=False).reset_index(drop=True)
ranking_hubs["posicao_hub"] = ranking_hubs.index + 1

print("\n9) HITS — RANKING DE AUTHORITIES")
print(ranking_authorities[["posicao_authority", "id", "titulo", "authority"]].round(6).to_string(index=False))

print("\n10) HITS — RANKING DE HUBS")
print(ranking_hubs[["posicao_hub", "id", "titulo", "hub"]].round(6).to_string(index=False))


# ranking combinado — normalizado e media ponderada

def normalizar_coluna(df, coluna):
    maior_valor = df[coluna].max()
    if maior_valor == 0:
        return 0
    return df[coluna] / maior_valor


def montar_ranking_combinado(ranking_texto, df_pr, df_hits, relevantes):
    df_rank = ranking_texto[["id", "titulo", "score_texto"]].merge(df_pr[["id", "pagerank"]], on="id")
    df_rank = df_rank.merge(df_hits[["id", "hub", "authority"]], on="id")

    for col in ["score_texto", "pagerank", "authority"]:
        df_rank[col + "_norm"] = normalizar_coluna(df_rank, col)

    peso_texto = 0.60
    peso_pagerank = 0.25
    peso_authority = 0.15

    df_rank["score_final"] = (
        peso_texto * df_rank["score_texto_norm"] +
        peso_pagerank * df_rank["pagerank_norm"] +
        peso_authority * df_rank["authority_norm"]
    )

    ranking_antes = df_rank.sort_values("score_final", ascending=False).reset_index(drop=True)
    ranking_antes["posicao_antes"] = ranking_antes.index + 1
    ranking_antes["top3_antes"] = ranking_antes["posicao_antes"] <= 3
    ranking_antes["relevante"] = ranking_antes["id"].isin(relevantes)

    return df_rank, ranking_antes


df_rank1, ranking_antes1 = montar_ranking_combinado(ranking_texto1, df_pr, df_hits, relevantes1)

tabela_antes1 = ranking_antes1[[
    "posicao_antes", "id", "titulo",
    "score_texto_norm", "pagerank_norm", "authority_norm",
    "score_final", "top3_antes", "relevante"
]]

print("\n11) RANKING FINAL ANTES DA MELHORIA — Consulta 1")
print("\nobs: normalizado e combinado por media ponderada (texto 60%, pagerank 25%, authority 15%)")
print(tabela_antes1.round(6).to_string(index=False))


# =============================================================================
# LOGICA PROPRIA DE MELHORIA
#
# Problema percebido:
#   A pagina J (spam) tem texto cheio de palavras repetidas como "jogo jogo jogo"
#   e "multiplayer multiplayer", o que faz ela aparecer bem no ranking textual
#   mesmo sem ter conteudo util de verdade.
#
# Regra criada:
#   Detectar paginas com alta repeticao de palavras no proprio texto.
#   Se uma palavra aparece mais de 2 vezes no texto da pagina, ela e considerada
#   suspeita de spam por keyword stuffing.
#   Paginas suspeitas perdem 70% do score final.
#
# Isso e melhor do que marcar a pagina "J" na mao, porque a regra funciona
# automaticamente para qualquer pagina que abuse de repeticao de palavras.
# =============================================================================

print("\n" + "=" * 70)
print("LOGICA PROPRIA DE MELHORIA: DETECCAO DE KEYWORD STUFFING")
print("=" * 70)
print("""
Problema: paginas que repetem palavras-chave varias vezes no texto (keyword stuffing)
conseguem score textual alto sem ter conteudo util de verdade.

Regra: se qualquer palavra aparece mais de 2 vezes no texto de uma pagina,
ela e marcada como suspeita e perde 70% do score final.
""")

def detectar_keyword_stuffing(texto, limite=2):
    palavras = texto.lower().split()
    contagem = {}
    for p in palavras:
        contagem[p] = contagem.get(p, 0) + 1
    return int(any(v > limite for v in contagem.values()))


def aplicar_melhoria(df_rank, ranking_antes, relevantes, label_consulta):
    df_rank = df_rank.copy()

    df_rank["spam_suspeito"] = df_rank["id"].map(
        lambda id_: detectar_keyword_stuffing(
            next(doc["texto"] for doc in docs if doc["id"] == id_)
        )
    )

    df_rank["score_final_seguro"] = (
        df_rank["score_final"] * (1 - 0.70 * df_rank["spam_suspeito"])
    )

    ranking_depois = df_rank.sort_values("score_final_seguro", ascending=False).reset_index(drop=True)
    ranking_depois["posicao_depois"] = ranking_depois.index + 1
    ranking_depois["top3_depois"] = ranking_depois["posicao_depois"] <= 3
    ranking_depois["relevante"] = ranking_depois["id"].isin(relevantes)

    tabela_depois = ranking_depois[[
        "posicao_depois", "id", "titulo",
        "score_final", "spam_suspeito", "score_final_seguro",
        "top3_depois", "relevante"
    ]]

    print(f"\n12) RANKING FINAL DEPOIS DA MELHORIA — {label_consulta}")
    print(tabela_depois.round(6).to_string(index=False))

    return df_rank, ranking_depois


df_rank1, ranking_depois1 = aplicar_melhoria(df_rank1, ranking_antes1, relevantes1, "Consulta 1")


# precision@k

def precision_k(tabela, k, relevantes, coluna_id="id"):
    topk = tabela.head(k)[coluna_id].tolist()
    acertos = sum(p in relevantes for p in topk)
    return topk, acertos, acertos / k


def tabela_precision(ranking_texto, ranking_antes, ranking_depois, relevantes, label):
    linhas = []
    for nome, tabela in [
        ("somente texto", ranking_texto),
        ("texto + links", ranking_antes),
        ("texto + links + melhoria", ranking_depois)
    ]:
        top3, acertos3, p3 = precision_k(tabela, 3, relevantes)
        top2, acertos2, p2 = precision_k(tabela, 2, relevantes)
        linhas.append({
            "ranking": nome,
            "top3": ", ".join(top3),
            "relevantes_no_top3": acertos3,
            "precision@3": p3,
            "top2_extra": ", ".join(top2),
            "relevantes_no_top2": acertos2,
            "precision@2_extra": p2,
            "spam_no_top3": "J" in top3
        })
    df = pd.DataFrame(linhas)
    print(f"\n13) TABELA DE AVALIACAO — {label}")
    print(df.round(4).to_string(index=False))
    return df


tabela_precision(ranking_texto1, ranking_antes1, ranking_depois1, relevantes1, "Consulta 1")


# comparacao antes e depois

def comparacao_antes_depois(ranking_antes, ranking_depois, df_rank, label):
    comp = ranking_antes[["id", "titulo", "posicao_antes"]].merge(
        ranking_depois[["id", "posicao_depois"]], on="id"
    )
    comp["mudanca"] = comp["posicao_antes"] - comp["posicao_depois"]
    comp = comp.merge(df_rank[["id", "score_final", "spam_suspeito", "score_final_seguro"]], on="id")
    comp = comp.sort_values("posicao_depois").reset_index(drop=True)
    print(f"\n14) COMPARACAO ANTES E DEPOIS — {label}")
    print(comp.round(6).to_string(index=False))


comparacao_antes_depois(ranking_antes1, ranking_depois1, df_rank1, "Consulta 1")


# =============================================================================
# CONSULTA 2: "jogos de RPG gratuitos"
# =============================================================================

print("\n" + "=" * 70)
print("CONSULTA 2: jogos de RPG gratuitos")
print("=" * 70)

consulta2 = "jogos de RPG gratuitos"
relevantes2 = {"D", "E", "I"}  # paginas de rpg e free to play

print("\n2b) CONSULTA USADA")
print(consulta2)

print("\n3b) PAGINAS RELEVANTES PARA ESSA CONSULTA")
print(sorted(relevantes2))

q2 = vectorizer.transform([consulta2])
scores_texto2 = cosine_similarity(q2, X).flatten()

ranking_texto2 = pd.DataFrame({
    "id": [doc["id"] for doc in docs],
    "titulo": [doc["titulo"] for doc in docs],
    "score_texto": scores_texto2
})
ranking_texto2 = ranking_texto2.sort_values("score_texto", ascending=False).reset_index(drop=True)
ranking_texto2["posicao_texto"] = ranking_texto2.index + 1
ranking_texto2["top3_texto"] = ranking_texto2["posicao_texto"] <= 3
ranking_texto2["relevante"] = ranking_texto2["id"].isin(relevantes2)

print("\n5b) RANKING TEXTUAL — Consulta 2")
print(ranking_texto2[["posicao_texto", "id", "titulo", "score_texto", "top3_texto", "relevante"]].round(6).to_string(index=False))

df_rank2, ranking_antes2 = montar_ranking_combinado(ranking_texto2, df_pr, df_hits, relevantes2)

tabela_antes2 = ranking_antes2[[
    "posicao_antes", "id", "titulo",
    "score_texto_norm", "pagerank_norm", "authority_norm",
    "score_final", "top3_antes", "relevante"
]]

print("\n11b) RANKING FINAL ANTES DA MELHORIA — Consulta 2")
print(tabela_antes2.round(6).to_string(index=False))

df_rank2, ranking_depois2 = aplicar_melhoria(df_rank2, ranking_antes2, relevantes2, "Consulta 2")

tabela_precision(ranking_texto2, ranking_antes2, ranking_depois2, relevantes2, "Consulta 2")
comparacao_antes_depois(ranking_antes2, ranking_depois2, df_rank2, "Consulta 2")


# =============================================================================
# 15) RESUMO FINAL
# =============================================================================

print("\n" + "=" * 70)
print("15) RESUMO FINAL")
print("=" * 70)

for label, rt, ra, rd, rel in [
    ("Consulta 1 — jogos multiplayer online", ranking_texto1, ranking_antes1, ranking_depois1, relevantes1),
    ("Consulta 2 — jogos de RPG gratuitos",   ranking_texto2, ranking_antes2, ranking_depois2, relevantes2),
]:
    top3_txt, _, p3_txt = precision_k(rt, 3, rel)
    top3_ant, _, p3_ant = precision_k(ra, 3, rel)
    top3_dep, _, p3_dep = precision_k(rd, 3, rel)
    top2_ant, _, p2_ant = precision_k(ra, 2, rel)
    top2_dep, _, p2_dep = precision_k(rd, 2, rel)

    print(f"\n{label}")
    print(f"  Top 3 somente texto:             {top3_txt} | Precision@3 = {p3_txt:.4f}")
    print(f"  Top 3 texto + links:             {top3_ant} | Precision@3 = {p3_ant:.4f}")
    print(f"  Top 3 texto + links + melhoria:  {top3_dep} | Precision@3 = {p3_dep:.4f}")
    print(f"  (extra) Top 2 antes da melhoria: {top2_ant} | Precision@2 = {p2_ant:.4f}")
    print(f"  (extra) Top 2 apos a melhoria:   {top2_dep} | Precision@2 = {p2_dep:.4f}")
