import cv2
import torch
import os
import glob
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from scipy.spatial.distance import cosine

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
pasta_dataset = os.path.join(diretorio_atual, "dataset")

# Suas coordenadas
queries = [
    {"imagem": "1.png", "box": (272, 203, 1015, 404)},
    {"imagem": "10.png", "box": (1404, 884, 2254, 1070)},
    {"imagem": "11.png", "box": (1880, 62, 2216, 216)},
    {"imagem": "20.png", "box": (158, 201, 543, 564)},
    {"imagem": "25.png", "box": (903, 1728, 1289, 1872)},
]

print("Carregando ResNet50...")
pesos = ResNet50_Weights.DEFAULT
modelo = resnet50(weights=pesos)
modelo.eval()

preprocessamento = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def carregar_imagem_seguro(caminho):
    try:
        return cv2.imdecode(np.fromfile(caminho, dtype=np.uint8), cv2.IMREAD_COLOR)
    except: return None

def extrair_features(crop_imagem):
    tensor_entrada = preprocessamento(crop_imagem).unsqueeze(0)
    with torch.no_grad():
        return modelo(tensor_entrada).numpy().flatten()

def calcular_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    areaA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    areaB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return inter / float(areaA + areaB - inter)

def gerar_propostas_regiao(caminho_imagem, max_propostas=30):
    img = carregar_imagem_seguro(caminho_imagem)
    if img is None: return None, []
    
    escala = 0.3 
    img_pequena = cv2.resize(img, (int(img.shape[1]*escala), int(img.shape[0]*escala)))
    
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img_pequena)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    
    propostas = []
    for (x, y, w, h) in rects[:max_propostas]:
        propostas.append((int(x/escala), int(y/escala), int((x+w)/escala), int((y+h)/escala)))
    return img, propostas

todas_imagens = glob.glob(os.path.join(pasta_dataset, "*.*"))
arquivo_log = open(os.path.join(diretorio_atual, "resultados_finais.txt"), "w", encoding="utf-8")

def log_e_print(texto):
    print(texto)
    arquivo_log.write(texto + "\n")

log_e_print("--- INICIANDO RECUPERAÇÃO CBIR ---")

for idx, q in enumerate(queries):
    log_e_print(f"\nProcessando Query {idx+1}: {q['imagem']}")
    img_q = carregar_imagem_seguro(os.path.join(pasta_dataset, q['imagem']))
    if img_q is None: continue
    
    recorte_q = img_q[q['box'][1]:q['box'][3], q['box'][0]:q['box'][2]]
    feat_q = extrair_features(recorte_q)
    resultados = []

    for caminho_doc in todas_imagens:
        nome_doc = os.path.basename(caminho_doc)
        img_doc, propostas = gerar_propostas_regiao(caminho_doc)
        if img_doc is None: continue

        for box in propostas:
            recorte_p = img_doc[box[1]:box[3], box[0]:box[2]]
            if recorte_p.size == 0 or recorte_p.shape[0] < 20: continue
            
            sim_v = 1 - cosine(feat_q, extrair_features(recorte_p))
            iou = calcular_iou(q['box'], box)
            score = (0.7 * sim_v) + (0.3 * iou)
            resultados.append({'doc': nome_doc, 'box': box, 'score': score})

    resultados.sort(key=lambda x: x['score'], reverse=True)
    for i, res in enumerate(resultados[:3]):
        log_e_print(f"  {i+1}º -> {res['doc']} | Score: {res['score']:.4f} | Box: {res['box']}")

log_e_print("\n--- PROCESSO CONCLUÍDO ---")
arquivo_log.close()

input("\nOs resultados foram salvos em 'resultados_finais.txt'. Pressione ENTER para sair...")