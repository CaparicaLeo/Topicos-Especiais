import cv2

caminho_imagem = "dataset/25.png" 
img = cv2.imread(caminho_imagem)

if img is None:
    print("ERRO: A imagem não foi encontrada. Verifique o caminho!")
else:
    cv2.namedWindow("Selecione a Query", cv2.WINDOW_NORMAL)
    
    cv2.resizeWindow("Selecione a Query", 700, 900)

    roi = cv2.selectROI("Selecione a Query", img, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, w, h = roi
    x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)

    if w > 0 and h > 0:
        print(f"Sua Bounding Box para essa query é: ({x1}, {y1}, {x2}, {y2})")
    else:
        print("Nenhuma região foi selecionada.")