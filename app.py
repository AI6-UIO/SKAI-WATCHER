'''
 # @ Author: wilfredo.martel - logica_razon@hotmail.com
 # @ Create Time: 2021-07-18 17:08:34
 # @ Modified by: wilfredo.martel
 # @ Modified time: 2021-07-18 17:08:36
 # @ Description: Servidor de aplicaciones para servir nuestro modelo
 '''
import io
import argparse
from PIL import Image
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


# model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=False, force_reload=True, classes=10).autoshape() 
pathLocla = 'C:/Users/Predator/Documents/Desarrollo_flask/vision_artificial/yolov5'
model = torch.hub.load(pathLocla, 'custom', path='firemodel.pt', source='local') 
model.eval()

@app.route('/alive', methods=['GET', 'POST'])
def index():
    return jsonify({'msg':'ok','data':'ok'}), 200

@app.route('/prediccion', methods=['POST'])
def prediccion():
    if 'file' not in request.files:
        return jsonify({'msg':'Es obligatorio adjuntar un archivo.','data':''})

    file = request.files["file"]
    if not file:
        return jsonify({'msg':'El archivo adjunto esta vacío.','data':''})

    img_bytes = request.files["file"].read()
    img = Image.open(io.BytesIO(img_bytes))

    # Pasamos la imagen del naevgador al modelo para su análisis
    results = model(img, size=640)
    # Devuelve el resultado de la inferencia en un json
    data = results.pandas().xyxy[0].to_json(orient="records")
    return jsonify({'msg':'Se ha inferido correctamente.','data':data})


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flask api para detección de incendios")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    app.run()