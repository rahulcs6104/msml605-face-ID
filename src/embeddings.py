import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

_device = None
_mtcnn = None
_model = None


def _get_device():
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def _get_mtcnn():
    global _mtcnn
    if _mtcnn is None:
        _mtcnn = MTCNN(image_size=160,margin=20,device=_get_device(),post_process=True,       
            select_largest=True,
        )
    return _mtcnn


def _get_model():
    global _model
    if _model is None:
        _model = InceptionResnetV1(pretrained="vggface2").eval().to(_get_device())
    return _model




def preprocess_face(image_path: str) -> torch.Tensor:
   
    img = Image.open(image_path).convert("RGB")
    mtcnn = _get_mtcnn()
    face = mtcnn(img)                       

    if face is None:
        
        img = img.resize((160, 160), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5             
        face = torch.from_numpy(arr).permute(2, 0, 1).float()

    return face


def extract_embedding(image_path: str) -> np.ndarray:

    face_tensor = preprocess_face(image_path)
    model = _get_model()
    with torch.no_grad():
        embedding = model(face_tensor.unsqueeze(0).to(_get_device()))
    return embedding.cpu().numpy().flatten()


def extract_embeddings_batch(image_paths: list) -> np.ndarray:

    embeddings = []
    for path in image_paths:
        embeddings.append(extract_embedding(path))
    return np.array(embeddings, dtype=np.float32)