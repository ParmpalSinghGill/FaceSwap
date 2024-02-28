import insightface,cv2,numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model('models/inswapper_128.onnx', download=True, download_zip=True)
img = cv2.imread('images/t1.jpg')
faces = app.get(img)

faces = sorted(faces, key=lambda x: x.bbox[0])
assert len(faces) == 6
source_face = faces[3]
res = img.copy()
for face in faces:
	res = swapper.get(res, face, source_face, paste_back=True)
cv2.imwrite("images/t1_swapped.jpg", res)
res = []
for face in faces:
	_img, _ = swapper.get(img, face, source_face, paste_back=False)
	res.append(_img)
res = np.concatenate(res, axis=1)
cv2.imwrite("images/t1_swapped2.jpg", res)
