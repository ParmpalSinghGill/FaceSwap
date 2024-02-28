import datetime
import numpy as np
import os
import os.path as osp
import glob
import roop.globals
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

from roop.processors.Enhance_GFPGAN import Enhance_GFPGAN

assert insightface.__version__>='0.7'


def paste_upscale( fake_face, upsk_face, M, target_img, scale_factor, mask_offsets):
    M_scale = M * scale_factor
    IM = cv2.invertAffineTransform(M_scale)

    face_matte = np.full((target_img.shape[0], target_img.shape[1]), 255, dtype=np.uint8)
    ##Generate white square sized as a upsk_face
    img_matte = np.full((upsk_face.shape[0], upsk_face.shape[1]), 255, dtype=np.uint8)
    if mask_offsets[0] > 0:
        img_matte[:mask_offsets[0], :] = 0
    if mask_offsets[1] > 0:
        img_matte[-mask_offsets[1]:, :] = 0

    ##Transform white square back to target_img
    img_matte = cv2.warpAffine(img_matte, IM, (target_img.shape[1], target_img.shape[0]), flags=cv2.INTER_NEAREST,
                               borderValue=0.0)
    ##Blacken the edges of face_matte by 1 pixels (so the mask in not expanded on the image edges)
    img_matte[:1, :] = img_matte[-1:, :] = img_matte[:, :1] = img_matte[:, -1:] = 0

    # Detect the affine transformed white area
    mask_h_inds, mask_w_inds = np.where(img_matte == 255)
    # Calculate the size (and diagonal size) of transformed white area width and height boundaries
    mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
    mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
    mask_size = int(np.sqrt(mask_h * mask_w))
    # Calculate the kernel size for eroding img_matte by kernel (insightface empirical guess for best size was max(mask_size//10,10))
    # k = max(mask_size//12, 8)
    k = max(mask_size // 10, 10)
    kernel = np.ones((k, k), np.uint8)
    img_matte = cv2.erode(img_matte, kernel, iterations=1)
    # Calculate the kernel size for blurring img_matte by blur_size (insightface empirical guess for best size was max(mask_size//20, 5))
    # k = max(mask_size//24, 4)
    k = max(mask_size // 20, 5)
    kernel_size = (k, k)
    blur_size = tuple(2 * i + 1 for i in kernel_size)
    img_matte = cv2.GaussianBlur(img_matte, blur_size, 0)

    # Normalize images to float values and reshape
    img_matte = img_matte.astype(np.float32) / 255
    face_matte = face_matte.astype(np.float32) / 255
    img_matte = np.minimum(face_matte, img_matte)
    img_matte = np.reshape(img_matte, [img_matte.shape[0], img_matte.shape[1], 1])
    ##Transform upcaled face back to target_img
    paste_face = cv2.warpAffine(upsk_face, IM, (target_img.shape[1], target_img.shape[0]),
                                borderMode=cv2.BORDER_REPLICATE)
    if upsk_face is not fake_face:
        fake_face = cv2.warpAffine(fake_face, IM, (target_img.shape[1], target_img.shape[0]),
                                   borderMode=cv2.BORDER_REPLICATE)
        paste_face = cv2.addWeighted(paste_face, self.options.blend_ratio, fake_face, 1.0 - self.options.blend_ratio, 0)

    ##Re-assemble image
    paste_face = img_matte * paste_face
    paste_face = paste_face + (1 - img_matte) * target_img.astype(np.float32)
    del img_matte
    del face_matte
    del upsk_face
    del fake_face
    return paste_face.astype(np.uint8)


def workingCode():
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model('models/inswapper_128.onnx', download=True, download_zip=True)
    img = cv2.imread('images/t1.jpg')
    faces = app.get(img)
    faces = sorted(faces, key = lambda x : x.bbox[0])
    assert len(faces)==6
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


def Improvement1():
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model('models/inswapper_128.onnx', download=True, download_zip=True)
    roop.globals.execution_providers.append("CUDAExecutionProvider")
    enhancer=Enhance_GFPGAN()
    enhancer.Initialize("cuda")
    img = cv2.imread('images/t1.jpg')
    faces = app.get(img)
    faces = sorted(faces, key = lambda x : x.bbox[0])
    assert len(faces)==6
    source_face = faces[3]
    res = img.copy()
    for face in faces:
        res = swapper.get(res, face, source_face, paste_back=True)
    cv2.imwrite("images/t1_swapped.jpg", res)
    res,_=enhancer.Run(None,None,res)
    cv2.imwrite("images/t1_swapped_Enhanced.jpg", res)
    res = []
    for face in faces:
        _img, _ = swapper.get(img, face, source_face, paste_back=False)
        res.append(_img)
    res = np.concatenate(res, axis=1)
    cv2.imwrite("images/t1_swapped2.jpg", res)

def Improvement2():
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model('models/inswapper_128.onnx', download=True, download_zip=True)
    roop.globals.execution_providers.append("CUDAExecutionProvider")
    enhancer=Enhance_GFPGAN()
    enhancer.Initialize("cuda")
    img = cv2.imread('images/t1.jpg')
    faces = app.get(img)
    faces = sorted(faces, key = lambda x : x.bbox[0])
    source_face = faces
    res = img.copy()
    for i,face in enumerate(faces):
        res = swapper.get(res, face, source_face[(i+1)%len(source_face)], paste_back=True)
    cv2.imwrite("images/t1_swapped.jpg", res)
    res,_=enhancer.Run(None,None,res)
    cv2.imwrite("images/t1_swapped_Enhanced.jpg", res)
    res = []
    for face in faces:
        _img, _ = swapper.get(img, face, source_face, paste_back=False)
        res.append(_img)
    res = np.concatenate(res, axis=1)
    cv2.imwrite("images/t1_swapped2.jpg", res)


def Improvement3():
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model('models/inswapper_128.onnx', download=True, download_zip=True)
    roop.globals.execution_providers.append("CUDAExecutionProvider")
    enhancer=Enhance_GFPGAN()
    enhancer.Initialize("cuda")
    img = cv2.imread('images/t1.jpg')
    faces = app.get(img)
    faces = sorted(faces, key = lambda x : x.bbox[0])
    source_face = faces
    res = img.copy()
    for i,face in enumerate(faces):
        res = swapper.get(res, face, source_face[(i+1)%len(source_face)], paste_back=True)
    cv2.imwrite("images/t1_swapped.jpg", res)
    temp_frame = img.copy()
    source_face = faces[2]
    for face in faces:
        fake_frame, _ = swapper.get(img, face, source_face, paste_back=False)
        enhance_img, scale_factor =enhancer.Run(img, face, fake_frame)
        upscale = 512
        orig_width = fake_frame.shape[1]
        fake_frame = cv2.resize(fake_frame, (upscale, upscale), cv2.INTER_CUBIC)
        mask_offsets =face.mask_offsets

        if enhance_img is None:
            scale_factor = int(upscale / orig_width)
            img = paste_upscale(fake_frame, fake_frame, face.matrix, temp_frame, scale_factor, mask_offsets)
        else:
            img = paste_upscale(fake_frame, enhance_img, face.matrix, temp_frame, scale_factor,
                                        mask_offsets)

    cv2.imwrite("images/t1_swapped2_enhanced.jpg", img)

def check():
    import insightface, cv2, numpy as np
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    img = cv2.imread('images/t1.jpg')
    faces = app.get(img)
    for k,v in faces[0].items():
        print(k,v)
    print(faces[0].keys())


if __name__ == '__main__':
    # Improvement3()
    check()