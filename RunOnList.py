import cv2,glob,os,pathlib
import roop.globals
from settings import Settings
from roop.ProcessEntry import ProcessEntry
from roop.face_util import extract_face_images

def setGlobals():
	roop.globals.CFG = Settings('config55.yaml')
	roop.globals.execution_threads = roop.globals.CFG.max_threads
	roop.globals.video_encoder = roop.globals.CFG.output_video_codec
	roop.globals.video_quality = roop.globals.CFG.video_quality
	roop.globals.execution_providers.append("CUDAExecutionProvider")
	roop.globals.output_path = os.path.abspath(os.path.join(os.getcwd(), "output"))
	os.makedirs(roop.globals.output_path, exist_ok=True)
	os.environ["TEMP"] = os.environ["TMP"] = os.path.abspath(os.path.join(os.getcwd(), "temp"))
	os.makedirs(os.environ["TEMP"], exist_ok=True)
	os.environ["GRADIO_TEMP_DIR"] = os.environ["TEMP"]



setGlobals()

def convert_to_gradio(image):
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def processSource(sourcepath):
	SELECTION_FACES_DATA=extract_face_images(sourcepath,  (False, 0))
	thumbs=[]
	for f in SELECTION_FACES_DATA:
		image = convert_to_gradio(f[1])
		thumbs.append(image)

	if len(thumbs) == 1:
		face = SELECTION_FACES_DATA[0][0]
		face.mask_offsets = (0, 0)
		# roop.globals.INPUT_FACES.append(face)
		roop.globals.INPUT_FACES=[face]

list_files_process=[]




def start_swap(enhancer, detection, keep_frames, skip_audio, face_distance, blend_ratio,
			   use_clip, clip_text, processing_method):

	from roop.core import batch_process
	global is_processing, list_files_process

	roop.globals.selected_enhancer = enhancer
	roop.globals.target_path = None
	roop.globals.distance_threshold = face_distance
	roop.globals.blend_ratio = blend_ratio
	roop.globals.keep_frames = keep_frames
	roop.globals.skip_audio = skip_audio
	roop.globals.face_swap_mode = "first"
	if use_clip and clip_text is None or len(clip_text) < 1:
		use_clip = False


	is_processing = True
	roop.globals.execution_threads = roop.globals.CFG.max_threads
	roop.globals.video_encoder = roop.globals.CFG.output_video_codec
	roop.globals.video_quality = roop.globals.CFG.video_quality
	roop.globals.max_memory = roop.globals.CFG.memory_limit if roop.globals.CFG.memory_limit > 0 else None
	print("list_files_process",len(list_files_process))
	batch_process(list_files_process, use_clip, clip_text, processing_method == "In-Memory processing", None)
	is_processing = False
	outdir = pathlib.Path(roop.globals.output_path)
	# outfiles = [item for item in outdir.rglob("*") if item.is_file()]
	print("Results are in ",roop.globals.output_path)

def ReplaceAllWith(sourceImage,targetFolder):
	processSource(sourceImage)
	for f in glob.glob(targetFolder + "*"):
		list_files_process.append(ProcessEntry(f, 0, 0, 0))
	# start_swap("GFPGAN","First found",False,False,0.65,0.65,use_clip=True,clip_text="cup",processing_method="In-Memory processing")
	start_swap("GFPGAN", "First found", False, False, 0.65, 0.65, use_clip=False, clip_text="",
			   processing_method="In-Memory processing")


def MANYTOMANNY(source,target):
	for f in os.listdir(source):
		sourcepath=source+f
		ReplaceAllWith(sourcepath, target)

def Experiments():
	sourcepath="/media/parmpal/Data/Media/Pictures/04beautiful-eyes1.jpg"
	# sourcepath="/media/parmpal/Data/Codes/Python/upwork/ADiscussions/justin/swap/swap/source/square5.jpg"
	tagetfolder="/media/parmpal/Data/Codes/Python/upwork/ADiscussions/justin/swap/swap/targets/"
	processSource(sourcepath)
	for f in glob.glob(tagetfolder+"*")[:2]:
		list_files_process.append(ProcessEntry(f, 0, 0, 0))
	# start_swap("GFPGAN","First found",False,False,0.65,0.65,use_clip=True,clip_text="cup",processing_method="In-Memory processing")
	start_swap("GFPGAN","First found",False,False,0.65,0.65,use_clip=False,clip_text="",processing_method="In-Memory processing")
	# start_swap(None,"First found",False,False,0.65,0.65,use_clip=False,clip_text="",processing_method="In-Memory processing")

	# frame = cv2.imread(sourcepath)
	# res=get_face_analyser().get(frame)
	# print(res)

# MANYTOMANNY(source="",target="")

# ReplaceAllWith("", targetFolder="")