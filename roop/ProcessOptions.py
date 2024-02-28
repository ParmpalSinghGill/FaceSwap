import json


class ProcessOptions:

    def __init__(self,processors, face_distance,  blend_ratio, swap_mode, selected_index, masking_text):
        self.processors = processors
        self.face_distance_threshold = face_distance
        self.blend_ratio = blend_ratio
        self.swap_mode = swap_mode
        self.selected_index = selected_index
        self.masking_text = masking_text

    def __repr__(self):
        return json.dumps({"processors":self.processors,"face_distance_threshold":self.face_distance_threshold,"blend_ratio":self.blend_ratio,
                "swap_mode":self.swap_mode,"selected_index":self.selected_index,"masking_text":self.masking_text})