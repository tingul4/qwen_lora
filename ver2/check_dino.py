
from transformers import AutoProcessor
dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
import inspect
print(inspect.signature(dino_processor.post_process_grounded_object_detection))
