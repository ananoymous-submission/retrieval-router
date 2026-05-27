import os
import cv2
import torch
import uuid
import numpy as np
from PIL import Image
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download

class LayoutAnalyzer:
    """
    Analyzes document layouts to detect visual elements like tables and figures.
    """
    
    def __init__(self, temp_dir="/tmp"):
        """
        Initialize the layout analyzer with a pre-trained model.
        
        Args:
            temp_dir (str): Directory for temporary files
        """
        # Select appropriate device for inference
        self.layout_device = ("cuda" if torch.cuda.is_available() else 
                             "mps" if torch.backends.mps.is_available() else 
                             "cpu")
        
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)

        # Download the pre-trained model
        self.layout_filepath = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="doclayout_yolo_docstructbench_imgsz1024.pt"
        )

        # Load the model
        self.layout_model = YOLOv10(self.layout_filepath).to(self.layout_device)

    def process_image(self, image, conf_threshold=0.5, image_size=1024):
        """
        Process an image to determine the portion of visual content.

        Args:
            image: Input image (file path, PIL Image, or numpy array)
            conf_threshold (float): Confidence threshold for detections
            image_size (int): Size to resize image for processing

        Returns:
            float: Portion of the detected content that is visual (tables/figures)
        """
        # Validate image input
        if image is None:
            raise ValueError("Image is None")

        # Prepare the image and save to temporary file with unique name
        img = self._prepare_image(image)
        unique_id = uuid.uuid4().hex
        temp_path = os.path.join(self.temp_dir, f"temp_{unique_id}.png")

        # Write image and verify it exists
        cv2.imwrite(temp_path, img)
        if not os.path.exists(temp_path):
            raise IOError(f"Failed to write temporary image file: {temp_path}")

        try:
            # Detect layout elements
            detections = self._detect_layout_elements(temp_path, conf_threshold, image_size)

            # Calculate visual portion
            visual_portion = self._calculate_visual_portion(image, detections)

        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return visual_portion
    
    def _detect_layout_elements(self, image_path, conf_threshold, image_size):
        """
        Detect layout elements in the document using the YOLO model.
        
        Args:
            image_path (str): Path to the image file
            conf_threshold (float): Confidence threshold for detections
            image_size (int): Size to resize image for processing
            
        Returns:
            dict: Dictionary of detected elements with their labels and coordinates
        """
        # Run inference with the layout model
        det_res = self.layout_model.predict(
            image_path,
            imgsz=image_size,
            conf=conf_threshold,
            device=self.layout_device
        )

        # Extract detection results
        detections = det_res[0].boxes.xyxy.cpu().numpy()
        class_names = det_res[0].names 
        labels = det_res[0].boxes.cls.cpu().numpy()  
        labels = [class_names[label] for label in labels]
    
        # Create a dictionary of detections
        return {i: (label, detections[i]) for i, label in enumerate(labels)}
    
    def _calculate_visual_portion(self, image, detections):
        """
        Calculate the portion of visual content in the detected elements.
        
        Args:
            image: Original image (for dimensions)
            detections: Dictionary of detected elements
            
        Returns:
            float: Portion of the detected content that is visual (tables/figures)
        """
        # Get image dimensions
        w, h = image.size
        
        # Create masks for tracking detected regions
        all_detections_mask = np.zeros((h, w), dtype=np.bool_)
        visual_mask = np.zeros((h, w), dtype=np.bool_)
        
        # Mark detected regions in the masks
        for (label, detection) in detections.values():
            x_min, y_min, x_max, y_max = detection
            
            # Convert to integers and ensure within image boundaries
            x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
            x_max, y_max = min(w, int(x_max)), min(h, int(y_max))
            
            # Mark this region in the all detections mask
            all_detections_mask[y_min:y_max, x_min:x_max] = True
            
            # Mark visual elements (tables and figures) in the visual mask
            if label in ("table", "figure"):
                visual_mask[y_min:y_max, x_min:x_max] = True
        
        # Calculate areas (avoiding double counting of overlapping regions)
        total_detected_pixels = np.sum(all_detections_mask)
        visual_pixels = np.sum(visual_mask)
        
        # Calculate the portion of visual content
        if total_detected_pixels > 0:
            return visual_pixels / total_detected_pixels
        else:
            return 0.0
    
    def _prepare_image(self, image):
        """
        Convert input to a numpy format for processing.
        
        Args:
            image: Input image (file path, PIL Image, or numpy array)
            
        Returns:
            numpy.ndarray: Image in BGR format for OpenCV processing
        """
        if isinstance(image, str):
            # Load from file path
            numpy_img = cv2.imread(image)
            if numpy_img is None:
                raise ValueError(f"Could not read image file: {image}")
            numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2RGB)

        elif isinstance(image, Image.Image):
            # Convert from PIL Image
            numpy_img = np.array(image)

        elif isinstance(image, np.ndarray):
            # Handle numpy array, checking if BGR conversion is needed
            numpy_img = image.copy()
            if numpy_img.shape[2] == 3:  
                if np.mean(numpy_img[:, :, 0]) > np.mean(numpy_img[:, :, 2]):
                    numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2RGB)
        else:
            raise TypeError("Input must be a file path, PIL Image, or numpy array")

        # Convert to BGR for OpenCV processing
        bgr_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)
        return bgr_img