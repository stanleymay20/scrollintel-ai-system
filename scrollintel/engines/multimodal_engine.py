"""
ScrollIntel MultimodalEngine for Cross-Modal Intelligence
Implements audio, image, and text processing with cross-modal fusion capabilities.

Requirements: 12.1, 12.2, 12.3, 12.4
"""

import asyncio
import logging
import json
import io
import base64
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from datetime import datetime
import tempfile
import os
from pathlib import Path

# Core ML frameworks
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torchvision.models import resnet50
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    transforms = None
    resnet50 = None

try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    tf = None

# Computer Vision
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None

# Machine Learning
try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    DBSCAN = None

try:
    from PIL import Image, ImageEnhance
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None
    ImageEnhance = None

# Audio processing
try:
    import librosa
    import soundfile as sf
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    librosa = None
    sf = None

logger = logging.getLogger(__name__)

class ModalityType(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"

class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class MultimodalInput:
    modality: ModalityType
    data: Union[str, bytes, np.ndarray]
    metadata: Dict[str, Any]
    timestamp: datetime
    source_id: Optional[str] = None

@dataclass
class ModalityEmbedding:
    modality: ModalityType
    embedding: np.ndarray
    features: Dict[str, Any]
    confidence: float
    processing_time: float

@dataclass
class CrossModalResult:
    input_modalities: List[ModalityType]
    fused_embedding: np.ndarray
    individual_embeddings: List[ModalityEmbedding]
    fusion_confidence: float
    insights: Dict[str, Any]
    processing_metadata: Dict[str, Any]

class TextProcessor:
    """Handles text processing and embedding generation"""
    
    def __init__(self):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize text embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info("Text embedding model initialized successfully")
        except ImportError:
            logger.warning("sentence-transformers not available, using fallback")
            self.model = None
    
    def process_text(self, text: str, metadata: Dict[str, Any] = None) -> ModalityEmbedding:
        """Process text and generate embeddings"""
        start_time = datetime.now()
        
        if self.model:
            # Use sentence-transformers for high-quality embeddings
            embedding = self.model.encode([text])[0]
            confidence = 0.9
        else:
            # Fallback: simple TF-IDF-like embedding
            embedding = self._simple_text_embedding(text)
            confidence = 0.6
        
        # Extract text features
        features = self._extract_text_features(text)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ModalityEmbedding(
            modality=ModalityType.TEXT,
            embedding=embedding,
            features=features,
            confidence=confidence,
            processing_time=processing_time
        )
    
    def _simple_text_embedding(self, text: str) -> np.ndarray:
        """Simple fallback text embedding"""
        # Basic word frequency embedding
        words = text.lower().split()
        vocab_size = 1000
        embedding = np.zeros(384)  # Match sentence-transformers dimension
        
        for i, word in enumerate(words[:50]):  # Limit to first 50 words
            word_hash = hash(word) % vocab_size
            embedding[word_hash % 384] += 1.0 / (i + 1)  # Position weighting
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _extract_text_features(self, text: str) -> Dict[str, Any]:
        """Extract text-specific features"""
        words = text.split()
        sentences = text.split('.')
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'character_count': len(text),
            'language': 'en',  # Simplified - could use language detection
            'sentiment_score': self._simple_sentiment(text),
            'readability_score': self._simple_readability(text)
        }
    
    def _simple_sentiment(self, text: str) -> float:
        """Simple sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'poor']
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def _simple_readability(self, text: str) -> float:
        """Simple readability score (Flesch-like)"""
        words = text.split()
        sentences = text.split('.')
        
        if not words or not sentences:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables = np.mean([self._count_syllables(word) for word in words])
        
        # Simplified Flesch formula
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        return max(0, min(100, score)) / 100.0  # Normalize to 0-1
    
    def _count_syllables(self, word: str) -> int:
        """Simple syllable counting"""
        vowels = 'aeiouy'
        word = word.lower()
        count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent e
        if word.endswith('e') and count > 1:
            count -= 1
        
        return max(1, count)

class ImageProcessor:
    """Handles image processing and computer vision"""
    
    def __init__(self):
        self.model = None
        self.transform = None
        self.ocr_engine = None
        self.object_detector = None
        self._initialize_model()
        self._initialize_computer_vision()
    
    def _initialize_computer_vision(self):
        """Initialize computer vision capabilities including OCR"""
        # Initialize OCR
        try:
            import pytesseract
            self.ocr_engine = pytesseract
            logger.info("OCR engine initialized successfully")
        except ImportError:
            logger.warning("pytesseract not available, OCR disabled")
            self.ocr_engine = None
        
        # Initialize object detection (using OpenCV's pre-trained models)
        if HAS_OPENCV:
            try:
                # Load YOLO or other object detection model if available
                # For now, we'll use basic feature detection
                self.object_detector = cv2
                logger.info("Object detection capabilities initialized")
            except Exception as e:
                logger.warning(f"Object detection initialization failed: {e}")
                self.object_detector = None
    
    def _initialize_model(self):
        """Initialize image processing model"""
        if HAS_TORCH:
            try:
                self.model = resnet50(pretrained=True)
                self.model.eval()
                
                # Remove final classification layer to get features
                self.model = nn.Sequential(*list(self.model.children())[:-1])
                
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
                logger.info("Image processing model initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize image model: {e}")
                self.model = None
    
    def process_image(self, image_data: Union[bytes, np.ndarray, str], 
                     metadata: Dict[str, Any] = None) -> ModalityEmbedding:
        """Process image and generate embeddings"""
        start_time = datetime.now()
        
        # Load image
        image = self._load_image(image_data)
        if image is None:
            raise ValueError("Failed to load image")
        
        # Generate embedding
        if self.model and HAS_TORCH:
            embedding = self._extract_deep_features(image)
            confidence = 0.9
        else:
            embedding = self._extract_basic_features(image)
            confidence = 0.6
        
        # Extract image features
        features = self._extract_image_features(image)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ModalityEmbedding(
            modality=ModalityType.IMAGE,
            embedding=embedding,
            features=features,
            confidence=confidence,
            processing_time=processing_time
        )
    
    def _load_image(self, image_data: Union[bytes, np.ndarray, str]) -> Optional[np.ndarray]:
        """Load image from various formats"""
        try:
            if isinstance(image_data, str):
                # Base64 encoded image
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image_data = image_bytes
            
            if isinstance(image_data, bytes):
                # Convert bytes to PIL Image then to numpy
                if HAS_PIL:
                    image = Image.open(io.BytesIO(image_data))
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    return np.array(image)
                else:
                    # Fallback without PIL
                    return None
            
            elif isinstance(image_data, np.ndarray):
                return image_data
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return None
    
    def _extract_deep_features(self, image: np.ndarray) -> np.ndarray:
        """Extract deep features using ResNet"""
        try:
            # Convert numpy to PIL to tensor
            pil_image = Image.fromarray(image.astype('uint8'))
            tensor = self.transform(pil_image).unsqueeze(0)
            
            with torch.no_grad():
                features = self.model(tensor)
                features = features.squeeze().numpy()
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract deep features: {e}")
            return self._extract_basic_features(image)
    
    def _extract_basic_features(self, image: np.ndarray) -> np.ndarray:
        """Extract basic image features without deep learning"""
        # Color histogram features
        if len(image.shape) == 3:
            # RGB image
            hist_r = np.histogram(image[:,:,0], bins=32, range=(0, 256))[0]
            hist_g = np.histogram(image[:,:,1], bins=32, range=(0, 256))[0]
            hist_b = np.histogram(image[:,:,2], bins=32, range=(0, 256))[0]
            color_features = np.concatenate([hist_r, hist_g, hist_b])
        else:
            # Grayscale
            hist = np.histogram(image, bins=96, range=(0, 256))[0]
            color_features = hist
        
        # Texture features (simplified)
        if HAS_OPENCV and cv2 is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            # Sobel edge detection
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            texture_features = np.histogram(edge_magnitude, bins=32)[0]
        else:
            # Simple gradient without OpenCV
            gray = np.mean(image, axis=2) if len(image.shape) == 3 else image
            grad_x = np.diff(gray, axis=1)
            grad_y = np.diff(gray, axis=0)
            texture_features = np.histogram(np.abs(grad_x), bins=32)[0]
        
        # Combine features
        features = np.concatenate([color_features, texture_features])
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        # Pad or truncate to standard size
        target_size = 2048
        if len(features) < target_size:
            features = np.pad(features, (0, target_size - len(features)))
        else:
            features = features[:target_size]
        
        return features
    
    def _extract_image_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract image metadata and features"""
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        
        # Basic statistics
        mean_brightness = np.mean(image)
        std_brightness = np.std(image)
        
        # Color analysis
        if channels == 3:
            dominant_color = [int(np.mean(image[:,:,i])) for i in range(3)]
        else:
            dominant_color = [int(mean_brightness)]
        
        # Perform OCR to extract text
        extracted_text = self._extract_text_from_image(image)
        
        # Perform object detection
        detected_objects = self._detect_objects(image)
        
        return {
            'width': width,
            'height': height,
            'channels': channels,
            'aspect_ratio': width / height,
            'total_pixels': width * height,
            'mean_brightness': float(mean_brightness),
            'std_brightness': float(std_brightness),
            'dominant_color': dominant_color,
            'is_grayscale': channels == 1,
            'estimated_file_size': width * height * channels,
            'extracted_text': extracted_text,
            'has_text_content': extracted_text is not None and len(extracted_text.strip()) > 0,
            'detected_objects': detected_objects,
            'num_objects_detected': len(detected_objects) if detected_objects else 0
        }
    
    def _extract_text_from_image(self, image: np.ndarray) -> Optional[str]:
        """Extract text from image using OCR"""
        if not self.ocr_engine:
            return None
        
        try:
            # Convert numpy array to PIL Image for OCR
            if HAS_PIL:
                pil_image = Image.fromarray(image.astype('uint8'))
                
                # Perform OCR
                text = self.ocr_engine.image_to_string(pil_image, config='--psm 6')
                
                # Clean up the text
                text = text.strip()
                if len(text) > 0:
                    logger.debug(f"OCR extracted text: {text[:100]}...")
                    return text
                else:
                    return None
            else:
                logger.warning("PIL not available for OCR")
                return None
                
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return None
    
    def _detect_objects(self, image: np.ndarray) -> Optional[List[Dict[str, Any]]]:
        """Detect objects in image using computer vision"""
        if not self.object_detector or not HAS_OPENCV:
            return None
        
        try:
            # Convert to grayscale for feature detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            
            # Use ORB feature detector as a simple object detection method
            orb = cv2.ORB_create()
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            
            # Group keypoints into potential objects
            objects = []
            if keypoints:
                # Simple clustering of keypoints to identify object regions
                keypoint_coords = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
                
                # Basic clustering - group nearby keypoints
                if len(keypoint_coords) > 5 and HAS_SKLEARN:
                    clustering = DBSCAN(eps=50, min_samples=3).fit(keypoint_coords)
                    
                    unique_labels = set(clustering.labels_)
                    for label in unique_labels:
                        if label != -1:  # Ignore noise points
                            cluster_points = keypoint_coords[clustering.labels_ == label]
                            
                            # Calculate bounding box
                            x_min, y_min = np.min(cluster_points, axis=0)
                            x_max, y_max = np.max(cluster_points, axis=0)
                            
                            objects.append({
                                'type': 'feature_cluster',
                                'confidence': 0.7,  # Basic confidence
                                'bbox': [int(x_min), int(y_min), int(x_max), int(y_max)],
                                'num_features': len(cluster_points)
                            })
                
            return objects if objects else []
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return None

class VideoProcessor:
    """Handles video processing and analysis"""
    
    def __init__(self):
        self.frame_sample_rate = 1.0  # Extract 1 frame per second
        self.image_processor = None
        self.audio_processor = None
    
    def set_processors(self, image_processor, audio_processor):
        """Set reference to image and audio processors"""
        self.image_processor = image_processor
        self.audio_processor = audio_processor
    
    def process_video(self, video_data: Union[bytes, str], metadata: Dict[str, Any] = None) -> ModalityEmbedding:
        """Process video and generate embeddings"""
        start_time = datetime.now()
        
        # For now, return a basic video embedding
        # In production, this would extract frames and audio
        features = self._extract_video_metadata(video_data)
        
        # Create a basic embedding (in production, would combine frame and audio embeddings)
        embedding = self._create_basic_video_embedding(video_data)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ModalityEmbedding(
            modality=ModalityType.VIDEO,
            embedding=embedding,
            features=features,
            confidence=0.7,  # Lower confidence for basic implementation
            processing_time=processing_time
        )
    
    def _extract_video_metadata(self, video_data: Union[bytes, str]) -> Dict[str, Any]:
        """Extract basic video metadata"""
        if isinstance(video_data, str):
            # Assume it's a base64 encoded video
            try:
                video_bytes = base64.b64decode(video_data.split(',')[1] if ',' in video_data else video_data)
                file_size = len(video_bytes)
            except:
                file_size = len(video_data.encode())
        else:
            file_size = len(video_data)
        
        return {
            'file_size': file_size,
            'estimated_duration': file_size / (1024 * 1024),  # Rough estimate
            'has_audio_track': True,  # Assume yes for now
            'has_video_track': True,
            'format': 'unknown',
            'processing_method': 'basic'
        }
    
    def _create_basic_video_embedding(self, video_data: Union[bytes, str]) -> np.ndarray:
        """Create basic video embedding"""
        # Create a hash-based embedding for now
        import hashlib
        
        if isinstance(video_data, str):
            data_hash = hashlib.md5(video_data.encode()).hexdigest()
        else:
            data_hash = hashlib.md5(video_data).hexdigest()
        
        # Convert hash to numeric embedding
        embedding = np.array([int(data_hash[i:i+2], 16) for i in range(0, min(32, len(data_hash)), 2)])
        
        # Pad to standard size
        target_size = 512
        if len(embedding) < target_size:
            embedding = np.pad(embedding, (0, target_size - len(embedding)))
        else:
            embedding = embedding[:target_size]
        
        # Normalize
        embedding = embedding.astype(np.float32) / 255.0
        
        return embedding

class AudioProcessor:
    """Handles audio processing and speech recognition"""
    
    def __init__(self):
        self.sample_rate = 16000
        self.n_mfcc = 13
        self.speech_recognizer = None
        self._initialize_speech_recognition()
    
    def _initialize_speech_recognition(self):
        """Initialize speech recognition capabilities"""
        try:
            import speech_recognition as sr
            self.speech_recognizer = sr.Recognizer()
            logger.info("Speech recognition initialized successfully")
        except ImportError:
            logger.warning("speech_recognition not available, using fallback")
            self.speech_recognizer = None
    
    def process_audio(self, audio_data: Union[bytes, np.ndarray, str], 
                     metadata: Dict[str, Any] = None) -> ModalityEmbedding:
        """Process audio and generate embeddings"""
        start_time = datetime.now()
        
        # Load audio
        audio_array, sr = self._load_audio(audio_data)
        if audio_array is None:
            raise ValueError("Failed to load audio")
        
        # Generate embedding
        embedding = self._extract_audio_features(audio_array, sr)
        
        # Extract audio features
        features = self._extract_audio_metadata(audio_array, sr)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ModalityEmbedding(
            modality=ModalityType.AUDIO,
            embedding=embedding,
            features=features,
            confidence=0.8,
            processing_time=processing_time
        )
    
    def _load_audio(self, audio_data: Union[bytes, np.ndarray, str]) -> Tuple[Optional[np.ndarray], int]:
        """Load audio from various formats"""
        try:
            if isinstance(audio_data, str):
                # Base64 encoded audio
                if audio_data.startswith('data:audio'):
                    audio_data = audio_data.split(',')[1]
                audio_bytes = base64.b64decode(audio_data)
                audio_data = audio_bytes
            
            if isinstance(audio_data, bytes):
                # Save to temporary file and load with librosa
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    tmp_file.write(audio_data)
                    tmp_path = tmp_file.name
                
                try:
                    if HAS_LIBROSA:
                        audio_array, sr = librosa.load(tmp_path, sr=self.sample_rate)
                    else:
                        # Fallback: assume it's raw PCM data
                        audio_array = np.frombuffer(audio_data, dtype=np.float32)
                        sr = self.sample_rate
                    
                    os.unlink(tmp_path)
                    return audio_array, sr
                    
                except Exception as e:
                    os.unlink(tmp_path)
                    logger.error(f"Failed to load audio file: {e}")
                    return None, 0
            
            elif isinstance(audio_data, np.ndarray):
                return audio_data, self.sample_rate
            
            return None, 0
            
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            return None, 0
    
    def _extract_audio_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract audio features for embedding"""
        if HAS_LIBROSA:
            return self._extract_librosa_features(audio, sr)
        else:
            return self._extract_basic_audio_features(audio, sr)
    
    def _extract_librosa_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract features using librosa"""
        try:
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
            
            # Combine features
            features = np.concatenate([
                mfcc_mean,
                mfcc_std,
                [np.mean(spectral_centroids)],
                [np.std(spectral_centroids)],
                [np.mean(spectral_rolloff)],
                [np.std(spectral_rolloff)],
                [np.mean(zero_crossing_rate)],
                [np.std(zero_crossing_rate)]
            ])
            
            # Pad to standard size
            target_size = 128
            if len(features) < target_size:
                features = np.pad(features, (0, target_size - len(features)))
            else:
                features = features[:target_size]
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract librosa features: {e}")
            return self._extract_basic_audio_features(audio, sr)
    
    def _extract_basic_audio_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract basic audio features without librosa"""
        # Time domain features
        rms_energy = np.sqrt(np.mean(audio**2))
        zero_crossings = np.sum(np.diff(np.sign(audio)) != 0)
        
        # Frequency domain features (simple FFT)
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft[:len(fft)//2])
        
        # Spectral centroid (simplified)
        freqs = np.fft.fftfreq(len(audio), 1/sr)[:len(magnitude)]
        spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0
        
        # Basic spectral features
        spectral_mean = np.mean(magnitude)
        spectral_std = np.std(magnitude)
        spectral_max = np.max(magnitude)
        
        # Combine features
        features = np.array([
            rms_energy,
            zero_crossings / len(audio),
            spectral_centroid,
            spectral_mean,
            spectral_std,
            spectral_max
        ])
        
        # Pad to standard size
        target_size = 128
        features = np.pad(features, (0, target_size - len(features)))
        
        return features
    
    def _extract_audio_metadata(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract audio metadata"""
        duration = len(audio) / sr
        rms_energy = np.sqrt(np.mean(audio**2))
        
        # Perform speech-to-text if available
        transcription = self._speech_to_text(audio, sr)
        
        return {
            'duration_seconds': float(duration),
            'sample_rate': sr,
            'num_samples': len(audio),
            'rms_energy': float(rms_energy),
            'max_amplitude': float(np.max(np.abs(audio))),
            'dynamic_range': float(np.max(audio) - np.min(audio)),
            'estimated_speech': self._detect_speech_activity(audio),
            'silence_ratio': self._calculate_silence_ratio(audio),
            'transcription': transcription,
            'has_speech_content': transcription is not None and len(transcription.strip()) > 0
        }
    
    def _speech_to_text(self, audio: np.ndarray, sr: int) -> Optional[str]:
        """Convert speech to text using speech recognition"""
        if not self.speech_recognizer:
            return None
        
        try:
            import speech_recognition as sr_lib
            import tempfile
            import soundfile as sf
            
            # Save audio to temporary file for speech recognition
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, audio, sr)
                tmp_path = tmp_file.name
            
            try:
                # Load audio file for speech recognition
                with sr_lib.AudioFile(tmp_path) as source:
                    audio_data = self.speech_recognizer.record(source)
                
                # Perform speech recognition
                try:
                    # Try Google Speech Recognition (free tier)
                    text = self.speech_recognizer.recognize_google(audio_data)
                    logger.info(f"Speech recognition successful: {text[:50]}...")
                    return text
                except sr_lib.UnknownValueError:
                    logger.debug("Speech recognition could not understand audio")
                    return None
                except sr_lib.RequestError as e:
                    logger.warning(f"Speech recognition service error: {e}")
                    return None
                    
            finally:
                os.unlink(tmp_path)
                
        except Exception as e:
            logger.error(f"Speech-to-text processing failed: {e}")
            return None
    
    def _detect_speech_activity(self, audio: np.ndarray) -> bool:
        """Simple speech activity detection"""
        # Basic energy-based detection
        energy = np.sqrt(np.mean(audio**2))
        return energy > 0.01  # Threshold for speech activity
    
    def _calculate_silence_ratio(self, audio: np.ndarray) -> float:
        """Calculate ratio of silence in audio"""
        threshold = 0.01 * np.max(np.abs(audio))
        silence_samples = np.sum(np.abs(audio) < threshold)
        return silence_samples / len(audio)

class CrossModalFusion:
    """Handles fusion of embeddings across modalities"""
    
    def __init__(self):
        self.fusion_methods = {
            'concatenation': self._concatenate_fusion,
            'attention': self._attention_fusion,
            'weighted_average': self._weighted_average_fusion,
            'canonical_correlation': self._cca_fusion
        }
    
    def fuse_embeddings(self, embeddings: List[ModalityEmbedding], 
                       method: str = 'attention') -> CrossModalResult:
        """Fuse embeddings from multiple modalities"""
        if not embeddings:
            raise ValueError("No embeddings provided for fusion")
        
        if len(embeddings) == 1:
            # Single modality - no fusion needed
            return CrossModalResult(
                input_modalities=[embeddings[0].modality],
                fused_embedding=embeddings[0].embedding,
                individual_embeddings=embeddings,
                fusion_confidence=embeddings[0].confidence,
                insights=self._generate_single_modal_insights(embeddings[0]),
                processing_metadata={'fusion_method': 'none', 'num_modalities': 1}
            )
        
        # Multi-modal fusion
        fusion_func = self.fusion_methods.get(method, self._attention_fusion)
        fused_embedding = fusion_func(embeddings)
        
        # Calculate fusion confidence
        fusion_confidence = self._calculate_fusion_confidence(embeddings)
        
        # Generate cross-modal insights
        insights = self._generate_cross_modal_insights(embeddings)
        
        return CrossModalResult(
            input_modalities=[emb.modality for emb in embeddings],
            fused_embedding=fused_embedding,
            individual_embeddings=embeddings,
            fusion_confidence=fusion_confidence,
            insights=insights,
            processing_metadata={
                'fusion_method': method,
                'num_modalities': len(embeddings),
                'embedding_dimensions': [len(emb.embedding) for emb in embeddings]
            }
        )
    
    def _concatenate_fusion(self, embeddings: List[ModalityEmbedding]) -> np.ndarray:
        """Simple concatenation fusion"""
        # Normalize embeddings to same dimension
        normalized_embeddings = []
        target_dim = 512  # Standard dimension
        
        for emb in embeddings:
            if len(emb.embedding) > target_dim:
                # Truncate
                normalized = emb.embedding[:target_dim]
            else:
                # Pad
                normalized = np.pad(emb.embedding, (0, target_dim - len(emb.embedding)))
            
            # L2 normalize
            norm = np.linalg.norm(normalized)
            if norm > 0:
                normalized = normalized / norm
            
            normalized_embeddings.append(normalized)
        
        return np.concatenate(normalized_embeddings)
    
    def _attention_fusion(self, embeddings: List[ModalityEmbedding]) -> np.ndarray:
        """Attention-based fusion"""
        # Normalize embeddings
        normalized_embeddings = []
        target_dim = 512
        
        for emb in embeddings:
            if len(emb.embedding) > target_dim:
                normalized = emb.embedding[:target_dim]
            else:
                normalized = np.pad(emb.embedding, (0, target_dim - len(emb.embedding)))
            
            norm = np.linalg.norm(normalized)
            if norm > 0:
                normalized = normalized / norm
            
            normalized_embeddings.append(normalized)
        
        # Calculate attention weights based on confidence and cross-modal similarity
        attention_weights = []
        for i, emb in enumerate(embeddings):
            # Base weight from confidence
            weight = emb.confidence
            
            # Add cross-modal similarity bonus
            for j, other_emb in enumerate(embeddings):
                if i != j:
                    similarity = np.dot(normalized_embeddings[i], normalized_embeddings[j])
                    weight += 0.1 * similarity  # Small bonus for complementary information
            
            attention_weights.append(weight)
        
        # Normalize attention weights
        attention_weights = np.array(attention_weights)
        attention_weights = attention_weights / np.sum(attention_weights)
        
        # Weighted combination
        fused = np.zeros(target_dim)
        for i, (emb_norm, weight) in enumerate(zip(normalized_embeddings, attention_weights)):
            fused += weight * emb_norm
        
        return fused
    
    def _weighted_average_fusion(self, embeddings: List[ModalityEmbedding]) -> np.ndarray:
        """Confidence-weighted average fusion"""
        normalized_embeddings = []
        weights = []
        target_dim = 512
        
        for emb in embeddings:
            if len(emb.embedding) > target_dim:
                normalized = emb.embedding[:target_dim]
            else:
                normalized = np.pad(emb.embedding, (0, target_dim - len(emb.embedding)))
            
            norm = np.linalg.norm(normalized)
            if norm > 0:
                normalized = normalized / norm
            
            normalized_embeddings.append(normalized)
            weights.append(emb.confidence)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Weighted average
        fused = np.zeros(target_dim)
        for emb_norm, weight in zip(normalized_embeddings, weights):
            fused += weight * emb_norm
        
        return fused
    
    def _cca_fusion(self, embeddings: List[ModalityEmbedding]) -> np.ndarray:
        """Canonical Correlation Analysis fusion (simplified)"""
        # For simplicity, fall back to attention fusion
        # In a full implementation, this would use proper CCA
        return self._attention_fusion(embeddings)
    
    def _calculate_fusion_confidence(self, embeddings: List[ModalityEmbedding]) -> float:
        """Calculate confidence in the fusion result"""
        individual_confidences = [emb.confidence for emb in embeddings]
        
        # Base confidence is weighted average
        base_confidence = np.mean(individual_confidences)
        
        # Bonus for multiple modalities (complementary information)
        modality_bonus = min(0.2, 0.05 * len(embeddings))
        
        # Penalty for low individual confidences
        min_confidence = min(individual_confidences)
        confidence_penalty = max(0, 0.3 - min_confidence)
        
        final_confidence = base_confidence + modality_bonus - confidence_penalty
        return max(0.0, min(1.0, final_confidence))
    
    def _generate_single_modal_insights(self, embedding: ModalityEmbedding) -> Dict[str, Any]:
        """Generate insights for single modality"""
        insights = {
            'modality': embedding.modality.value,
            'confidence': embedding.confidence,
            'processing_time': embedding.processing_time,
            'features': embedding.features
        }
        
        # Modality-specific insights
        if embedding.modality == ModalityType.TEXT:
            insights['text_analysis'] = {
                'complexity': embedding.features.get('readability_score', 0),
                'sentiment': embedding.features.get('sentiment_score', 0),
                'length_category': self._categorize_text_length(embedding.features.get('word_count', 0))
            }
        elif embedding.modality == ModalityType.IMAGE:
            insights['image_analysis'] = {
                'quality': self._assess_image_quality(embedding.features),
                'content_type': self._classify_image_content(embedding.features),
                'complexity': self._assess_image_complexity(embedding.features)
            }
        elif embedding.modality == ModalityType.AUDIO:
            insights['audio_analysis'] = {
                'quality': self._assess_audio_quality(embedding.features),
                'content_type': 'speech' if embedding.features.get('estimated_speech') else 'non_speech',
                'duration_category': self._categorize_audio_duration(embedding.features.get('duration_seconds', 0))
            }
        
        return insights
    
    def _generate_cross_modal_insights(self, embeddings: List[ModalityEmbedding]) -> Dict[str, Any]:
        """Generate insights from cross-modal analysis"""
        modalities = [emb.modality.value for emb in embeddings]
        
        insights = {
            'modalities_present': modalities,
            'cross_modal_coherence': self._assess_cross_modal_coherence(embeddings),
            'dominant_modality': self._identify_dominant_modality(embeddings),
            'complementarity_score': self._calculate_complementarity(embeddings)
        }
        
        # Specific cross-modal patterns
        if ModalityType.TEXT in [emb.modality for emb in embeddings] and \
           ModalityType.IMAGE in [emb.modality for emb in embeddings]:
            insights['text_image_alignment'] = self._assess_text_image_alignment(embeddings)
        
        if ModalityType.AUDIO in [emb.modality for emb in embeddings]:
            insights['audio_content_match'] = self._assess_audio_content_match(embeddings)
        
        return insights
    
    def _categorize_text_length(self, word_count: int) -> str:
        """Categorize text by length"""
        if word_count < 10:
            return 'very_short'
        elif word_count < 50:
            return 'short'
        elif word_count < 200:
            return 'medium'
        elif word_count < 1000:
            return 'long'
        else:
            return 'very_long'
    
    def _assess_image_quality(self, features: Dict[str, Any]) -> str:
        """Assess image quality"""
        resolution = features.get('total_pixels', 0)
        if resolution > 2000000:  # > 2MP
            return 'high'
        elif resolution > 500000:  # > 0.5MP
            return 'medium'
        else:
            return 'low'
    
    def _classify_image_content(self, features: Dict[str, Any]) -> str:
        """Classify image content type"""
        aspect_ratio = features.get('aspect_ratio', 1.0)
        if abs(aspect_ratio - 1.0) < 0.1:
            return 'square'
        elif aspect_ratio > 1.5:
            return 'landscape'
        elif aspect_ratio < 0.7:
            return 'portrait'
        else:
            return 'standard'
    
    def _assess_image_complexity(self, features: Dict[str, Any]) -> str:
        """Assess image complexity"""
        std_brightness = features.get('std_brightness', 0)
        if std_brightness > 80:
            return 'high'
        elif std_brightness > 40:
            return 'medium'
        else:
            return 'low'
    
    def _assess_audio_quality(self, features: Dict[str, Any]) -> str:
        """Assess audio quality"""
        sample_rate = features.get('sample_rate', 0)
        if sample_rate >= 44100:
            return 'high'
        elif sample_rate >= 22050:
            return 'medium'
        else:
            return 'low'
    
    def _categorize_audio_duration(self, duration: float) -> str:
        """Categorize audio by duration"""
        if duration < 5:
            return 'very_short'
        elif duration < 30:
            return 'short'
        elif duration < 300:  # 5 minutes
            return 'medium'
        elif duration < 1800:  # 30 minutes
            return 'long'
        else:
            return 'very_long'
    
    def _assess_cross_modal_coherence(self, embeddings: List[ModalityEmbedding]) -> float:
        """Assess how well modalities work together"""
        if len(embeddings) < 2:
            return 1.0
        
        # Calculate pairwise similarities between normalized embeddings
        similarities = []
        target_dim = 512
        
        normalized_embeddings = []
        for emb in embeddings:
            if len(emb.embedding) > target_dim:
                normalized = emb.embedding[:target_dim]
            else:
                normalized = np.pad(emb.embedding, (0, target_dim - len(emb.embedding)))
            
            norm = np.linalg.norm(normalized)
            if norm > 0:
                normalized = normalized / norm
            
            normalized_embeddings.append(normalized)
        
        for i in range(len(normalized_embeddings)):
            for j in range(i + 1, len(normalized_embeddings)):
                similarity = np.dot(normalized_embeddings[i], normalized_embeddings[j])
                similarities.append(abs(similarity))  # Use absolute value
        
        return np.mean(similarities) if similarities else 0.0
    
    def _identify_dominant_modality(self, embeddings: List[ModalityEmbedding]) -> str:
        """Identify the most confident/informative modality"""
        if not embeddings:
            return 'none'
        
        # Find modality with highest confidence
        best_embedding = max(embeddings, key=lambda x: x.confidence)
        return best_embedding.modality.value
    
    def _calculate_complementarity(self, embeddings: List[ModalityEmbedding]) -> float:
        """Calculate how complementary the modalities are"""
        if len(embeddings) < 2:
            return 0.0
        
        # Higher complementarity when modalities have different strengths
        confidences = [emb.confidence for emb in embeddings]
        confidence_variance = np.var(confidences)
        
        # Normalize to 0-1 range
        return min(1.0, confidence_variance * 4)  # Scale factor
    
    def _assess_text_image_alignment(self, embeddings: List[ModalityEmbedding]) -> float:
        """Assess alignment between text and image content"""
        # Simplified alignment assessment
        # In practice, this would use more sophisticated cross-modal models
        text_emb = next((emb for emb in embeddings if emb.modality == ModalityType.TEXT), None)
        image_emb = next((emb for emb in embeddings if emb.modality == ModalityType.IMAGE), None)
        
        if not text_emb or not image_emb:
            return 0.0
        
        # Simple heuristic based on confidence correlation
        confidence_diff = abs(text_emb.confidence - image_emb.confidence)
        alignment = 1.0 - confidence_diff
        
        return max(0.0, alignment)
    
    def _assess_audio_content_match(self, embeddings: List[ModalityEmbedding]) -> float:
        """Assess how well audio matches other content"""
        audio_emb = next((emb for emb in embeddings if emb.modality == ModalityType.AUDIO), None)
        
        if not audio_emb:
            return 0.0
        
        # Simple assessment based on audio features
        is_speech = audio_emb.features.get('estimated_speech', False)
        has_text = any(emb.modality == ModalityType.TEXT for emb in embeddings)
        
        if is_speech and has_text:
            return 0.8  # High match for speech + text
        elif is_speech and not has_text:
            return 0.4  # Medium match for speech without text
        else:
            return 0.6  # Default for non-speech audio

class MultimodalEngine:
    """Main multimodal processing engine"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
        self.audio_processor = AudioProcessor()
        self.video_processor = VideoProcessor()
        self.fusion_engine = CrossModalFusion()
        
        # Set cross-references for video processor
        self.video_processor.set_processors(self.image_processor, self.audio_processor)
        
        # Processing cache
        self.processing_cache = {}
        
        # Configuration
        self.config = {
            'max_file_size': 100 * 1024 * 1024,  # 100MB for video support
            'supported_image_formats': ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'],
            'supported_audio_formats': ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'],
            'supported_video_formats': ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
            'supported_text_formats': ['.txt', '.md', '.json', '.csv', '.xml', '.html'],
            'cache_ttl': 3600,  # 1 hour
            'fusion_method': 'attention',
            'enable_ocr': True,
            'enable_speech_recognition': True,
            'enable_object_detection': True
        }
        
        logger.info("MultimodalEngine initialized successfully with all modality processors")
    
    async def process_multimodal_input(self, inputs: List[MultimodalInput], 
                                     fusion_method: str = None) -> CrossModalResult:
        """Process multiple modality inputs and fuse them"""
        if not inputs:
            raise ValueError("No inputs provided")
        
        # Process each modality
        embeddings = []
        for input_data in inputs:
            try:
                embedding = await self._process_single_modality(input_data)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to process {input_data.modality.value}: {e}")
                continue
        
        if not embeddings:
            raise ValueError("Failed to process any inputs")
        
        # Fuse embeddings
        fusion_method = fusion_method or self.config['fusion_method']
        result = self.fusion_engine.fuse_embeddings(embeddings, fusion_method)
        
        # Cache result if Redis available
        if self.redis_client:
            await self._cache_result(inputs, result)
        
        return result
    
    async def _process_single_modality(self, input_data: MultimodalInput) -> ModalityEmbedding:
        """Process a single modality input"""
        # Check cache first
        cache_key = self._generate_cache_key(input_data)
        if self.redis_client:
            cached = await self._get_cached_embedding(cache_key)
            if cached:
                return cached
        
        # Process based on modality
        if input_data.modality == ModalityType.TEXT:
            embedding = self.text_processor.process_text(
                input_data.data, input_data.metadata
            )
        elif input_data.modality == ModalityType.IMAGE:
            embedding = self.image_processor.process_image(
                input_data.data, input_data.metadata
            )
        elif input_data.modality == ModalityType.AUDIO:
            embedding = self.audio_processor.process_audio(
                input_data.data, input_data.metadata
            )
        elif input_data.modality == ModalityType.VIDEO:
            embedding = self.video_processor.process_video(
                input_data.data, input_data.metadata
            )
        else:
            raise ValueError(f"Unsupported modality: {input_data.modality}")
        
        # Cache embedding
        if self.redis_client:
            await self._cache_embedding(cache_key, embedding)
        
        return embedding
    
    def _generate_cache_key(self, input_data: MultimodalInput) -> str:
        """Generate cache key for input"""
        import hashlib
        
        # Create hash of data
        if isinstance(input_data.data, str):
            data_hash = hashlib.md5(input_data.data.encode()).hexdigest()
        elif isinstance(input_data.data, bytes):
            data_hash = hashlib.md5(input_data.data).hexdigest()
        else:
            data_hash = hashlib.md5(str(input_data.data).encode()).hexdigest()
        
        return f"multimodal:{input_data.modality.value}:{data_hash}"
    
    async def _get_cached_embedding(self, cache_key: str) -> Optional[ModalityEmbedding]:
        """Get cached embedding"""
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                return ModalityEmbedding(
                    modality=ModalityType(data['modality']),
                    embedding=np.array(data['embedding']),
                    features=data['features'],
                    confidence=data['confidence'],
                    processing_time=data['processing_time']
                )
        except Exception as e:
            logger.warning(f"Failed to get cached embedding: {e}")
        
        return None
    
    async def _cache_embedding(self, cache_key: str, embedding: ModalityEmbedding):
        """Cache embedding"""
        try:
            data = {
                'modality': embedding.modality.value,
                'embedding': embedding.embedding.tolist(),
                'features': embedding.features,
                'confidence': embedding.confidence,
                'processing_time': embedding.processing_time
            }
            
            await self.redis_client.setex(
                cache_key, 
                self.config['cache_ttl'], 
                json.dumps(data)
            )
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")
    
    async def _cache_result(self, inputs: List[MultimodalInput], result: CrossModalResult):
        """Cache fusion result"""
        try:
            # Generate cache key for the combination
            input_hashes = []
            for inp in inputs:
                cache_key = self._generate_cache_key(inp)
                input_hashes.append(cache_key)
            
            result_key = f"fusion:{':'.join(sorted(input_hashes))}"
            
            # Serialize result
            data = {
                'input_modalities': [mod.value for mod in result.input_modalities],
                'fused_embedding': result.fused_embedding.tolist(),
                'fusion_confidence': result.fusion_confidence,
                'insights': result.insights,
                'processing_metadata': result.processing_metadata
            }
            
            await self.redis_client.setex(
                result_key,
                self.config['cache_ttl'],
                json.dumps(data)
            )
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
    
    async def process_text(self, text: str, metadata: Dict[str, Any] = None) -> ModalityEmbedding:
        """Process text input"""
        input_data = MultimodalInput(
            modality=ModalityType.TEXT,
            data=text,
            metadata=metadata or {},
            timestamp=datetime.now()
        )
        return await self._process_single_modality(input_data)
    
    async def process_image(self, image_data: Union[bytes, str], 
                          metadata: Dict[str, Any] = None) -> ModalityEmbedding:
        """Process image input"""
        input_data = MultimodalInput(
            modality=ModalityType.IMAGE,
            data=image_data,
            metadata=metadata or {},
            timestamp=datetime.now()
        )
        return await self._process_single_modality(input_data)
    
    async def process_audio(self, audio_data: Union[bytes, str], 
                          metadata: Dict[str, Any] = None) -> ModalityEmbedding:
        """Process audio input"""
        input_data = MultimodalInput(
            modality=ModalityType.AUDIO,
            data=audio_data,
            metadata=metadata or {},
            timestamp=datetime.now()
        )
        return await self._process_single_modality(input_data)
    
    async def process_video(self, video_data: Union[bytes, str], 
                          metadata: Dict[str, Any] = None) -> ModalityEmbedding:
        """Process video input"""
        input_data = MultimodalInput(
            modality=ModalityType.VIDEO,
            data=video_data,
            metadata=metadata or {},
            timestamp=datetime.now()
        )
        return await self._process_single_modality(input_data)
    
    async def search_similar(self, query_embedding: np.ndarray, 
                           modality_filter: Optional[ModalityType] = None,
                           limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar embeddings"""
        # This would integrate with a vector database in production
        # For now, return empty results
        logger.info(f"Searching for similar embeddings (modality: {modality_filter}, limit: {limit})")
        return []
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get supported file formats"""
        return {
            'image': self.config['supported_image_formats'],
            'audio': self.config['supported_audio_formats'],
            'text': ['.txt', '.md', '.json', '.csv']
        }
    
    def validate_input_size(self, data: Union[str, bytes]) -> bool:
        """Validate input size"""
        if isinstance(data, str):
            size = len(data.encode('utf-8'))
        else:
            size = len(data)
        
        return size <= self.config['max_file_size']
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        # In production, this would query actual metrics
        return {
            'total_processed': 0,
            'by_modality': {
                'text': 0,
                'image': 0,
                'audio': 0
            },
            'average_processing_time': {
                'text': 0.1,
                'image': 0.5,
                'audio': 0.3
            },
            'cache_hit_rate': 0.0
        }

# Global multimodal engine instance
_multimodal_engine = None

def get_multimodal_engine(redis_client=None) -> MultimodalEngine:
    """Get global multimodal engine instance"""
    global _multimodal_engine
    
    if _multimodal_engine is None:
        _multimodal_engine = MultimodalEngine(redis_client)
    
    return _multimodal_engine

async def initialize_multimodal_engine():
    """Initialize the multimodal engine"""
    logger.info("Initializing ScrollIntel MultimodalEngine")
    
    # Check dependencies
    missing_deps = []
    if not HAS_TORCH:
        missing_deps.append("PyTorch")
    if not HAS_TENSORFLOW:
        missing_deps.append("TensorFlow")
    if not HAS_OPENCV:
        missing_deps.append("OpenCV")
    if not HAS_PIL:
        missing_deps.append("Pillow")
    if not HAS_LIBROSA:
        missing_deps.append("librosa")
    
    if missing_deps:
        logger.warning(f"Missing optional dependencies: {', '.join(missing_deps)}")
        logger.warning("Some multimodal features may not be available")
    
    # Initialize global engine
    engine = get_multimodal_engine()
    
    logger.info("MultimodalEngine initialization completed")
    return engine