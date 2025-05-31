from dotenv import load_dotenv
import base64
import io
import json
import os
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from PIL import Image, ImageFilter, ImageStat
import numpy as np
from PIL.ExifTags import TAGS, GPSTAGS
import openai
import aiohttp
import asyncio
import logging
from functools import lru_cache
import httpx
import psutil
import piexif  # New library for better EXIF handling
import exifread  # Alternative EXIF library
from geopy.geocoders import Nominatim  # For reverse geocoding
from config import (
    OPENAI_API_KEY, AVAILABLE_MODELS, DEFAULT_MODEL, VISION_DETAIL_LEVEL,
    QUALITY_RATIO, RESIZE_RATIO_SINGLE, RESIZE_RATIO_MULTIPLE,
    MAX_WIDTH_SINGLE, MAX_HEIGHT_SINGLE, MAX_WIDTH_MULTIPLE, MAX_HEIGHT_MULTIPLE,
    TEMPERATURE, PRESENCE_PENALTY, FREQUENCY_PENALTY, MAX_MEMORY_USAGE_MB
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ImageInput:
    """Input structure for base64 images"""
    image_base64: str
    imageid: str

@dataclass
class OpenAIParams:
    """OpenAI API parameters"""
    max_tokens: int = 600
    detail: str = VISION_DETAIL_LEVEL
    model: str = DEFAULT_MODEL
    temperature: float = TEMPERATURE
    presence_penalty: float = PRESENCE_PENALTY
    frequency_penalty: float = FREQUENCY_PENALTY

@dataclass
class ImageManipulationParams:
    """Image processing parameters"""
    quality_ratio: float = QUALITY_RATIO
    resize_ratio: float = RESIZE_RATIO_SINGLE
    max_width: int = MAX_WIDTH_SINGLE
    max_height: int = MAX_HEIGHT_SINGLE

@dataclass
class EXIFData:
    """Enhanced EXIF metadata structure"""
    datetime: Optional[str] = None
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    gps_altitude: Optional[float] = None
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    orientation: Optional[int] = None
    iso: Optional[int] = None
    focal_length: Optional[str] = None
    aperture: Optional[str] = None
    shutter_speed: Optional[str] = None
    flash: Optional[str] = None

@dataclass
class ImageAnalysis:
    """Individual image analysis result"""
    imageid: str
    quality_score: float
    quality_metrics: Dict[str, Any]
    exif_data: EXIFData
    analysis: Dict[str, Any]

@dataclass
class CollectiveAnalysis:
    """Event-level collective analysis result"""
    event_insights: Dict[str, Any]
    image_analyses: List[ImageAnalysis]
    total_images: int
    average_quality: float

class ImageQualityScorer:
    """Compute image quality score based on multiple factors"""
    def analyze_image_quality(self, img):
        try:
            gray_img = img.convert('L')
            stat = ImageStat.Stat(img)
            stat_gray = ImageStat.Stat(gray_img)
            brightness = sum(stat.mean[:3]) / 3
            contrast = sum(stat.stddev[:3]) / 3
            r, g, b = stat.mean[:3]
            perceived_brightness = (0.299 * r + 0.587 * g + 0.114 * b)
            sharpness = min(100, contrast * 2)
            noise_estimate = min(100, (stat_gray.stddev[0] / 2.5))
            brightness_score = min(100, max(0, 100 - abs(perceived_brightness - 128) / 1.28))
            contrast_score = min(100, contrast)
            overall_score = (
                    0.2 * brightness_score +
                    0.3 * contrast_score +
                    0.4 * sharpness +
                    0.1 * (100 - noise_estimate)
            )
            return {
                "overall_quality": f"{overall_score:.2f}/100",
                "brightness": f"{brightness:.2f}/255",
                "perceived_brightness": f"{perceived_brightness:.2f}/255",
                "contrast": f"{contrast:.2f}",
                "sharpness": f"{sharpness:.2f}/100",
                "noise_estimate": f"{noise_estimate:.2f}/100",
                "quality_category": self.get_quality_category(overall_score)
            }, overall_score
        except Exception as e:
            logger.warning(f"Error analyzing image quality: {e}")
            return {"overall_quality": "Error analyzing", "error": str(e)}, 50.0

    def get_quality_category(self, score):
        if score >= 90:
            return "Excellent"
        elif score >= 80:
            return "Very Good"
        elif score >= 70:
            return "Good"
        elif score >= 60:
            return "Fair"
        elif score >= 50:
            return "Poor"
        else:
            return "Very Poor"

    @staticmethod
    @lru_cache(maxsize=500)
    def calculate_sharpness(image_data: bytes) -> float:
        image = Image.open(io.BytesIO(image_data)).convert('L')
        np_image = np.array(image)
        laplacian = np.abs(np.gradient(np.gradient(np_image, axis=0), axis=0) +
                           np.gradient(np.gradient(np_image, axis=1)))
        variance = np.var(laplacian)
        return min(100, variance / 1000 * 100)

    @staticmethod
    @lru_cache(maxsize=500)
    def calculate_noise_level(image_data: bytes) -> float:
        image = Image.open(io.BytesIO(image_data)).convert('L')
        np_image = np.array(image)
        patch_size = 4
        noise_values = []
        for i in range(0, np_image.shape[0] - patch_size, patch_size * 2):
            for j in range(0, np_image.shape[1] - patch_size, patch_size * 2):
                patch = np_image[i:i + patch_size, j:j + patch_size]
                noise_values.append(np.std(patch))
        avg_noise = np.mean(noise_values) if noise_values else 0
        return max(0, 100 - (avg_noise / 255 * 100))

    @staticmethod
    def calculate_blur_detection(image: Image.Image) -> float:
        gray_image = image.convert('L')
        filtered = gray_image.filter(ImageFilter.FIND_EDGES)
        stat = ImageStat.Stat(filtered)
        edge_strength = stat.mean[0]
        return min(100, edge_strength / 255 * 100 * 2)

    def calculate_quality_score(self, image: Image.Image) -> Tuple[float, Dict[str, Any]]:
        try:
            quality_metrics, overall_score = self.analyze_image_quality(image)
            return round(overall_score, 2), quality_metrics
        except Exception as e:
            logger.warning(f"Error in calculate_quality_score: {e}")
            return 50.0, {"overall_quality": "Error analyzing", "error": str(e)}

class EnhancedEXIFExtractor:
    """Enhanced EXIF extractor with multiple library support and better GPS handling"""

    def __init__(self):
        self.geocoder = Nominatim(user_agent="image_analyzer")

    def extract_exif_data(self, image_data: bytes) -> EXIFData:
        """Extract EXIF data using multiple methods for better reliability"""
        # Try method 1: PIL with improved GPS handling
        exif_data = self._extract_with_pil(image_data)

        # If GPS data is missing, try method 2: piexif
        if exif_data.gps_latitude is None or exif_data.gps_longitude is None:
            piexif_data = self._extract_with_piexif(image_data)
            if piexif_data.gps_latitude is not None:
                exif_data.gps_latitude = piexif_data.gps_latitude
                exif_data.gps_longitude = piexif_data.gps_longitude
                exif_data.gps_altitude = piexif_data.gps_altitude

        # If still missing, try method 3: exifread
        if exif_data.gps_latitude is None or exif_data.gps_longitude is None:
            exifread_data = self._extract_with_exifread(image_data)
            if exifread_data.gps_latitude is not None:
                exif_data.gps_latitude = exifread_data.gps_latitude
                exif_data.gps_longitude = exifread_data.gps_longitude
                exif_data.gps_altitude = exifread_data.gps_altitude

        logger.info(f"Final extracted EXIF data: {asdict(exif_data)}")
        return exif_data

    def _extract_with_pil(self, image_data: bytes) -> EXIFData:
        """Extract EXIF using PIL with enhanced GPS handling"""
        exif_data = EXIFData()
        try:
            image = Image.open(io.BytesIO(image_data))

            # Try multiple methods to get EXIF
            exif_dict = None

            # Method 1: Modern getexif()
            try:
                exif_dict = image.getexif()
                logger.info(f"PIL getexif() found {len(exif_dict)} tags")
            except AttributeError:
                pass

            # Method 2: Legacy _getexif()
            if not exif_dict:
                try:
                    exif_dict = image._getexif()
                    logger.info(f"PIL _getexif() found {len(exif_dict or {})} tags")
                except AttributeError:
                    pass

            if not exif_dict:
                logger.warning("No EXIF data found with PIL")
                return exif_data

            # Extract standard EXIF data
            for tag_id, value in exif_dict.items():
                tag = TAGS.get(tag_id, tag_id)
                try:
                    if tag == "DateTime" and value:
                        exif_data.datetime = str(value).strip()
                    elif tag == "Make" and value:
                        exif_data.camera_make = str(value).strip()
                    elif tag == "Model" and value:
                        exif_data.camera_model = str(value).strip()
                    elif tag == "Orientation" and value:
                        exif_data.orientation = int(value)
                    elif tag == "ISOSpeedRatings" and value:
                        exif_data.iso = int(value)
                    elif tag == "FocalLength" and value:
                        exif_data.focal_length = str(value)
                    elif tag == "FNumber" and value:
                        exif_data.aperture = str(value)
                    elif tag == "ExposureTime" and value:
                        exif_data.shutter_speed = str(value)
                    elif tag == "Flash" and value:
                        exif_data.flash = str(value)
                    elif tag == "GPSInfo" and value:
                        gps_data = self._extract_gps_data_pil(value)
                        exif_data.gps_latitude = gps_data.get('latitude')
                        exif_data.gps_longitude = gps_data.get('longitude')
                        exif_data.gps_altitude = gps_data.get('altitude')
                        logger.info(f"PIL GPS extracted: lat={exif_data.gps_latitude}, lon={exif_data.gps_longitude}")
                except Exception as e:
                    logger.error(f"Error processing PIL EXIF tag {tag}: {str(e)}")

            image.close()
            return exif_data

        except Exception as e:
            logger.error(f"PIL EXIF extraction failed: {str(e)}")
            return exif_data

    def _extract_gps_data_pil(self, gps_info: Dict) -> Dict[str, Optional[float]]:
        """Enhanced GPS extraction from PIL GPSInfo"""

        def convert_to_degrees(value):
            try:
                if isinstance(value, (list, tuple)) and len(value) >= 3:
                    d, m, s = value[0], value[1], value[2]

                    # Handle different value types (fractions, integers, floats)
                    def to_float(val):
                        if hasattr(val, 'num') and hasattr(val, 'den'):  # Fraction
                            return float(val.num) / float(val.den) if val.den != 0 else 0
                        return float(val)

                    d_float = to_float(d)
                    m_float = to_float(m)
                    s_float = to_float(s)

                    return d_float + (m_float / 60.0) + (s_float / 3600.0)
                elif isinstance(value, (int, float)):
                    return float(value)
                return None
            except Exception as e:
                logger.error(f"Error converting GPS coordinates: {str(e)}")
                return None

        gps_data = {'latitude': None, 'longitude': None, 'altitude': None}

        try:
            # Check different key formats (numeric and string)
            lat_keys = [1, 2, 'GPSLatitude']  # GPSLatitudeRef, GPSLatitude
            lon_keys = [3, 4, 'GPSLongitude']  # GPSLongitudeRef, GPSLongitude
            alt_keys = [5, 6, 'GPSAltitude']  # GPSAltitudeRef, GPSAltitude

            # Extract latitude
            lat_ref = None
            lat_val = None
            for key in [1, 'GPSLatitudeRef']:
                if key in gps_info:
                    lat_ref = gps_info[key]
                    break
            for key in [2, 'GPSLatitude']:
                if key in gps_info:
                    lat_val = convert_to_degrees(gps_info[key])
                    break

            if lat_val is not None and lat_ref:
                lat_ref_str = str(lat_ref).upper()
                gps_data['latitude'] = lat_val if lat_ref_str == 'N' else -lat_val
                logger.debug(f"Latitude: {gps_data['latitude']} (ref: {lat_ref_str})")

            # Extract longitude
            lon_ref = None
            lon_val = None
            for key in [3, 'GPSLongitudeRef']:
                if key in gps_info:
                    lon_ref = gps_info[key]
                    break
            for key in [4, 'GPSLongitude']:
                if key in gps_info:
                    lon_val = convert_to_degrees(gps_info[key])
                    break

            if lon_val is not None and lon_ref:
                lon_ref_str = str(lon_ref).upper()
                gps_data['longitude'] = lon_val if lon_ref_str == 'E' else -lon_val
                logger.debug(f"Longitude: {gps_data['longitude']} (ref: {lon_ref_str})")

            # Extract altitude
            for key in [6, 'GPSAltitude']:
                if key in gps_info:
                    alt_val = convert_to_degrees(gps_info[key])
                    if alt_val is not None:
                        gps_data['altitude'] = alt_val
                        logger.debug(f"Altitude: {gps_data['altitude']}")
                    break

        except Exception as e:
            logger.error(f"Error extracting GPS data with PIL: {str(e)}")

        return gps_data

    def _extract_with_piexif(self, image_data: bytes) -> EXIFData:
        """Extract EXIF using piexif library"""
        exif_data = EXIFData()
        try:
            exif_dict = piexif.load(image_data)
            logger.info(f"piexif found data in sections: {list(exif_dict.keys())}")

            # Extract GPS data
            if "GPS" in exif_dict and exif_dict["GPS"]:
                gps_ifd = exif_dict["GPS"]
                logger.info(f"piexif GPS tags found: {list(gps_ifd.keys())}")

                # Extract latitude
                if piexif.GPSIFD.GPSLatitude in gps_ifd and piexif.GPSIFD.GPSLatitudeRef in gps_ifd:
                    lat_dms = gps_ifd[piexif.GPSIFD.GPSLatitude]
                    lat_ref = gps_ifd[piexif.GPSIFD.GPSLatitudeRef].decode('utf-8')
                    lat_dd = self._dms_to_dd(lat_dms)
                    if lat_dd is not None:
                        exif_data.gps_latitude = lat_dd if lat_ref == 'N' else -lat_dd

                # Extract longitude
                if piexif.GPSIFD.GPSLongitude in gps_ifd and piexif.GPSIFD.GPSLongitudeRef in gps_ifd:
                    lon_dms = gps_ifd[piexif.GPSIFD.GPSLongitude]
                    lon_ref = gps_ifd[piexif.GPSIFD.GPSLongitudeRef].decode('utf-8')
                    lon_dd = self._dms_to_dd(lon_dms)
                    if lon_dd is not None:
                        exif_data.gps_longitude = lon_dd if lon_ref == 'E' else -lon_dd

                # Extract altitude
                if piexif.GPSIFD.GPSAltitude in gps_ifd:
                    alt_data = gps_ifd[piexif.GPSIFD.GPSAltitude]
                    if isinstance(alt_data, tuple) and len(alt_data) == 2:
                        exif_data.gps_altitude = float(alt_data[0]) / float(alt_data[1])

                logger.info(f"piexif GPS: lat={exif_data.gps_latitude}, lon={exif_data.gps_longitude}")

            # Extract other EXIF data
            if "Exif" in exif_dict:
                exif_ifd = exif_dict["Exif"]
                if piexif.ExifIFD.DateTimeOriginal in exif_ifd:
                    exif_data.datetime = exif_ifd[piexif.ExifIFD.DateTimeOriginal].decode('utf-8')

            if "0th" in exif_dict:
                zeroth_ifd = exif_dict["0th"]
                if piexif.ImageIFD.Make in zeroth_ifd:
                    exif_data.camera_make = zeroth_ifd[piexif.ImageIFD.Make].decode('utf-8').strip()
                if piexif.ImageIFD.Model in zeroth_ifd:
                    exif_data.camera_model = zeroth_ifd[piexif.ImageIFD.Model].decode('utf-8').strip()

        except Exception as e:
            logger.error(f"piexif extraction failed: {str(e)}")

        return exif_data

    def _extract_with_exifread(self, image_data: bytes) -> EXIFData:
        """Extract EXIF using exifread library"""
        exif_data = EXIFData()
        try:
            tags = exifread.process_file(io.BytesIO(image_data), details=False)
            logger.info(f"exifread found {len(tags)} tags")

            # Extract GPS data
            gps_lat = tags.get('GPS GPSLatitude')
            gps_lat_ref = tags.get('GPS GPSLatitudeRef')
            gps_lon = tags.get('GPS GPSLongitude')
            gps_lon_ref = tags.get('GPS GPSLongitudeRef')

            if gps_lat and gps_lat_ref:
                lat_dd = self._exifread_gps_to_dd(str(gps_lat))
                if lat_dd is not None:
                    exif_data.gps_latitude = lat_dd if str(gps_lat_ref) == 'N' else -lat_dd

            if gps_lon and gps_lon_ref:
                lon_dd = self._exifread_gps_to_dd(str(gps_lon))
                if lon_dd is not None:
                    exif_data.gps_longitude = lon_dd if str(gps_lon_ref) == 'E' else -lon_dd

            # Extract altitude
            gps_alt = tags.get('GPS GPSAltitude')
            if gps_alt:
                try:
                    alt_str = str(gps_alt)
                    if '/' in alt_str:
                        num, den = alt_str.split('/')
                        exif_data.gps_altitude = float(num) / float(den)
                    else:
                        exif_data.gps_altitude = float(alt_str)
                except:
                    pass

            logger.info(f"exifread GPS: lat={exif_data.gps_latitude}, lon={exif_data.gps_longitude}")

            # Extract other data
            if 'EXIF DateTimeOriginal' in tags:
                exif_data.datetime = str(tags['EXIF DateTimeOriginal'])
            elif 'Image DateTime' in tags:
                exif_data.datetime = str(tags['Image DateTime'])

            if 'Image Make' in tags:
                exif_data.camera_make = str(tags['Image Make']).strip()
            if 'Image Model' in tags:
                exif_data.camera_model = str(tags['Image Model']).strip()

        except Exception as e:
            logger.error(f"exifread extraction failed: {str(e)}")

        return exif_data

    def _dms_to_dd(self, dms_coords) -> Optional[float]:
        """Convert DMS (Degrees, Minutes, Seconds) to DD (Decimal Degrees)"""
        try:
            if isinstance(dms_coords, (list, tuple)) and len(dms_coords) >= 3:
                degrees = float(dms_coords[0][0]) / float(dms_coords[0][1]) if dms_coords[0][1] != 0 else 0
                minutes = float(dms_coords[1][0]) / float(dms_coords[1][1]) if dms_coords[1][1] != 0 else 0
                seconds = float(dms_coords[2][0]) / float(dms_coords[2][1]) if dms_coords[2][1] != 0 else 0
                return degrees + (minutes / 60.0) + (seconds / 3600.0)
            return None
        except Exception as e:
            logger.error(f"Error converting DMS to DD: {str(e)}")
            return None

    def _exifread_gps_to_dd(self, gps_str: str) -> Optional[float]:
        """Convert exifread GPS string to decimal degrees"""
        try:
            # Format: [DD/1, MM/1, SS/1] or similar
            gps_str = gps_str.strip('[]')
            parts = gps_str.split(', ')

            degrees = 0
            minutes = 0
            seconds = 0

            if len(parts) >= 1:
                deg_part = parts[0].strip()
                if '/' in deg_part:
                    num, den = deg_part.split('/')
                    degrees = float(num) / float(den) if float(den) != 0 else 0
                else:
                    degrees = float(deg_part)

            if len(parts) >= 2:
                min_part = parts[1].strip()
                if '/' in min_part:
                    num, den = min_part.split('/')
                    minutes = float(num) / float(den) if float(den) != 0 else 0
                else:
                    minutes = float(min_part)

            if len(parts) >= 3:
                sec_part = parts[2].strip()
                if '/' in sec_part:
                    num, den = sec_part.split('/')
                    seconds = float(num) / float(den) if float(den) != 0 else 0
                else:
                    seconds = float(sec_part)

            return degrees + (minutes / 60.0) + (seconds / 3600.0)
        except Exception as e:
            logger.error(f"Error parsing exifread GPS string '{gps_str}': {str(e)}")
            return None

    def get_address_from_coordinates(self, latitude: float, longitude: float) -> Optional[str]:
        """Get address from GPS coordinates using reverse geocoding"""
        try:
            location = self.geocoder.reverse(f"{latitude}, {longitude}", timeout=10)
            return location.address if location else None
        except Exception as e:
            logger.error(f"Reverse geocoding failed: {str(e)}")
            return None

class ImageProcessor:
    """Handle image manipulation and preprocessing"""
    @staticmethod
    def process_image(image: Image.Image, params: ImageManipulationParams) -> Image.Image:
        processed_image = image.copy()
        if params.resize_ratio != 1.0:
            new_width = int(image.width * params.resize_ratio)
            new_height = int(image.height * params.resize_ratio)
            processed_image = processed_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        if processed_image.width > params.max_width or processed_image.height > params.max_height:
            processed_image.thumbnail((params.max_width, params.max_height), Image.Resampling.LANCZOS)
        return processed_image

    @staticmethod
    def image_to_base64(image: Image.Image, quality: int = 40) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality, optimize=True)
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

class OpenAIImageAnalyzer:
    """Main integration class for OpenAI image analysis"""
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.api_key = api_key
        self.quality_scorer = ImageQualityScorer()
        self.exif_extractor = EnhancedEXIFExtractor()
        self.image_processor = ImageProcessor()

    def get_api_key_status(self) -> Dict[str, Any]:
        """Return API key status for front-end"""
        return {
            "api_key_set": bool(self.api_key),
            "api_key_prefix": self.api_key[:4] if self.api_key else None
        }

    def load_images_from_paths(self, image_paths: Union[str, List[str]]) -> List[ImageInput]:
        if isinstance(image_paths, str):
            paths = [p.strip() for p in image_paths.split(',')]
        else:
            paths = image_paths
        images = []
        for i, path in enumerate(paths):
            try:
                with open(path, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                    images.append(ImageInput(image_base64=image_data, imageid=f"img_{i}_{os.path.basename(path)}"))
            except Exception as e:
                logger.error(f"Failed to load image {path}: {e}")
        return images

    def preprocess_image(
            self,
            image_input: ImageInput,
            manipulation_params: ImageManipulationParams
    ) -> Tuple[Image.Image, str, float, Dict[str, Any], EXIFData]:
        start_time = time.time()
        # Check memory usage before processing
        process = psutil.Process(os.getpid())
        mem_usage_mb = process.memory_info().rss / 1024 / 1024
        if mem_usage_mb > MAX_MEMORY_USAGE_MB:
            raise MemoryError(f"Memory usage exceeds limit of {MAX_MEMORY_USAGE_MB}MB: {mem_usage_mb:.2f}MB")

        image_data = base64.b64decode(image_input.image_base64)
        exif_data = self.exif_extractor.extract_exif_data(image_data)
        image = Image.open(io.BytesIO(image_data))
        quality_score, quality_metrics = self.quality_scorer.calculate_quality_score(image)
        processed_image = self.image_processor.process_image(image, manipulation_params)
        processed_base64 = self.image_processor.image_to_base64(
            processed_image, quality=int(manipulation_params.quality_ratio * 100))

        # Clean up
        image.close()
        processed_image.close()

        logger.debug(f"Preprocess image {image_input.imageid}: {time.time() - start_time:.2f}s")
        return processed_image, processed_base64, quality_score, quality_metrics, exif_data

    def create_openai_message(
            self,
            user_prompt: str,
            system_prompt: str,
            images_data: List[Tuple],
            openai_params: OpenAIParams
    ) -> Dict[str, Any]:
        """Create OpenAI API message structure with optimized prompts"""
        enhanced_system_prompt = system_prompt + " Return concise JSON with eventName for events."
        modified_prompt = user_prompt
        for _, _, _, _, exif_data in images_data:
            if exif_data.gps_latitude is not None and exif_data.gps_longitude is not None:
                modified_prompt += f"\nImage GPS Coordinates: Latitude {exif_data.gps_latitude}, Longitude {exif_data.gps_longitude}"
        modified_prompt += "\nReturn concise JSON with eventName for events."
        final_detail_level = "low"
        content = [{"type": "text", "text": modified_prompt}]
        for _, image_base64, _, _, _ in images_data:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}",
                    "detail": final_detail_level
                }
            })
        model_to_use = openai_params.model if openai_params.model in AVAILABLE_MODELS else DEFAULT_MODEL
        api_params = {
            "model": model_to_use,
            "messages": [
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": content}
            ],
            "max_tokens": openai_params.max_tokens,
            "temperature": openai_params.temperature,
            "presence_penalty": openai_params.presence_penalty,
            "frequency_penalty": openai_params.frequency_penalty
        }
        if model_to_use in ["gpt-4.1", "gpt-4o", "gpt-4-turbo"]:
            api_params["response_format"] = {"type": "json_object"}
        return api_params

    async def async_openai_call(
            self,
            session: aiohttp.ClientSession,
            request_data: Dict[str, Any],
            image_input: ImageInput,
            quality_score: float,
            quality_metrics: Dict[str, Any],
            exif_data: EXIFData
    ) -> ImageAnalysis:
        start_time = time.time()
        try:
            async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    json=request_data,
                    headers={"Authorization": f"Bearer {self.client.api_key}"}
            ) as response:
                data = await response.json()
                if 'error' in data:
                    raise Exception(data['error']['message'])
                analysis_result = json.loads(data['choices'][0]['message']['content'])
                analysis_result = analysis_result[0] if isinstance(analysis_result, list) else analysis_result
                analysis_result['qualityScore'] = quality_score
                analysis_result['_metadata'] = {
                    "time_taken": f"{time.time() - start_time:.2f} seconds",
                    "model": request_data["model"]
                }
                logger.debug(f"Async API call for {image_input.imageid}: {time.time() - start_time:.2f}s")
                return ImageAnalysis(
                    imageid=image_input.imageid,
                    quality_score=quality_score,
                    quality_metrics=quality_metrics,
                    exif_data=exif_data,
                    analysis=analysis_result
                )
        except httpx.TimeoutException as e:
            logger.error(f"Timeout during API call for {image_input.imageid}: {str(e)}")
            return ImageAnalysis(
                imageid=image_input.imageid,
                quality_score=quality_score,
                quality_metrics=quality_metrics,
                exif_data=exif_data,
                analysis={"error": "OpenAI API call timed out."}
            )
        except httpx.RequestError as e:
            logger.error(f"Network error during API call for {image_input.imageid}: {str(e)}")
            return ImageAnalysis(
                imageid=image_input.imageid,
                quality_score=quality_score,
                quality_metrics=quality_metrics,
                exif_data=exif_data,
                analysis={"error": "Network error occurred."}
            )
        except Exception as e:
            logger.error(f"Async API call failed for {image_input.imageid}: {e}")
            return ImageAnalysis(
                imageid=image_input.imageid,
                quality_score=quality_score,
                quality_metrics=quality_metrics,
                exif_data=exif_data,
                analysis={"error": str(e), "qualityScore": quality_score}
            )

    async def analyze_images_individual(
            self,
            images: List[ImageInput],
            user_prompt: str,
            system_prompt: str,
            openai_params: OpenAIParams = OpenAIParams(),
            manipulation_params: ImageManipulationParams = ImageManipulationParams()
    ) -> Tuple[List[ImageAnalysis], Dict[str, Any]]:
        start_time = time.time()
        results = []
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        cost = {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}

        async with aiohttp.ClientSession() as session:
            tasks = []
            for image_input in images:
                _, processed_base64, quality_score, quality_metrics, exif_data = self.preprocess_image(
                    image_input, manipulation_params)
                request_data = self.create_openai_message(
                    user_prompt, system_prompt, [(None, processed_base64, quality_score, quality_metrics, exif_data)],
                    openai_params)
                tasks.append(self.async_openai_call(session, request_data, image_input, quality_score, quality_metrics,
                                                    exif_data))
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        for response in responses:
            if isinstance(response, ImageAnalysis):
                results.append(response)
                token_usage["prompt_tokens"] += 170
                token_usage["completion_tokens"] += openai_params.max_tokens // len(images)
                token_usage["total_tokens"] = token_usage["prompt_tokens"] + token_usage["completion_tokens"]

        if openai_params.model == "gpt-4.1":
            cost["input_cost"] = (token_usage["prompt_tokens"] / 1000) * 0.005
            cost["output_cost"] = (token_usage["completion_tokens"] / 1000) * 0.015
            cost["total_cost"] = cost["input_cost"] + cost["output_cost"]

        time_taken = time.time() - start_time
        logger.info(
            f"Individual Analysis: {len(images)} images, Time: {time_taken:.2f}s, Token Usage: {token_usage}, Cost: ${cost['total_cost']:.6f}")

        return results, {
            "token_usage": token_usage,
            "time_taken": round(time_taken, 2),
            "cost": cost,
            "model": openai_params.model
        }

    def analyze_images_collective(
            self,
            images: List[ImageInput],
            user_prompt: str,
            system_prompt: str,
            openai_params: OpenAIParams = OpenAIParams(),
            manipulation_params: ImageManipulationParams = ImageManipulationParams()
    ) -> Tuple[CollectiveAnalysis, Dict[str, Any]]:
        start_time = time.time()
        processed_images = []
        individual_analyses = []
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        cost = {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}

        for image_input in images:
            try:
                _, processed_base64, quality_score, quality_metrics, exif_data = self.preprocess_image(
                    image_input, manipulation_params)
                processed_images.append((image_input, processed_base64, quality_score, quality_metrics, exif_data))
                individual_analyses.append(ImageAnalysis(
                    imageid=image_input.imageid,
                    quality_score=quality_score,
                    quality_metrics=quality_metrics,
                    exif_data=exif_data,
                    analysis={}
                ))
            except Exception as e:
                logger.error(f"Failed to preprocess image {image_input.imageid}: {e}")
                individual_analyses.append(ImageAnalysis(
                    imageid=image_input.imageid,
                    quality_score=0.0,
                    quality_metrics={"error": str(e)},
                    exif_data=EXIFData(),
                    analysis={"error": str(e)}
                ))

        exif_context = self._create_exif_context([data[4] for data in processed_images])
        enhanced_user_prompt = f"{user_prompt}\n\nEXIF Context: {exif_context}\nIf GPS coordinates are provided, use them to determine the precise address for eventLocation.address and eventActivities[].activityLocation.address."

        try:
            request_data = self.create_openai_message(
                enhanced_user_prompt, system_prompt, processed_images, openai_params)
            response = self.client.chat.completions.create(**request_data)
            event_insights = json.loads(response.choices[0].message.content)
            event_insights['_metadata'] = {
                "time_taken": f"{time.time() - start_time:.2f} seconds",
                "tokens": {
                    "input": response.usage.prompt_tokens,
                    "output": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                },
                "model": request_data["model"]
            }
            token_usage["prompt_tokens"] = response.usage.prompt_tokens
            token_usage["completion_tokens"] = response.usage.completion_tokens
            token_usage["total_tokens"] = response.usage.total_tokens

            if openai_params.model == "gpt-4.1":
                cost["input_cost"] = (token_usage["prompt_tokens"] / 1000) * 0.005
                cost["output_cost"] = (token_usage["completion_tokens"] / 1000) * 0.015
                cost["total_cost"] = cost["input_cost"] + cost["output_cost"]

        except httpx.TimeoutException as e:
            logger.error(f"Timeout during collective API call: {str(e)}")
            event_insights = {"error": "OpenAI API call timed out."}
        except httpx.RequestError as e:
            logger.error(f"Network error during collective API call: {str(e)}")
            event_insights = {"error": "Network error occurred."}
        except Exception as e:
            logger.error(f"Failed collective analysis: {e}")
            event_insights = {"error": str(e)}
            for analysis in individual_analyses:
                analysis.analysis = {
                    "imageId": analysis.imageid,
                    "description": "",
                    "objectsDetected": [],
                    "mood": "neutral",
                    "qualityScore": analysis.quality_score
                }

        avg_quality = sum(analysis.quality_score for analysis in individual_analyses) / len(
            individual_analyses) if individual_analyses else 0
        time_taken = time.time() - start_time
        logger.info(
            f"Collective Analysis: {len(images)} images, Time: {time_taken:.2f}s, Token Usage: {token_usage}, Cost: ${cost['total_cost']:.6f}")

        return CollectiveAnalysis(
            event_insights=event_insights,
            image_analyses=individual_analyses,
            total_images=len(images),
            average_quality=round(avg_quality, 2)
        ), {
            "token_usage": token_usage,
            "time_taken": round(time_taken, 2),
            "cost": cost,
            "model": openai_params.model
        }

    def _create_exif_context(self, exif_data_list: List[EXIFData]) -> str:
        context_parts = []
        dates = [exif.datetime for exif in exif_data_list if exif.datetime]
        if dates:
            context_parts.append(f"Dates: {', '.join(set(dates))}")
        locations = [(exif.gps_latitude, exif.gps_longitude) for exif in exif_data_list
                     if exif.gps_latitude is not None and exif.gps_longitude is not None]
        if locations:
            context_parts.append(f"GPS found for {len(locations)} images")
        cameras = [f"{exif.camera_make} {exif.camera_model}".strip()
                   for exif in exif_data_list
                   if exif.camera_make or exif.camera_model]
        if cameras:
            unique_cameras = list(set(cameras))
            context_parts.append(f"Cameras: {', '.join(unique_cameras)}")
        return '\n'.join(context_parts) if context_parts else "No EXIF metadata"

    async def process_request(
            self,
            images: Union[str, List[str], List[ImageInput]],
            user_prompt: str,
            system_prompt: str,
            event_level: bool = True,
            openai_params: OpenAIParams = OpenAIParams(),
            manipulation_params: ImageManipulationParams = ImageManipulationParams()
    ) -> Tuple[Union[CollectiveAnalysis, List[ImageAnalysis]], Dict[str, Any]]:
        start_time = time.time()
        if isinstance(images, (str, list)) and not isinstance(images[0] if images else None, ImageInput):
            image_inputs = self.load_images_from_paths(images)
        else:
            image_inputs = images

        if len(image_inputs) > 10:
            raise ValueError("Maximum of 10 images allowed for input")

        if not image_inputs:
            raise ValueError("No valid images provided")

        if len(image_inputs) == 1:
            openai_params.detail = "high"
            manipulation_params.resize_ratio = RESIZE_RATIO_SINGLE
            manipulation_params.max_width = MAX_WIDTH_SINGLE
            manipulation_params.max_height = MAX_HEIGHT_SINGLE
        else:
            openai_params.detail = "low"
            manipulation_params.resize_ratio = RESIZE_RATIO_MULTIPLE
            manipulation_params.max_width = MAX_WIDTH_MULTIPLE
            manipulation_params.max_height = MAX_HEIGHT_MULTIPLE

        if event_level:
            result, metrics = self.analyze_images_collective(
                image_inputs, user_prompt, system_prompt, openai_params, manipulation_params)
        else:
            result, metrics = await self.analyze_images_individual(
                image_inputs, user_prompt, system_prompt, openai_params, manipulation_params)
        metrics["total_time_taken"] = round(time.time() - start_time, 2)
        logger.info(f"Process Request: {len(image_inputs)} images, Total Time: {metrics['total_time_taken']:.2f}s")
        return result, metrics

def test_exif_extraction(image_path: str):
    try:
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        image_input = ImageInput(image_base64=image_data, imageid="test_image")
        extractor = EnhancedEXIFExtractor()
        image_data_decoded = base64.b64decode(image_input.image_base64)
        exif_data = extractor.extract_exif_data(image_data_decoded)
        print("Enhanced EXIF Data:")
        print(f"DateTime: {exif_data.datetime}")
        print(f"GPS Latitude: {exif_data.gps_latitude}")
        print(f"GPS Longitude: {exif_data.gps_longitude}")
        print(f"GPS Altitude: {exif_data.gps_altitude}")
        print(f"Camera Make: {exif_data.camera_make}")
        print(f"Camera Model: {exif_data.camera_model}")
        print(f"ISO: {exif_data.iso}")
        print(f"Focal Length: {exif_data.focal_length}")
        print(f"Aperture: {exif_data.aperture}")
        print(f"Shutter Speed: {exif_data.shutter_speed}")
        print(f"Flash: {exif_data.flash}")
        # Try reverse geocoding if GPS data is available
        if exif_data.gps_latitude and exif_data.gps_longitude:
            address = extractor.get_address_from_coordinates(
                exif_data.gps_latitude, exif_data.gps_longitude
            )
            print(f"Address: {address}")
    except Exception as e:
        logger.error(f"Error in test_exif_extraction: {str(e)}")
        raise

if __name__ == "__main__":
    EXAMPLE_SYSTEM_PROMPT = """
Generate batch-level metadata for a {{eventType}} event from images. Return concise JSON with:
{
  "eventType": "{{eventType}}",
  "eventMetadata": {
    "eventName": "string",
    "eventSubtype": "string",
    "eventTheme": "string",
    "eventTitle": "string"
  },
  "eventLocation": {
    "name": "string",
    "type": "place" | "venue" | "landmark" | "nature" | "building" | "bridge" | "worship" | "civic" | "market",
    "address": "string"
  },
  "eventActivities": [
    {
      "name": "string",
      "estimatedTime": "string",
      "activityLocation": {
        "name": "string",
        "type": "place" | "venue" | "landmark" | "nature" | "building" | "bridge" | "worship" | "civic" | "market",
        "address": "string"
      }
    }
  ]
}
Rules:
- Focus on batch-level insights.
- Ensure eventTitle is specific.
- Use hh:mm AM/PM for estimatedTime.
- If GPS coordinates are provided, use them to determine the precise address for eventLocation.address and eventActivities[].activityLocation.address.
"""

    analyzer = OpenAIImageAnalyzer(api_key=OPENAI_API_KEY)

    print("=== EXIF EXTRACTION TEST ===")
    try:
        test_exif_extraction("wedding_1.jpg")
    except Exception as e:
        print(f"EXIF extraction test error: {e}")

    print("\n" + "=" * 50 + "\n")
    print("=== EVENT-LEVEL ANALYSIS EXAMPLE ===")
    try:
        system_prompt = EXAMPLE_SYSTEM_PROMPT.replace(
            "{{eventType}}", "wedding"
        ).replace(
            "{{event_subtype_options}}", "ceremony, reception, engagement"
        ).replace(
            "{{event_theme_options}}", "romantic, traditional, modern, cultural"
        )

        event_result, metrics = asyncio.run(analyzer.process_request(
            images=["wedding_1.jpg", "wedding_2.jpg", "wedding_3.jpg"],
            user_prompt="Generate batch-level metadata for a wedding event from these photos. Include event name, subtype, theme, title, location, and activities with times based on EXIF or visual cues. Use provided GPS coordinates to determine the precise address for eventLocation.address and eventActivities[].activityLocation.address.",
            system_prompt=system_prompt,
            event_level=True,
            openai_params=OpenAIParams(max_tokens=600, detail="low"),
            manipulation_params=ImageManipulationParams(resize_ratio=0.25, quality_ratio=0.4)
        ))
        print("Event Analysis Result:")
        print(json.dumps(asdict(event_result), indent=2, default=str))
        print("Metrics:", json.dumps(metrics, indent=2))
    except Exception as e:
        print(f"Event analysis error: {e}")
