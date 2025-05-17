import os
import yaml
import random
import logging
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass
from google.cloud import aiplatform
from vertexai.language_models import TextGenerationModel
from vertexai.vision_models import ImageGenerationModel
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from dotenv import load_dotenv
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_image_generator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ImageMetadata:
    title: str
    date: str
    image_path: str
    thumbnail_path: str
    google_drive_url: str
    thumbnail_url: str
    description: str
    ai_metadata: Dict

class AIImageGenerator:
    def __init__(self):
        try:
            load_dotenv()
            self._load_constants()
            self._init_services()
            self.THUMBNAIL_SIZE = (250, 250)
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def _load_constants(self):
        """상수 설정 로드"""
        try:
            constants_path = os.path.join(os.path.dirname(__file__), '../_data/constants.yaml')
            with open(constants_path, 'r', encoding='utf-8') as f:
                self.constants = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load constants: {str(e)}")
            raise

    def _init_services(self):
        """서비스 초기화"""
        try:
            self.drive_service = self._init_drive_service()
            self.text_model = TextGenerationModel.from_pretrained(
                model_name="text-bison@001"
            )
            self.image_model = ImageGenerationModel.from_pretrained(
                model_name="imagegeneration@002"
            )
        except Exception as e:
            logger.error(f"Service initialization failed: {str(e)}")
            raise

    def _init_drive_service(self):
        """Google Drive API 서비스 초기화 with 재시도 로직"""
        SCOPES = ['https://www.googleapis.com/auth/drive.file']
        MAX_RETRIES = 3
        
        for attempt in range(MAX_RETRIES):
            try:
                creds = self._get_credentials(SCOPES)
                return build('drive', 'v3', credentials=creds)
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Failed to initialize Drive service: {str(e)}")
                    raise
                logger.warning(f"Retry {attempt + 1}/{MAX_RETRIES}")

    def _get_credentials(self, SCOPES):
        """인증 정보 획득"""
        creds = None
        token_path = 'token.json'
        
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
            
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
                
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
                
        return creds

    def get_random_concepts(self):
        """컨셉과 스타일 무작위 선택"""
        num_concepts = random.randint(1, 3)
        concepts = random.sample(self.constants['IMAGE_CONCEPT'], num_concepts)
        style = random.choice(self.constants['IMAGE_STYLE'])
        return concepts, style

    def get_image_keywords(self, concepts: List[str]) -> List[str]:
        """LLM을 사용하여 키워드 확장 with 검증"""
        try:
            prompt = f"""
            For the following concepts: {', '.join(concepts)}
            Please provide 5-10 related English keywords.
            Return only comma-separated single words, no phrases.
            """
            response = self.text_model.predict(prompt)
            keywords = [word.strip() for word in response.text.split(',')]
            
            # Validate keywords
            if not keywords or len(keywords) < 5:
                raise ValueError("Insufficient keywords generated")
                
            return keywords[:10]  # 최대 10개로 제한
        except Exception as e:
            logger.error(f"Keyword generation failed: {str(e)}")
            return self._get_fallback_keywords(concepts)

    def _get_fallback_keywords(self, concepts: List[str]) -> List[str]:
        """키워드 생성 실패시 폴백 옵션"""
        return concepts + ["art", "creative", "digital", "beautiful", "original"]

    def generate_image_prompt(self, keywords, style):
        """이미지 생성을 위한 프롬프트 생성"""
        prompt = f"""
        Create a detailed image generation prompt that combines these elements:
        Keywords: {', '.join(keywords)}
        Style: {style}
        The prompt should be specific and detailed, suitable for AI image generation.
        Return only the prompt text without any explanations.
        """
        return self.text_model.predict(prompt).text.strip()

    def generate_image(self, prompt: str) -> Image:
        """AI 이미지 생성 with 재시도 로직"""
        MAX_RETRIES = 3
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self.image_model.generate_images(
                    prompt=prompt,
                    number_of_images=1,
                    resolution="1024x1024"
                )
                return response[0]
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Image generation failed: {str(e)}")
                    raise
                logger.warning(f"Retry {attempt + 1}/{MAX_RETRIES}")

    def create_filename(self, concepts, style):
        """파일 이름 생성"""
        date = datetime.now().strftime("%Y-%m-%d")
        concept_text = '-'.join(concepts).lower().replace(' ', '-')
        style_text = style.lower().replace(' ', '-')
        return f"{date}-{concept_text}-{style_text}.png"

    def create_thumbnail(self, image):
        """원본 이미지로부터 썸네일 생성"""
        thumbnail = image.copy()
        thumbnail.thumbnail(self.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
        
        # 정사각형 썸네일을 위한 크기 조정
        if thumbnail.size != self.THUMBNAIL_SIZE:
            background = Image.new('RGB', self.THUMBNAIL_SIZE, (255, 255, 255))
            offset = ((self.THUMBNAIL_SIZE[0] - thumbnail.size[0]) // 2,
                     (self.THUMBNAIL_SIZE[1] - thumbnail.size[1]) // 2)
            background.paste(thumbnail, offset)
            thumbnail = background
            
        return thumbnail

    def upload_to_drive(self, image_path, filename):
        """Google Drive에 이미지 업로드"""
        # 원본 이미지 업로드
        file_metadata = {
            'name': filename,
            'parents': [os.getenv('GOOGLE_DRIVE_FOLDER_ID')]
        }
        media = MediaFileUpload(image_path, mimetype='image/png')
        file = self.drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id,webContentLink'
        ).execute()

        # 썸네일 이미지 업로드
        thumbnail_filename = filename.replace('.png', '.thumbnail.png')
        thumbnail_path = image_path.replace('.png', '.thumbnail.png')
        thumbnail_metadata = {
            'name': thumbnail_filename,
            'parents': [os.getenv('GOOGLE_DRIVE_FOLDER_ID')]
        }
        thumbnail_media = MediaFileUpload(thumbnail_path, mimetype='image/png')
        thumbnail_file = self.drive_service.files().create(
            body=thumbnail_metadata,
            media_body=thumbnail_media,
            fields='id,webContentLink'
        ).execute()

        return {
            'original': file,
            'thumbnail': thumbnail_file
        }

    def update_image_data(self, image_data):
        """이미지 데이터를 YAML 파일에 추가하고 Jekyll collection 파일 생성"""
        # 1. Update images.yml
        yaml_path = '../_data/images.yml'
        existing_data = {}
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r', encoding='utf-8') as f:
                existing_data = yaml.safe_load(f) or {}
        
        existing_data.update(image_data)
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(existing_data, f, allow_unicode=True, sort_keys=False)
        
        # 2. Create Jekyll collection file
        key = list(image_data.keys())[0]
        data = image_data[key]
        
        content = f"""---
layout: image
title: "{data['title']}"
date: {data['date']}
image_path: {data['image_path']}
thumbnail_path: {data['thumbnail_path']}
description: "{data['description']}"
image_id: "{key}"
ai_metadata: {data['ai_metadata']}
---
"""
        
        # Ensure _images directory exists
        os.makedirs('../_images', exist_ok=True)
        
        # Create markdown file
        with open(f'../_images/{key}.md', 'w', encoding='utf-8') as f:
            f.write(content)

    def _validate_image_data(self, image_data: Dict) -> bool:
        """이미지 메타데이터 검증"""
        required_fields = ['title', 'date', 'image_path', 'thumbnail_path', 
                         'google_drive_url', 'description', 'ai_metadata']
        
        try:
            for key, data in image_data.items():
                if not all(field in data for field in required_fields):
                    return False
                if not isinstance(data['ai_metadata'], dict):
                    return False
            return True
        except Exception:
            return False

    def generate(self) -> Dict:
        """전체 이미지 생성 프로세스 실행"""
        try:
            concepts, style = self.get_random_concepts()
            logger.info(f"Selected concepts: {concepts}, style: {style}")

            keywords = self.get_image_keywords(concepts)
            logger.info(f"Generated keywords: {keywords}")

            prompt = self.generate_image_prompt(keywords, style)
            logger.info(f"Generated prompt: {prompt}")

            image = self.generate_image(prompt)
            filename = self.create_filename(concepts, style)
            
            # 임시 파일 처리를 with 문으로 안전하게 관리
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
                image.save(temp_path)
                
                thumbnail = self.create_thumbnail(image)
                temp_thumbnail_path = temp_path.replace('.png', '.thumbnail.png')
                thumbnail.save(temp_thumbnail_path)
                
                drive_files = self.upload_to_drive(temp_path, filename)
            
            # 임시 파일 정리
            os.unlink(temp_path)
            os.unlink(temp_thumbnail_path)
            
            image_data = self._create_image_metadata(filename, concepts, style, 
                                                   keywords, prompt, drive_files)
            
            if not self._validate_image_data(image_data):
                raise ValueError("Invalid image metadata")
                
            self.update_image_data(image_data)
            logger.info("Image generation process completed successfully")
            
            return image_data
            
        except Exception as e:
            logger.error(f"Image generation process failed: {str(e)}")
            raise

if __name__ == '__main__':
    generator = AIImageGenerator()
    result = generator.generate()
    print("Image generated and data saved successfully!")
    print(f"Image data: {result}")
