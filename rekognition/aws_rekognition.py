import boto3
import cv2
import base64
import json
from typing import Dict, Any, Optional
import os
import sys
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), '..', 'client', '.env')
load_dotenv(dotenv_path=env_path)

class RekognitionAnalyzer:
    
    def __init__(self):
        self.rekognition_client = None
        self.is_available = False
        
        aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        aws_region = os.environ.get('AWS_REGION', 'us-west-2')
        
        if aws_access_key_id and aws_secret_access_key:
            try:
                self.rekognition_client = boto3.client(
                    'rekognition',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=aws_region
                )
                self.is_available = True
                print("AWS Rekognition client initialized successfully", file=sys.stderr)
            except Exception as e:
                print(f"Failed to initialize AWS Rekognition: {e}", file=sys.stderr)
                self.is_available = False
        else:
            print("AWS credentials not provided - Rekognition features disabled", file=sys.stderr)

    def analyze_face(self, base64_image: str) -> Dict[str, Any]:
        if not self.is_available or not self.rekognition_client:
            return {
                'available': False,
                'error': 'AWS Rekognition not available'
            }
        
        try:
            response = self.rekognition_client.detect_faces(
                Image={'Bytes': base64.b64decode(base64_image)},
                Attributes=['ALL']
            )
            
            print(response)

            faces = response['FaceDetails']
            face_summaries = []
            
            for face in faces:
                smile_data = face.get('Smile', {})
                beard_data = face.get('Beard', {})
                emotions = face.get('Emotions', [])
                
                face_summaries.append({
                    'confidence': face['Confidence'],
                    'smiling': smile_data.get('Value', False),
                    'smile_confidence': smile_data.get('Confidence', 0),
                    'has_beard': beard_data.get('Value', False),
                    'primary_emotion': emotions[0].get('Type') if emotions else "Unknown",
                    'bounding_box': face['BoundingBox']
                })
            
            return {
                'available': True,
                'face_count': len(faces),
                'faces': face_summaries
            }
            
        except Exception as e:
            return {
                'available': False,
                'error': str(e)
            }
    
    
    