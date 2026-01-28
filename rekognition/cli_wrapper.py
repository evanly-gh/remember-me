import sys
import json
import base64
from aws_rekognition import RekognitionAnalyzer

def main():
    # Read base64 image from stdin
    try:
        input_data = sys.stdin.read().strip()
        if not input_data:
            print(json.dumps({'available': False, 'error': 'No input data provided'}))
            return
        
        analyzer = RekognitionAnalyzer()
        if not analyzer.is_available:
            print(json.dumps({'available': False, 'error': 'Rekognition analyzer not initialized'}))
            return
            
        result = analyzer.analyze_face(input_data)
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({'available': False, 'error': str(e)}))

if __name__ == "__main__":
    main()
