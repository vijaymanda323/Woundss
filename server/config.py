import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for Flask app"""
    
    # Server configuration
    PORT = int(os.environ.get('PORT', 5000))
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key')
    FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
    
    # CORS configuration
    CORS_ORIGINS = [
        'http://localhost:3000',  # Expo web
        'http://localhost:19006', # Expo web alternative
        'http://10.0.2.2:3000',   # Android emulator
        'http://10.0.2.2:19006',  # Android emulator alternative
    ]
    
    # Add your local IP for real device testing
    # CORS_ORIGINS.append('http://YOUR_LOCAL_IP:3000')
    
    @staticmethod
    def init_app(app):
        """Initialize app with configuration"""
        pass






