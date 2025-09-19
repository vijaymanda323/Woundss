import os
from flask import Flask
from flask_cors import CORS
from config import Config
from routes.profile import bp as profile_bp

def create_app():
    """Application factory pattern"""
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(Config)
    
    # Initialize CORS with configuration
    CORS(app, origins=Config.CORS_ORIGINS)
    
    # Register blueprints
    app.register_blueprint(profile_bp, url_prefix='/api')
    
    # Add CORS headers for all routes
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response
    
    return app

# Create app instance
app = create_app()

if __name__ == '__main__':
    # Run the Flask development server
    print(f"Starting Flask server on http://0.0.0.0:{Config.PORT}")
    print(f"API endpoints available at:")
    print(f"  - http://localhost:{Config.PORT}/api/health")
    print(f"  - http://localhost:{Config.PORT}/api/profile/<username>")
    print(f"\nCORS enabled for origins: {Config.CORS_ORIGINS}")
    
    app.run(
        host='0.0.0.0',
        port=Config.PORT,
        debug=Config.FLASK_ENV == 'development'
    )






