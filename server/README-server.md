# Flask API Server

Simple Flask REST API with CORS support for the Expo React Native app.

## Quick Start

1. **Set up virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   cp env.example .env
   # Edit .env if needed (default values work for development)
   ```

4. **Run the server:**
   ```bash
   python app.py
   ```

The server will start on `http://localhost:5000`

## API Endpoints

### Health Check
```bash
curl http://localhost:5000/api/health
```
Response:
```json
{
  "status": "ok",
  "message": "Flask API is running"
}
```

### Get Profile
```bash
curl http://localhost:5000/api/profile/testuser
```
Response (real profile):
```json
{
  "username": "testuser",
  "name": "Alex Johnson",
  "bio": "Software developer passionate about clean code and user experience.",
  "isReal": true,
  "avatar": "https://ui-avatars.com/api/?name=Alex+Johnson&background=random"
}
```

Response (fake profile - usernames with digits):
```json
{
  "username": "user123",
  "name": null,
  "bio": null,
  "isReal": false,
  "message": "Profile not found"
}
```

## Configuration

Edit `.env` file to configure:

- `PORT`: Server port (default: 5000)
- `SECRET_KEY`: Flask secret key
- `FLASK_ENV`: Environment (development/production)

## CORS Configuration

The server is configured to accept requests from:
- `http://localhost:3000` (Expo web)
- `http://localhost:19006` (Expo web alternative)
- `http://10.0.2.2:3000` (Android emulator)
- `http://10.0.2.2:19006` (Android emulator alternative)

To add your local IP for real device testing, edit `config.py` and add:
```python
CORS_ORIGINS.append('http://YOUR_LOCAL_IP:3000')
```

## Integrating Real External APIs

To replace the simulated profile data with real API calls:

1. **Add API credentials to `.env`:**
   ```
   EXTERNAL_API_KEY=your-api-key
   EXTERNAL_API_URL=https://api.example.com
   ```

2. **Modify `routes/profile.py`:**
   ```python
   import requests
   import os
   
   @bp.route('/profile/<username>')
   def get_profile(username):
       api_key = os.environ.get('EXTERNAL_API_KEY')
       api_url = os.environ.get('EXTERNAL_API_URL')
       
       try:
           response = requests.get(
               f"{api_url}/users/{username}",
               headers={'Authorization': f'Bearer {api_key}'}
           )
           response.raise_for_status()
           return jsonify(response.json())
       except requests.RequestException as e:
           return jsonify({'error': str(e)}), 500
   ```

3. **Add error handling and caching as needed**

## Development Tips

- The server runs with debug mode enabled in development
- Hot reloading is enabled for code changes
- Check console output for request/response logging
- Use `curl` or Postman to test API endpoints
- Monitor CORS errors in browser developer tools






