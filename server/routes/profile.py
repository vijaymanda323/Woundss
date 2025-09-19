from flask import Blueprint, jsonify
import random

# Create blueprint for profile routes
bp = Blueprint('profile', __name__)

@bp.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'Flask API is running'
    })

@bp.route('/profile/<username>')
def get_profile(username):
    """
    Get user profile by username
    
    Simulated profile data for demonstration:
    - If username contains digits -> fake profile (isReal: false)
    - Otherwise -> real profile with generated data
    
    To integrate with real API later:
    1. Replace this function with actual API calls
    2. Add API keys to .env file
    3. Use requests library to call external services
    4. Handle authentication and error cases
    """
    
    # Simulate real/fake profile logic
    has_digits = any(char.isdigit() for char in username)
    
    if has_digits:
        # Return fake profile
        return jsonify({
            'username': username,
            'name': None,
            'bio': None,
            'isReal': False,
            'message': 'Profile not found'
        })
    else:
        # Generate realistic profile data
        names = [
            'Alex Johnson', 'Sarah Chen', 'Michael Rodriguez', 
            'Emma Wilson', 'David Kim', 'Lisa Thompson',
            'James Brown', 'Maria Garcia', 'John Smith', 'Anna Davis'
        ]
        
        bios = [
            'Software developer passionate about clean code and user experience.',
            'Designer who loves creating beautiful and functional interfaces.',
            'Product manager focused on building products that matter.',
            'Full-stack developer with expertise in React and Python.',
            'UX researcher dedicated to understanding user needs.',
            'Mobile app developer specializing in React Native.',
            'Data scientist exploring the intersection of AI and human behavior.',
            'DevOps engineer ensuring reliable and scalable systems.',
            'Frontend developer crafting responsive and accessible web experiences.',
            'Backend developer building robust APIs and microservices.'
        ]
        
        selected_name = random.choice(names)
        selected_bio = random.choice(bios)
        
        return jsonify({
            'username': username,
            'name': selected_name,
            'bio': selected_bio,
            'isReal': True,
            'avatar': f'https://ui-avatars.com/api/?name={selected_name.replace(" ", "+")}&background=random'
        })






