# Expo React Native App

Cross-platform React Native app built with Expo that runs on web, Android, and iOS.

## Quick Start

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start the development server:**
   ```bash
   npx expo start
   ```

3. **Run on different platforms:**
   - Press `w` for web browser
   - Press `a` for Android emulator
   - Press `i` for iOS simulator
   - Scan QR code with Expo Go app on real device

## Project Structure

```
src/
├── screens/           # App screens
│   ├── HomeScreen.jsx     # Main screen with username input
│   ├── ProfileScreen.jsx  # Profile display screen
│   └── NotFoundScreen.jsx # 404 error screen
├── components/        # Reusable components
│   ├── Header.jsx         # Custom header component
│   └── ProfileCard.jsx    # Profile display card
├── navigation/        # Navigation setup
│   └── AppNavigator.jsx   # Main navigation container
├── api/              # API integration
│   └── apiClient.js       # Axios client configuration
└── utils/            # Utilities
    └── env.js             # Environment configuration
```

## Features

- **Cross-platform**: Runs on web, Android, and iOS
- **Navigation**: React Navigation with stack navigator
- **API Integration**: Axios client with error handling
- **Responsive Design**: Works on mobile and web
- **Loading States**: Spinners and error handling
- **Environment Config**: Easy API URL switching

## API Configuration

The app automatically configures API URLs based on platform:

- **Web**: `http://localhost:5000`
- **Android Emulator**: `http://10.0.2.2:5000`
- **iOS Simulator**: `http://localhost:5000`
- **Real Device**: Update `src/utils/env.js` with your local IP

To change the API URL, edit `src/utils/env.js`:

```javascript
// For real device testing
if (Platform.OS === 'android') {
  return 'http://192.168.1.100:5000'; // Your local IP
}
```

## Development Tips

### Web Development
- Use browser developer tools for debugging
- Hot reloading works automatically
- Responsive design adapts to different screen sizes

### Android Development
- Use Android Studio emulator or real device
- For emulator: API calls go to `10.0.2.2:5000`
- For real device: Use your computer's local IP

### iOS Development
- Use iOS Simulator or real device
- Simulator can access `localhost:5000`
- Real device needs your local IP

### Debugging
- Check console logs for API requests/responses
- Use React Native Debugger for advanced debugging
- Expo DevTools provides additional debugging features

## Building for Production

### Web Build
```bash
npx expo build:web
```

### Mobile Builds
```bash
# Android
npx expo build:android

# iOS
npx expo build:ios
```

## Common Issues

### CORS Errors
- Ensure Flask server is running with CORS enabled
- Check that API URL matches server configuration
- Verify network connectivity

### Network Errors
- Check if Flask server is running on correct port
- Verify API URL configuration for your platform
- Test API endpoints with curl or Postman

### Build Errors
- Clear Expo cache: `npx expo start --clear`
- Delete node_modules and reinstall
- Check for conflicting dependencies

## Adding New Features

### New Screens
1. Create screen component in `src/screens/`
2. Add route to `AppNavigator.jsx`
3. Update navigation calls

### New API Endpoints
1. Add methods to `apiClient.js`
2. Update error handling as needed
3. Test with different platforms

### Styling
- Use StyleSheet for consistent styling
- Add responsive design with Platform.select()
- Test on different screen sizes





