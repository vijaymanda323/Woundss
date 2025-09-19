import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  Image,
  Platform,
} from 'react-native';

/**
 * Profile card component for displaying user information
 * 
 * Features:
 * - User avatar (using UI Avatars service)
 * - Name, username, and bio display
 * - Responsive design for web and mobile
 * - Clean card layout with shadows
 */

export default function ProfileCard({ profile }) {
  return (
    <View style={styles.card}>
      <View style={styles.avatarContainer}>
        <Image
          source={{ uri: profile.avatar }}
          style={styles.avatar}
          defaultSource={require('../../assets/placeholder-avatar.png')}
        />
      </View>
      
      <View style={styles.infoContainer}>
        <Text style={styles.name}>{profile.name}</Text>
        <Text style={styles.username}>@{profile.username}</Text>
        
        {profile.bio && (
          <View style={styles.bioContainer}>
            <Text style={styles.bio}>{profile.bio}</Text>
          </View>
        )}
        
        <View style={styles.statusContainer}>
          <View style={styles.statusBadge}>
            <Text style={styles.statusText}>Verified Profile</Text>
          </View>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 24,
    marginBottom: 20,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.1,
    shadowRadius: 6,
    elevation: 8,
    ...Platform.select({
      web: {
        maxWidth: 400,
        marginHorizontal: 'auto',
      },
    }),
  },
  avatarContainer: {
    marginBottom: 20,
  },
  avatar: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#f3f4f6',
  },
  infoContainer: {
    alignItems: 'center',
    width: '100%',
  },
  name: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 4,
    textAlign: 'center',
  },
  username: {
    fontSize: 16,
    color: '#6b7280',
    marginBottom: 16,
    textAlign: 'center',
  },
  bioContainer: {
    width: '100%',
    marginBottom: 20,
  },
  bio: {
    fontSize: 16,
    color: '#374151',
    lineHeight: 24,
    textAlign: 'center',
  },
  statusContainer: {
    alignItems: 'center',
  },
  statusBadge: {
    backgroundColor: '#10b981',
    borderRadius: 20,
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  statusText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
});





