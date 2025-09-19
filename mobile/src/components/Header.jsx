import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  Platform,
} from 'react-native';

/**
 * Custom header component
 * Can be used for consistent header styling across screens
 */

export default function Header({ title, subtitle }) {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>{title}</Text>
      {subtitle && <Text style={styles.subtitle}>{subtitle}</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#6366f1',
    paddingVertical: 16,
    paddingHorizontal: 20,
    alignItems: 'center',
    ...Platform.select({
      web: {
        paddingTop: 20,
      },
    }),
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
  },
  subtitle: {
    fontSize: 14,
    color: '#e0e7ff',
    marginTop: 4,
  },
});





