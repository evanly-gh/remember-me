import { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Alert, Switch } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useAuth } from '../context/AuthContext';

export default function SettingsScreen() {
  const { user, signOut } = useAuth();
  const [showFacialDetails, setShowFacialDetails] = useState(false);

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      const value = await AsyncStorage.getItem('@hcp:settings:showFacialDetails');
      if (value !== null) {
        setShowFacialDetails(value === 'true');
      }
    } catch (error) {
      console.error('Error loading settings:', error);
    }
  };

  const toggleFacialDetails = async (value) => {
    try {
      setShowFacialDetails(value);
      await AsyncStorage.setItem('@hcp:settings:showFacialDetails', value.toString());
    } catch (error) {
      console.error('Error saving settings:', error);
      Alert.alert('Error', 'Failed to save setting');
    }
  };

  const handleSignOut = async () => {
    try {
      await signOut();
    } catch (error) {
      Alert.alert('Error', error.message);
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.section}>
        <Text style={styles.label}>Email</Text>
        <Text style={styles.value}>{user?.email}</Text>
      </View>

      <View style={styles.section}>
        <View style={styles.settingRow}>
          <View style={styles.settingText}>
            <Text style={styles.settingLabel}>Show Facial Details</Text>
            <Text style={styles.settingDescription}>Display facial analysis data in profile views</Text>
          </View>
          <Switch
            value={showFacialDetails}
            onValueChange={toggleFacialDetails}
            trackColor={{ false: '#E5E7EB', true: '#C4B5FD' }}
            thumbColor={showFacialDetails ? '#8B5CF6' : '#f4f3f4'}
          />
        </View>
      </View>

      <TouchableOpacity style={styles.signOutButton} onPress={handleSignOut}>
        <Text style={styles.signOutText}>Sign Out</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    padding: 20,
  },
  section: {
    marginBottom: 30,
  },
  settingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 10,
  },
  settingText: {
    flex: 1,
    marginRight: 15,
  },
  settingLabel: {
    fontSize: 16,
    color: '#1F2937',
    fontWeight: '600',
    marginBottom: 4,
  },
  settingDescription: {
    fontSize: 13,
    color: '#6B7280',
  },
  label: {
    fontSize: 14,
    color: '#6B7280',
    marginBottom: 5,
  },
  value: {
    fontSize: 16,
    color: '#1F2937',
  },
  signOutButton: {
    borderWidth: 1,
    borderColor: '#FEE2E2',
    backgroundColor: '#FEF2F2',
    padding: 15,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 20,
  },
  signOutText: {
    color: '#EF4444',
    fontSize: 16,
    fontWeight: '600',
  },
});
