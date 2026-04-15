import { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Alert, Switch } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useAuth } from '../context/AuthContext';
import { useTheme } from '../context/ThemeContext';

const THEME_OPTIONS = [
  { value: 'system', label: 'System' },
  { value: 'light', label: 'Light' },
  { value: 'dark', label: 'Dark' },
];

export default function SettingsScreen() {
  const { user, signOut } = useAuth();
  const { colors, themePreference, setThemePreference } = useTheme();
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
    <View style={[styles.container, { backgroundColor: colors.background }]}>
      <View style={styles.section}>
        <Text style={[styles.label, { color: colors.textSecondary }]}>Email</Text>
        <Text style={[styles.value, { color: colors.text }]}>{user?.email}</Text>
      </View>

      <View style={styles.section}>
        <Text style={[styles.settingLabel, { color: colors.text }]}>Appearance</Text>
        <View style={[styles.segmentedControl, { backgroundColor: colors.inputBg }]}>
          {THEME_OPTIONS.map((option) => {
            const isActive = themePreference === option.value;
            return (
              <TouchableOpacity
                key={option.value}
                style={[
                  styles.segmentOption,
                  isActive && [styles.segmentOptionActive, { backgroundColor: colors.card }],
                ]}
                onPress={() => setThemePreference(option.value)}
              >
                <Text
                  style={[
                    styles.segmentText,
                    { color: colors.textSecondary },
                    isActive && { color: colors.text, fontWeight: '600' },
                  ]}
                >
                  {option.label}
                </Text>
              </TouchableOpacity>
            );
          })}
        </View>
      </View>

      <View style={styles.section}>
        <View style={styles.settingRow}>
          <View style={styles.settingText}>
            <Text style={[styles.settingLabel, { color: colors.text }]}>Show Facial Details</Text>
            <Text style={[styles.settingDescription, { color: colors.textSecondary }]}>Display facial analysis data in profile views</Text>
          </View>
          <Switch
            value={showFacialDetails}
            onValueChange={toggleFacialDetails}
            trackColor={{ false: colors.switchTrackOff, true: colors.accentTrack }}
            thumbColor={showFacialDetails ? colors.accent : colors.switchThumbOff}
          />
        </View>
      </View>

      <TouchableOpacity
        style={[styles.signOutButton, { borderColor: colors.destructiveBorder, backgroundColor: colors.destructiveBg }]}
        onPress={handleSignOut}
      >
        <Text style={styles.signOutText}>Sign Out</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
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
    fontWeight: '600',
    marginBottom: 4,
  },
  settingDescription: {
    fontSize: 13,
  },
  label: {
    fontSize: 14,
    marginBottom: 5,
  },
  value: {
    fontSize: 16,
  },

  // Segmented control
  segmentedControl: {
    flexDirection: 'row',
    borderRadius: 10,
    padding: 3,
    marginTop: 10,
  },
  segmentOption: {
    flex: 1,
    paddingVertical: 8,
    alignItems: 'center',
    borderRadius: 8,
  },
  segmentOptionActive: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  segmentText: {
    fontSize: 14,
  },

  // Sign out
  signOutButton: {
    borderWidth: 1,
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
