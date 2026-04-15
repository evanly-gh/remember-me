import { createContext, useState, useEffect, useContext, useMemo } from 'react';
import { useColorScheme } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

const THEME_KEY = '@hcp:settings:theme';

const lightColors = {
  background: '#FFFFFF',
  card: '#FFFFFF',
  text: '#1F2937',
  textSecondary: '#6B7280',
  textTertiary: '#9CA3AF',
  border: '#F3F4F6',
  inputBg: '#F3F4F6',
  inputBgAlt: '#F9F9F9',
  accent: '#8B5CF6',
  accentLight: '#EDE9FE',
  accentTrack: '#C4B5FD',
  switchTrackOff: '#E5E7EB',
  switchThumbOff: '#F4F3F4',
  destructive: '#EF4444',
  destructiveBg: '#FEF2F2',
  destructiveBorder: '#FEE2E2',
  destructiveCardBg: '#FFFFFF',
  placeholder: '#9CA3AF',
};

const darkColors = {
  background: '#000000',
  card: '#1C1C1E',
  text: '#F9FAFB',
  textSecondary: '#9CA3AF',
  textTertiary: '#6B7280',
  border: '#2C2C2E',
  inputBg: '#2C2C2E',
  inputBgAlt: '#2C2C2E',
  accent: '#8B5CF6',
  accentLight: '#2D2054',
  accentTrack: '#C4B5FD',
  switchTrackOff: '#3A3A3C',
  switchThumbOff: '#F4F3F4',
  destructive: '#EF4444',
  destructiveBg: '#2C1A1A',
  destructiveBorder: '#4A2020',
  destructiveCardBg: '#1C1C1E',
  placeholder: '#6B7280',
};

const ThemeContext = createContext({});

export const ThemeProvider = ({ children }) => {
  const systemScheme = useColorScheme();
  const [themePreference, setThemePreferenceState] = useState('system');
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    AsyncStorage.getItem(THEME_KEY).then((value) => {
      if (value === 'light' || value === 'dark') {
        setThemePreferenceState(value);
      }
      setLoaded(true);
    });
  }, []);

  const setThemePreference = async (pref) => {
    setThemePreferenceState(pref);
    if (pref === 'system') {
      await AsyncStorage.removeItem(THEME_KEY);
    } else {
      await AsyncStorage.setItem(THEME_KEY, pref);
    }
  };

  const isDark = useMemo(() => {
    if (themePreference === 'system') return systemScheme === 'dark';
    return themePreference === 'dark';
  }, [themePreference, systemScheme]);

  const colors = isDark ? darkColors : lightColors;

  const value = useMemo(
    () => ({ colors, isDark, themePreference, setThemePreference, loaded }),
    [colors, isDark, themePreference, loaded],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
};

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};
