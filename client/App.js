import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { StatusBar } from 'expo-status-bar';
import { Ionicons } from '@expo/vector-icons';
import { View, ActivityIndicator } from 'react-native';

import { AuthProvider, useAuth } from './src/context/AuthContext';
import RecordScreen from './src/screens/RecordScreen';
import LookupScreen from './src/screens/LookupScreen';
import SettingsScreen from './src/screens/SettingsScreen';
import AuthScreen from './src/screens/AuthScreen';

const Tab = createBottomTabNavigator();

function MainNavigator() {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
        <ActivityIndicator size="large" color="#007AFF" />
      </View>
    );
  }

  if (!user) {
    return <AuthScreen />;
  }

  return (
    <Tab.Navigator
      initialRouteName="Lookup"
      screenOptions={({ route }) => ({
        tabBarActiveTintColor: '#007AFF',
        tabBarInactiveTintColor: '#8E8E93',
        tabBarIcon: ({ focused, color, size }) => {
          let iconName;

          if (route.name === 'Record') {
            iconName = focused ? 'camera' : 'camera-outline';
          } else if (route.name === 'Lookup') {
            iconName = focused ? 'search' : 'search-outline';
          } else if (route.name === 'Settings') {
            iconName = focused ? 'settings' : 'settings-outline';
          }

          return <Ionicons name={iconName} size={size} color={color} />;
        },
      })}
    >
      <Tab.Screen
        name="Record"
        component={RecordScreen}
        options={{
          headerShown: false,
        }}
      />
      <Tab.Screen
        name="Lookup"
        component={LookupScreen}
      />
      <Tab.Screen
        name="Settings"
        component={SettingsScreen}
      />
    </Tab.Navigator>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <NavigationContainer>
        <StatusBar style="auto" />
        <MainNavigator />
      </NavigationContainer>
    </AuthProvider>
  );
}
