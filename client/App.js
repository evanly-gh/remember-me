import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
// Added: Stack navigator to allow nested navigation (Lookup -> EditProfile)
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { StatusBar } from 'expo-status-bar';
import { Ionicons } from '@expo/vector-icons';
import { View, ActivityIndicator } from 'react-native';

import { AuthProvider, useAuth } from './src/context/AuthContext';
import RecordScreen from './src/screens/RecordScreen';
import LookupScreen from './src/screens/LookupScreen';
// Added: New screen for editing profile information
import EditProfileScreen from './src/screens/EditProfileScreen';
import SettingsScreen from './src/screens/SettingsScreen';
import AuthScreen from './src/screens/AuthScreen';

const Tab = createBottomTabNavigator();
// Added: Stack navigator for nested navigation within Lookup tab
const Stack = createNativeStackNavigator();

// Added: Nested stack navigator for Lookup tab screens
// This allows LookupScreen and EditProfileScreen to share navigation context
// while remaining under the Lookup bottom tab
function LookupStack() {
  return (
    <Stack.Navigator
      screenOptions={{
        headerShown: true,
        cardStyle: { backgroundColor: '#fff' },
      }}
    >
      {/* Added: Main Lookup screen that lists all profiles */}
      <Stack.Screen
        name="LookupMain"
        component={LookupScreen}
        options={{ headerShown: false }}
      />
      {/* Added: Edit profile screen for modifying profile details */}
      <Stack.Screen
        name="EditProfile"
        component={EditProfileScreen}
        options={{ headerShown: false }}
      />
    </Stack.Navigator>
  );
}

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
      {/* Added: Use LookupStack instead of direct LookupScreen to enable nested navigation */}
      <Tab.Screen
        name="Lookup"
        component={LookupStack}
        options={{
          headerShown: false,
        }}
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
