import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { StatusBar } from 'expo-status-bar';
import { Ionicons } from '@expo/vector-icons';

import RecordScreen from './src/screens/RecordScreen';
import LookupScreen from './src/screens/LookupScreen';
import SettingsScreen from './src/screens/SettingsScreen';

const Tab = createBottomTabNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <StatusBar style="auto" />
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
    </NavigationContainer>
  );
}
