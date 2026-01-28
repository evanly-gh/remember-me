import React, { useState, useEffect } from 'react';
import { SafeAreaView, View, TextInput, StyleSheet, Text, ScrollView, TouchableOpacity, Image, ActivityIndicator, Platform, StatusBar } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
// Added: Supabase client for querying profiles from database
import { supabase } from '../lib/supabase';
// Added: Authentication context to get current user ID
import { useAuth } from '../context/AuthContext';
// Added: allow reloading when screen gains focus
import { useFocusEffect } from '@react-navigation/native';

export default function LookupScreen({ navigation }) {
  // Added: State for search query to filter profiles by name
  const [searchQuery, setSearchQuery] = useState('');
  // Added: State to store all unique profiles fetched from Supabase
  const [profiles, setProfiles] = useState([]);
  // Added: Loading state while fetching data from database
  const [loading, setLoading] = useState(true);
  // Added: Get current user ID from auth context
  const { user } = useAuth();

  // Added: Load profiles on component mount and when user changes
  useEffect(() => {
    if (user) {
      loadProfiles();
    }
  }, [user]);

  // Added: Also reload when the screen becomes focused (ensures Lookup reflects edits)
  useFocusEffect(
    React.useCallback(() => {
      if (user) loadProfiles();
    }, [user])
  );

  // Added: Function to fetch all records from Supabase and group by unique name
  const loadProfiles = async () => {
    if (!user) return;

    try {
      setLoading(true);
      // Added: Query all records for current user, ordered by date (most recent first)
      const { data: allRecords, error } = await supabase
        .from('people')
        .select('*')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false });

      if (error) throw error;

      // Added: Group by name and keep only the most recent photo for each name
      // This creates a unique profile per person showing their latest photo
      const uniqueProfiles = {};
      allRecords.forEach(record => {
        if (!uniqueProfiles[record.name] || 
            new Date(record.created_at) > new Date(uniqueProfiles[record.name].created_at)) {
          uniqueProfiles[record.name] = record;
        }
      });

      // Added: Convert grouped object to array for mapping
      const profilesArray = Object.values(uniqueProfiles);
      setProfiles(profilesArray);
    } catch (error) {
      console.error('Error loading profiles:', error);
    } finally {
      setLoading(false);
    }
  };

  // Added: Filter profiles based on search query (searches by name)
  const filteredProfiles = profiles.filter(profile =>
    profile.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Added: Parse facial_details JSON and extract gender and age_range for display
  const getFacialDetails = (facialDetails) => {
    if (!facialDetails) return { gender: '', ageRange: '' };
    
    try {
      // Added: Handle both string (raw JSON) and object formats
      const details = typeof facialDetails === 'string' 
        ? JSON.parse(facialDetails) 
        : facialDetails;
      return {
        gender: details.gender || '',
        ageRange: details.age_range || '',
      };
    } catch (error) {
      return { gender: '', ageRange: '' };
    }
  };

  // Added: Show loading spinner while fetching profiles from Supabase
  if (loading) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <View style={styles.container}>
          <ActivityIndicator size="large" color="#007AFF" />
        </View>
      </SafeAreaView>
    );
  }

  // Added: Main render - wrap in SafeAreaView so search bar is reachable on all devices
  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.container}>
      {/* Added: Search bar for filtering profiles by name */}
      <View style={styles.searchContainer}>
        <TextInput
          style={styles.searchBar}
          placeholder="Search..."
          value={searchQuery}
          onChangeText={setSearchQuery}
        />
      </View>
      {/* Added: Scrollable list of profiles with filtering */}
      <ScrollView style={styles.content}>
        {filteredProfiles.length === 0 ? (
          // Added: Empty state when no profiles match search or no profiles exist
          <View style={styles.emptyState}>
            <Text style={styles.emptyText}>No profiles yet. Start recording!</Text>
          </View>
        ) : (
          // Added: Map through filtered profiles and display as cards
          filteredProfiles.map((profile) => {
            // Added: Extract facial details for display
            const details = getFacialDetails(profile.facial_details);
            // Added: Format facial details as readable text (e.g., "Male • 25-30")
            const description = [details.gender, details.ageRange].filter(Boolean).join(' • ');

            return (
              // Added: Touchable card that navigates to EditProfileScreen
              <TouchableOpacity 
                key={profile.id} 
                style={styles.card}
                onPress={() => {
                  navigation.navigate('EditProfile', { profileName: profile.name });
                }}
              >
                {/* Added: Profile photo from Supabase Storage */}
                <Image
                  source={{ uri: profile.photo_url }}
                  style={styles.cardImage}
                />
                {/* Added: Profile information section */}
                <View style={styles.cardContent}>
                  <Text style={styles.cardName}>{profile.name}</Text>
                  {/* Added: Show gender and age from facial_details JSON */}
                  {description && (
                    <Text style={styles.cardDescription}>{description}</Text>
                  )}
                  {/* Added: Show event if available */}
                  {profile.event && (
                    <Text style={styles.cardSubtext}>Event: {profile.event}</Text>
                  )}
                  {/* Added: Show location if available */}
                  {profile.location && (
                    <Text style={styles.cardSubtext}>Location: {profile.location}</Text>
                  )}
                </View>
              </TouchableOpacity>
            );
          })
        )}
      </ScrollView>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  // Safe area wrapper to ensure search bar isn't obscured by status bar
  safeArea: {
    flex: 1,
    backgroundColor: '#fff',
    paddingTop: Platform.OS === 'android' ? StatusBar.currentHeight || 0 : 0,
  },
  searchContainer: {
    padding: 10,
    backgroundColor: '#f5f5f5',
    // ensure the input is tappable and visible
    paddingTop: 6,
  },
  searchBar: {
    height: 40,
    backgroundColor: '#fff',
    borderRadius: 8,
    paddingHorizontal: 15,
    fontSize: 16,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  content: {
    flex: 1,
  },
  card: {
    flexDirection: 'row',
    padding: 20,
    marginHorizontal: 10,
    marginTop: 10,
    minHeight: 100,
    backgroundColor: '#fff',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e0e0e0',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  cardImage: {
    width: 70,
    height: 70,
    borderRadius: 35,
    marginRight: 15,
    backgroundColor: '#f0f0f0',
  },
  cardContent: {
    flex: 1,
    justifyContent: 'center',
  },
  cardName: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 5,
    color: '#333',
  },
  cardDescription: {
    fontSize: 14,
    color: '#666',
    marginBottom: 5,
  },
  cardSubtext: {
    fontSize: 12,
    color: '#999',
    marginBottom: 2,
  },
  emptyState: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    minHeight: 400,
  },
  emptyText: {
    fontSize: 16,
    color: '#999',
    textAlign: 'center',
  },
});
