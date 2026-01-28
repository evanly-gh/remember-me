import React, { useState, useEffect } from 'react';
import {
  SafeAreaView,
  View,
  TextInput,
  StyleSheet,
  Text,
  ScrollView,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  Alert,
  Platform,
  StatusBar,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
// Added: Supabase client for querying and updating profile records
import { supabase } from '../lib/supabase';
// Added: Authentication context to get current user ID
import { useAuth } from '../context/AuthContext';

// Added: Screen to edit an existing profile with photo selection and metadata editing
export default function EditProfileScreen({ route, navigation }) {
  // Added: Extract profile name from navigation route params
  const { profileName } = route.params;
  // Added: Store all records for this profile (all photos with same name)
  const [records, setRecords] = useState([]);
  // Added: Loading state while fetching records from database
  const [loading, setLoading] = useState(true);
  // Added: Loading state while saving updated profile
  const [saving, setSaving] = useState(false);
  // Added: Currently selected record for editing and display
  const [selectedPhoto, setSelectedPhoto] = useState(null);
  // Added: Get current user ID from auth context
  const { user } = useAuth();

  // Added: Form state for editable fields
  const [name, setName] = useState(profileName);
  const [event, setEvent] = useState('');
  const [location, setLocation] = useState('');
  const [date, setDate] = useState('');

  // Added: Load all records for this profile when component mounts
  useEffect(() => {
    if (user) {
      loadRecords();
    }
  }, [user, profileName]);

  // Added: Fetch all records from Supabase with matching profile name
  const loadRecords = async () => {
    try {
      setLoading(true);
      // Added: Query all records for current user with matching name
      const { data, error } = await supabase
        .from('people')
        .select('*')
        .eq('user_id', user.id)
        .eq('name', profileName)
        .order('created_at', { ascending: false });

      if (error) throw error;

      setRecords(data);
      
      // Added: Initialize form with most recent record's data
      if (data && data.length > 0) {
        const mostRecent = data[0];
        setSelectedPhoto(mostRecent);
        setEvent(mostRecent.event || '');
        setLocation(mostRecent.location || '');
        setDate(mostRecent.date || '');
      }
    } catch (error) {
      console.error('Error loading records:', error);
      Alert.alert('Error', 'Failed to load profile records');
    } finally {
      setLoading(false);
    }
  };

  // Added: Save updated profile information to selected record
  const handleSave = async () => {
    if (!selectedPhoto) {
      Alert.alert('Error', 'Please select a photo');
      return;
    }

    try {
      setSaving(true);
      // Added: Update the selected record with new field values
      const { error } = await supabase
        .from('people')
        .update({
          name: name.trim(),
          event: event.trim() || null,
          location: location.trim() || null,
          date: date || null,
        })
        .eq('id', selectedPhoto.id);

      if (error) throw error;

      // Added: After saving, navigate back to Lookup; LookupScreen uses focus listener to reload
      Alert.alert('Success', 'Profile updated successfully');
      navigation.goBack();
    } catch (error) {
      console.error('Error saving profile:', error);
      Alert.alert('Error', 'Failed to save profile');
    } finally {
      setSaving(false);
    }
  };

  // Added: Delete a single photo record from profile (ask for confirmation first)
  const handleDelete = async (recordId) => {
    Alert.alert(
      'Delete Record',
      'Are you sure you want to delete this record?',
      [
        { text: 'Cancel', onPress: () => {}, style: 'cancel' },
        {
          text: 'Delete',
          onPress: async () => {
            try {
              // Added: Delete the specific record from database
              const { error } = await supabase
                .from('people')
                .delete()
                .eq('id', recordId);

              if (error) throw error;

              Alert.alert('Success', 'Record deleted');
              // Added: Reload records to update display
              loadRecords();
            } catch (error) {
              console.error('Error deleting record:', error);
              Alert.alert('Error', 'Failed to delete record');
            }
          },
          style: 'destructive',
        },
      ]
    );
  };

  // Added: Show loading spinner while fetching profile records
  if (loading) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#007AFF" />
      </View>
    );
  }

  // Added: Extract recent history from all records for display
  const recentDates = records
    .slice(0, 3)
    .map(r => r.date)
    .filter(Boolean);

  const recentLocations = records
    .slice(0, 3)
    .map(r => r.location)
    .filter(Boolean);

  const recentEvents = records
    .slice(0, 3)
    .map(r => r.event)
    .filter(Boolean);

  // Added: Main render with header, photo display, edit form, and recent history
  // Wrapped in SafeAreaView so header buttons are reachable and not overlapped
  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView style={styles.container}>
      {/* Added: Header with back button, title, and save button */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Ionicons name="chevron-back" size={28} color="#007AFF" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Edit Profile</Text>
        <TouchableOpacity 
          onPress={handleSave}
          disabled={saving}
        >
          {saving ? (
            <ActivityIndicator size="small" color="#007AFF" />
          ) : (
            <Text style={styles.saveButton}>Save</Text>
          )}
        </TouchableOpacity>
      </View>

      {/* Added: Display main profile photo with date */}
      {selectedPhoto && (
        <View style={styles.photoSection}>
          <Image
            source={{ uri: selectedPhoto.photo_url }}
            style={styles.mainPhoto}
          />
          <Text style={styles.photoDate}>
            {selectedPhoto.created_at 
              ? new Date(selectedPhoto.created_at).toLocaleDateString() 
              : 'No date'}
          </Text>
        </View>
      )}

      {/* Added: Horizontal scrollable gallery of all photos for this profile */}
      {records.length > 1 && (
        <View style={styles.gallerySection}>
          <Text style={styles.sectionTitle}>All Photos ({records.length})</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.gallery}>
            {records.map((record) => (
              <TouchableOpacity
                key={record.id}
                onPress={() => setSelectedPhoto(record)}
                style={[
                  styles.galleryItem,
                  selectedPhoto?.id === record.id && styles.galleryItemSelected,
                ]}
              >
                <Image
                  source={{ uri: record.photo_url }}
                  style={styles.galleryPhoto}
                />
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>
      )}

      {/* Added: Editable form fields for profile information */}
      <View style={styles.formSection}>
        <Text style={styles.sectionTitle}>Profile Information</Text>

        <View style={styles.formGroup}>
          <Text style={styles.label}>Name</Text>
          <TextInput
            style={styles.input}
            value={name}
            onChangeText={setName}
            placeholder="Enter name"
          />
        </View>

        <View style={styles.formGroup}>
          <Text style={styles.label}>Event</Text>
          <TextInput
            style={styles.input}
            value={event}
            onChangeText={setEvent}
            placeholder="Enter event"
          />
        </View>

        <View style={styles.formGroup}>
          <Text style={styles.label}>Location</Text>
          <TextInput
            style={styles.input}
            value={location}
            onChangeText={setLocation}
            placeholder="Enter location"
          />
        </View>

        <View style={styles.formGroup}>
          <Text style={styles.label}>Date</Text>
          <TextInput
            style={styles.input}
            value={date}
            onChangeText={setDate}
            placeholder="YYYY-MM-DD"
          />
        </View>
      </View>

      {/* Added: Display recent history from all records with this name */}
      <View style={styles.recentSection}>
        <Text style={styles.sectionTitle}>Recent History</Text>

        {recentDates.length > 0 && (
          <View style={styles.recentItem}>
            <Text style={styles.recentLabel}>Recent Dates:</Text>
            <Text style={styles.recentValue}>{recentDates.join(', ')}</Text>
          </View>
        )}

        {recentLocations.length > 0 && (
          <View style={styles.recentItem}>
            <Text style={styles.recentLabel}>Recent Locations:</Text>
            <Text style={styles.recentValue}>{recentLocations.join(', ')}</Text>
          </View>
        )}

        {recentEvents.length > 0 && (
          <View style={styles.recentItem}>
            <Text style={styles.recentLabel}>Recent Events:</Text>
            <Text style={styles.recentValue}>{recentEvents.join(', ')}</Text>
          </View>
        )}
      </View>

      {/* Added: Delete button for individual photo records (only shown when multiple photos exist) */}
      {selectedPhoto && records.length > 1 && (
        <View style={styles.deleteSection}>
          <TouchableOpacity
            onPress={() => handleDelete(selectedPhoto.id)}
            style={styles.deleteButton}
          >
            <Ionicons name="trash-outline" size={18} color="#FF3B30" />
            <Text style={styles.deleteButtonText}>Delete This Photo</Text>
          </TouchableOpacity>
        </View>
      )}

      <View style={styles.spacer} />
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f9f9f9',
  },
  // Safe area wrapper to avoid header overlap on Android (status bar)
  safeArea: {
    flex: 1,
    backgroundColor: '#f9f9f9',
    paddingTop: Platform.OS === 'android' ? StatusBar.currentHeight || 0 : 0,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 15,
    paddingVertical: 15,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  saveButton: {
    fontSize: 16,
    color: '#007AFF',
    fontWeight: '600',
  },
  photoSection: {
    alignItems: 'center',
    paddingVertical: 20,
    backgroundColor: '#fff',
    marginBottom: 10,
  },
  mainPhoto: {
    width: 200,
    height: 200,
    borderRadius: 10,
    marginBottom: 10,
  },
  photoDate: {
    fontSize: 12,
    color: '#999',
  },
  gallerySection: {
    backgroundColor: '#fff',
    paddingVertical: 15,
    marginBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    paddingHorizontal: 15,
    marginBottom: 12,
  },
  gallery: {
    paddingHorizontal: 15,
  },
  galleryItem: {
    marginRight: 10,
    borderRadius: 8,
    overflow: 'hidden',
    borderWidth: 2,
    borderColor: 'transparent',
  },
  galleryItemSelected: {
    borderColor: '#007AFF',
  },
  galleryPhoto: {
    width: 80,
    height: 80,
  },
  formSection: {
    backgroundColor: '#fff',
    paddingVertical: 15,
    marginBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  formGroup: {
    paddingHorizontal: 15,
    marginBottom: 15,
  },
  label: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 8,
  },
  input: {
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 14,
    borderWidth: 1,
    borderColor: '#e0e0e0',
    color: '#333',
  },
  recentSection: {
    backgroundColor: '#fff',
    paddingVertical: 15,
    marginBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  recentItem: {
    paddingHorizontal: 15,
    marginBottom: 12,
  },
  recentLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#666',
    marginBottom: 4,
  },
  recentValue: {
    fontSize: 13,
    color: '#333',
    lineHeight: 18,
  },
  deleteSection: {
    backgroundColor: '#fff',
    paddingVertical: 15,
    alignItems: 'center',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  deleteButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 12,
    backgroundColor: '#fff3cd',
    borderRadius: 8,
  },
  deleteButtonText: {
    marginLeft: 10,
    fontSize: 14,
    color: '#FF3B30',
    fontWeight: '600',
  },
  spacer: {
    height: 20,
  },
});
