import { useState, useEffect, useCallback } from 'react';
import { View, TextInput, StyleSheet, Text, ScrollView, TouchableOpacity, Image, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { supabase } from '../lib/supabase';
import { useAuth } from '../context/AuthContext';
import { useFocusEffect } from '@react-navigation/native';

export default function LookupScreen({ navigation }) {
  const [searchQuery, setSearchQuery] = useState('');
  const [profiles, setProfiles] = useState([]);
  const [loading, setLoading] = useState(true);
  const { user } = useAuth();

  useEffect(() => {
    if (user) loadProfiles();
  }, [user]);

  useFocusEffect(
    useCallback(() => {
      if (user) loadProfiles();
    }, [user])
  );

  const loadProfiles = async () => {
    if (!user) return;

    try {
      setLoading(true);
      const { data: allRecords, error } = await supabase
        .from('people')
        .select('*')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false });

      if (error) throw error;

      const uniqueProfiles = {};
      allRecords.forEach(record => {
        if (!uniqueProfiles[record.name] ||
            new Date(record.created_at) > new Date(uniqueProfiles[record.name].created_at)) {
          uniqueProfiles[record.name] = record;
        }
      });

      setProfiles(Object.values(uniqueProfiles));
    } catch (error) {
      console.error('Error loading profiles:', error);
    } finally {
      setLoading(false);
    }
  };

  const filteredProfiles = profiles.filter(profile => {
    const query = searchQuery.toLowerCase();

    if (profile.name.toLowerCase().includes(query)) return true;

    if (profile.facial_details) {
      try {
        const details = typeof profile.facial_details === 'string'
          ? JSON.parse(profile.facial_details)
          : profile.facial_details;

        const d = details?.data || details;
        if (d) {
          if ((query.includes('smil') || query.includes('happy')) && (d.smiling || d.smiling_celeba)) return true;
          if (query.includes('beard') && d.has_beard) return true;
          if (d.primary_emotion && d.primary_emotion.toLowerCase().includes(query)) return true;
          if (query.includes('glass') && (d.wearing_glasses || d.glasses_detected)) return true;
          if (d.gender && d.gender.toLowerCase().includes(query)) return true;
          if (d.face_shape && d.face_shape.toLowerCase().includes(query)) return true;
          if (d.hair_color_celeba && d.hair_color_celeba.toLowerCase().includes(query)) return true;
          if (d.hair_color?.name && d.hair_color.name.toLowerCase().includes(query)) return true;
          if (d.eye_color && typeof d.eye_color === 'string' && d.eye_color.toLowerCase().includes(query)) return true;
          if (d.ethnicity && d.ethnicity.toLowerCase().includes(query)) return true;
          if (d.age_range && d.age_range.includes(query)) return true;
          if (d.hair_length && d.hair_length.toLowerCase().includes(query)) return true;
          if (d.eye_shape && d.eye_shape.toLowerCase().includes(query)) return true;
          if (query.includes('hat') && (d.wearing_hat || d.hat_detected)) return true;
          if (query.includes('bald') && d.is_bald) return true;
          if (query.includes('young') && d.young) return true;
          const textFields = ['jawline_type', 'chin_type', 'nose_shape', 'lip_fullness', 'eye_depth', 'eye_spacing', 'hair_texture', 'skin_undertone'];
          for (const field of textFields) {
            if (d[field] && d[field].toLowerCase().includes(query)) return true;
          }
        }
      } catch (error) {
        console.error('Error parsing facial_details:', error);
      }
    }

    return false;
  });

  if (searchQuery.trim()) {
    filteredProfiles.sort((a, b) => a.name.localeCompare(b.name));
  }

  if (loading) {
    return (
      <SafeAreaView style={styles.safeArea}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#8B5CF6" />
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.container}>
        <Text style={styles.screenTitle}>Contacts</Text>

        {/* Search bar */}
        <View style={styles.searchContainer}>
          <Ionicons name="search" size={18} color="#9CA3AF" style={styles.searchIcon} />
          <TextInput
            style={styles.searchInput}
            placeholder="Search..."
            placeholderTextColor="#9CA3AF"
            value={searchQuery}
            onChangeText={setSearchQuery}
          />
          {searchQuery.length > 0 && (
            <TouchableOpacity onPress={() => setSearchQuery('')}>
              <Ionicons name="close-circle" size={18} color="#9CA3AF" />
            </TouchableOpacity>
          )}
        </View>

        {/* Contact list */}
        <ScrollView style={styles.list} showsVerticalScrollIndicator={false}>
          {filteredProfiles.length === 0 ? (
            <View style={styles.emptyState}>
              <Ionicons name="people-outline" size={48} color="#D1D5DB" />
              <Text style={styles.emptyText}>No contacts yet.</Text>
              <Text style={styles.emptySubtext}>Tap + to add someone.</Text>
            </View>
          ) : (
            filteredProfiles.map((profile) => (
              <TouchableOpacity
                key={profile.id}
                style={styles.contactRow}
                onPress={() => navigation.navigate('EditProfile', { profileName: profile.name })}
                activeOpacity={0.6}
              >
                {profile.photo_url ? (
                  <Image source={{ uri: profile.photo_url }} style={styles.contactPhoto} />
                ) : (
                  <View style={[styles.contactPhoto, styles.contactPhotoPlaceholder]}>
                    <Ionicons name="person" size={22} color="#8B5CF6" />
                  </View>
                )}
                <View style={styles.contactInfo}>
                  <Text style={styles.contactName}>{profile.name}</Text>
                  {profile.title && (
                    <Text style={styles.contactTitle}>{profile.title}</Text>
                  )}
                </View>
                <Ionicons name="chevron-forward" size={18} color="#D1D5DB" />
              </TouchableOpacity>
            ))
          )}
        </ScrollView>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#fff',
  },
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  screenTitle: {
    fontSize: 28,
    fontWeight: '700',
    color: '#1F2937',
    paddingHorizontal: 20,
    paddingTop: 10,
    paddingBottom: 12,
  },

  // Search
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginHorizontal: 20,
    marginBottom: 12,
    paddingHorizontal: 14,
    height: 40,
    backgroundColor: '#F3F4F6',
    borderRadius: 20,
  },
  searchIcon: {
    marginRight: 8,
  },
  searchInput: {
    flex: 1,
    fontSize: 15,
    color: '#1F2937',
  },

  // List
  list: {
    flex: 1,
  },
  contactRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#F3F4F6',
  },
  contactPhoto: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: '#F3F4F6',
    marginRight: 14,
  },
  contactPhotoPlaceholder: {
    backgroundColor: '#EDE9FE',
    justifyContent: 'center',
    alignItems: 'center',
  },
  contactInfo: {
    flex: 1,
  },
  contactName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1F2937',
  },
  contactTitle: {
    fontSize: 14,
    color: '#6B7280',
    marginTop: 2,
  },

  // Empty
  emptyState: {
    alignItems: 'center',
    paddingTop: 80,
  },
  emptyText: {
    fontSize: 17,
    color: '#6B7280',
    marginTop: 12,
  },
  emptySubtext: {
    fontSize: 14,
    color: '#9CA3AF',
    marginTop: 4,
  },
});
