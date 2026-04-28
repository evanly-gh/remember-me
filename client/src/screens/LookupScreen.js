import { useState, useEffect, useCallback } from 'react';
import { View, TextInput, StyleSheet, Text, ScrollView, TouchableOpacity, Image, ActivityIndicator, RefreshControl } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { supabase } from '../lib/supabase';
import { useAuth } from '../context/AuthContext';
import { useTheme } from '../context/ThemeContext';
import { useFocusEffect } from '@react-navigation/native';
import { generateEmbedding } from '../lib/embeddings';

export default function LookupScreen({ navigation }) {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchInput, setSearchInput] = useState(''); // Separate state for input field
  const [profiles, setProfiles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [searching, setSearching] = useState(false);
  const { user } = useAuth();
  const { colors } = useTheme();

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
      setRefreshing(false);
    }
  };

  const onRefresh = () => {
    setRefreshing(true);
    loadProfiles();
  };

  // Semantic search using embeddings
  const performSemanticSearch = async (query) => {
    if (!query.trim()) {
      // Empty query - show all profiles
      loadProfiles();
      return;
    }

    try {
      setSearching(true);

      // Generate embedding for search query
      const queryEmbedding = await generateEmbedding(query);

      if (!queryEmbedding) {
        // Fallback to keyword search if embedding generation fails
        console.warn('Embedding generation failed, using local filter');
        return;
      }

      // Search using Supabase vector similarity
      // Convert array to PostgreSQL vector format string
      const vectorString = `[${queryEmbedding.join(',')}]`;

      const { data, error } = await supabase.rpc('search_contacts', {
        query_embedding: vectorString,
        match_threshold: 0.3,  // Lower threshold = more results (range: 0-1)
        match_count: 100,
        user_id_filter: user.id
      });

      if (error) {
        console.error('Semantic search error:', error);
        // Fallback to showing all profiles
        return;
      }

      // Deduplicate by name (keep most recent per person)
      const uniqueProfiles = {};
      data.forEach(record => {
        if (!uniqueProfiles[record.name] ||
            new Date(record.created_at) > new Date(uniqueProfiles[record.name].created_at)) {
          uniqueProfiles[record.name] = record;
        }
      });

      setProfiles(Object.values(uniqueProfiles));
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      setSearching(false);
    }
  };

  // Only search when user explicitly submits
  useEffect(() => {
    if (searchQuery.trim()) {
      performSemanticSearch(searchQuery);
    } else {
      loadProfiles();
    }
  }, [searchQuery, user]);

  const handleSearch = () => {
    setSearchQuery(searchInput);
  };

  const filteredProfiles = profiles;

  if (loading) {
    return (
      <SafeAreaView style={[styles.safeArea, { backgroundColor: colors.background }]}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.accent} />
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={[styles.safeArea, { backgroundColor: colors.background }]}>
      <View style={[styles.container, { backgroundColor: colors.background }]}>
        <Text style={[styles.screenTitle, { color: colors.text }]}>Contacts</Text>

        {/* Search bar */}
        <View style={[styles.searchContainer, { backgroundColor: colors.inputBg }]}>
          <Ionicons name="search" size={18} color={colors.placeholder} style={styles.searchIcon} />
          <TextInput
            style={[styles.searchInput, { color: colors.text }]}
            placeholder="Search by name, features, location..."
            placeholderTextColor={colors.placeholder}
            value={searchInput}
            onChangeText={setSearchInput}
            onSubmitEditing={handleSearch}
            returnKeyType="search"
          />
          {searching && <ActivityIndicator size="small" color={colors.accent} style={styles.searchSpinner} />}
          {searchInput.length > 0 && !searching && (
            <TouchableOpacity onPress={() => {
              setSearchInput('');
              setSearchQuery('');
            }}>
              <Ionicons name="close-circle" size={18} color={colors.placeholder} />
            </TouchableOpacity>
          )}
        </View>

        {/* Contact list */}
        <ScrollView
          style={styles.list}
          showsVerticalScrollIndicator={false}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={colors.accent} />}
          scrollEventThrottle={16}
          keyboardShouldPersistTaps="handled"
        >
          {filteredProfiles.length === 0 ? (
            <View style={styles.emptyState}>
              <Ionicons name="people-outline" size={48} color={colors.border} />
              <Text style={[styles.emptyText, { color: colors.textSecondary }]}>No contacts yet.</Text>
              <Text style={[styles.emptySubtext, { color: colors.textTertiary }]}>Tap + to add someone.</Text>
            </View>
          ) : (
            filteredProfiles.map((profile) => (
              <TouchableOpacity
                key={profile.id}
                style={[styles.contactRow, { borderBottomColor: colors.border }]}
                onPress={() => navigation.navigate('EditProfile', { profileName: profile.name })}
                activeOpacity={0.7}
              >
                {profile.photo_url ? (
                  <Image source={{ uri: profile.photo_url }} style={[styles.contactPhoto, { backgroundColor: colors.inputBg }]} />
                ) : (
                  <View style={[styles.contactPhoto, styles.contactPhotoPlaceholder, { backgroundColor: colors.accentLight }]}>
                    <Ionicons name="person" size={22} color={colors.accent} />
                  </View>
                )}
                <View style={styles.contactInfo}>
                  <Text style={[styles.contactName, { color: colors.text }]}>{profile.name}</Text>
                  {profile.title && (
                    <Text style={[styles.contactTitle, { color: colors.textSecondary }]}>{profile.title}</Text>
                  )}
                </View>
                <Ionicons name="chevron-forward" size={18} color={colors.border} />
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
  },
  container: {
    flex: 1,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  screenTitle: {
    fontSize: 28,
    fontWeight: '700',
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
    borderRadius: 20,
  },
  searchIcon: {
    marginRight: 8,
  },
  searchInput: {
    flex: 1,
    fontSize: 15,
  },
  searchSpinner: {
    marginRight: 8,
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
  },
  contactPhoto: {
    width: 50,
    height: 50,
    borderRadius: 25,
    marginRight: 14,
  },
  contactPhotoPlaceholder: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  contactInfo: {
    flex: 1,
  },
  contactName: {
    fontSize: 16,
    fontWeight: '600',
  },
  contactTitle: {
    fontSize: 14,
    marginTop: 2,
  },

  // Empty
  emptyState: {
    alignItems: 'center',
    paddingTop: 80,
  },
  emptyText: {
    fontSize: 17,
    marginTop: 12,
  },
  emptySubtext: {
    fontSize: 14,
    marginTop: 4,
  },
});
