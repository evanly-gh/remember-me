import { useState, useEffect, useCallback } from 'react';
import { View, TextInput, StyleSheet, Text, ScrollView, TouchableOpacity, Image, ActivityIndicator, RefreshControl, DeviceEventEmitter } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { supabase } from '../lib/supabase';
import { useAuth } from '../context/AuthContext';
import { useTheme } from '../context/ThemeContext';
import { useFocusEffect } from '@react-navigation/native';
import { generateEmbedding } from '../lib/embeddings';

/**
 * LookupScreen - Contact search with hybrid approach
 *
 * Search strategy:
 * 1. Exact text matching (priority 1): name, phone, title, location, event, notes
 * 2. Semantic vector search (priority 2): facial features, descriptive queries
 * 3. Results merged and deduplicated, exact matches appear first
 */
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
      if (user) {
        // Clear search query and show all contacts when screen is focused
        setSearchInput('');
        setSearchQuery('');
        loadProfiles();
      }
    }, [user])
  );

  // Clear search when tapping the contacts tab (even when already on this screen)
  useEffect(() => {
    const subscription = DeviceEventEmitter.addListener('contactsTabPress', () => {
      setSearchInput('');
      setSearchQuery('');
    });

    return () => subscription.remove();
  }, []);

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

  // Hybrid search: exact matches first, then semantic search
  const performSemanticSearch = async (query) => {
    if (!query.trim()) {
      // Empty query - show all profiles
      loadProfiles();
      return;
    }

    try {
      setSearching(true);

      const resultsMap = new Map(); // Use Map to track by ID and avoid duplicates
      const queryLower = query.trim();

      // STEP 1: Exact text matching (highest priority)
      // Search across name, phone, title, location, event, and notes
      const { data: exactMatches, error: exactError } = await supabase
        .from('people')
        .select('*')
        .eq('user_id', user.id)
        .or(`name.ilike.%${queryLower}%,phone.ilike.%${queryLower}%,title.ilike.%${queryLower}%,location.ilike.%${queryLower}%,event.ilike.%${queryLower}%,notes.ilike.%${queryLower}%`);

      if (exactError) {
        console.error('Exact match error:', exactError);
      } else if (exactMatches) {
        exactMatches.forEach(record => {
          resultsMap.set(record.id, { ...record, matchType: 'exact', priority: 1 });
        });
      }

      // STEP 2: Semantic search for complex/descriptive queries
      try {
        const queryEmbedding = await generateEmbedding(query);

        if (queryEmbedding) {
          const vectorString = `[${queryEmbedding.join(',')}]`;

          const { data: semanticMatches, error: semanticError } = await supabase.rpc('search_contacts', {
            query_embedding: vectorString,
            match_threshold: 0.3,  // Lower threshold = more results (range: 0-1)
            match_count: 100,
            user_id_filter: user.id
          });

          if (semanticError) {
            console.error('Semantic search error:', semanticError);
          } else if (semanticMatches) {
            semanticMatches.forEach(record => {
              // Only add if not already found via exact match
              if (!resultsMap.has(record.id)) {
                resultsMap.set(record.id, { ...record, matchType: 'semantic', priority: 2 });
              }
            });
          }
        } else {
          console.warn('Embedding generation failed, using exact matches only');
        }
      } catch (embeddingError) {
        console.warn('Semantic search failed, using exact matches only:', embeddingError.message);
      }

      // STEP 3: Combine and sort results (exact matches first, then semantic)
      const allResults = Array.from(resultsMap.values()).sort((a, b) => {
        // Sort by priority (1 = exact, 2 = semantic), then by created_at
        if (a.priority !== b.priority) {
          return a.priority - b.priority;
        }
        return new Date(b.created_at) - new Date(a.created_at);
      });

      // STEP 4: Deduplicate by name (keep most recent per person)
      const uniqueProfiles = {};
      allResults.forEach(record => {
        if (!uniqueProfiles[record.name] ||
            new Date(record.created_at) > new Date(uniqueProfiles[record.name].created_at)) {
          uniqueProfiles[record.name] = record;
        }
      });

      setProfiles(Object.values(uniqueProfiles));

      // Debug logging
      console.log(`Search: "${query}" → ${exactMatches?.length || 0} exact, ${resultsMap.size - (exactMatches?.length || 0)} semantic, ${Object.keys(uniqueProfiles).length} unique`);

    } catch (error) {
      console.error('Search error:', error);
      // Fallback to showing all profiles on error
      loadProfiles();
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
