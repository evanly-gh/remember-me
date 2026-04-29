import { useState, useEffect, useCallback } from 'react';
import {
  View,
  TextInput,
  StyleSheet,
  Text,
  ScrollView,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  Alert,
  RefreshControl,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useFocusEffect } from '@react-navigation/native';
import { supabase } from '../lib/supabase';
import { useAuth } from '../context/AuthContext';
import { useTheme } from '../context/ThemeContext';

export default function EditProfileScreen({ route, navigation }) {
  const { profileName } = route.params;
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedPhoto, setSelectedPhoto] = useState(null);
  const { user } = useAuth();
  const { colors } = useTheme();

  const [name, setName] = useState(profileName);
  const [phone, setPhone] = useState('');
  const [title, setTitle] = useState('');
  const [event, setEvent] = useState('');
  const [location, setLocation] = useState('');
  const [date, setDate] = useState('');
  const [notes, setNotes] = useState('');
  const [showFacialDetails, setShowFacialDetails] = useState(false);

  useEffect(() => {
    if (user) loadRecords();
  }, [user, profileName]);


  useFocusEffect(
    useCallback(() => {
      loadSettings();
    }, [])
  );

  const loadSettings = async () => {
    try {
      const value = await AsyncStorage.getItem('@hcp:settings:showFacialDetails');
      setShowFacialDetails(value === 'true');
    } catch (error) {
      console.error('Error loading settings:', error);
    }
  };

  const loadRecords = async () => {
    try {
      setLoading(true);
      const { data, error } = await supabase
        .from('people')
        .select('*')
        .eq('user_id', user.id)
        .eq('name', profileName)
        .order('created_at', { ascending: false });

      if (error) throw error;

      setRecords(data);

      if (data && data.length > 0) {
        const mostRecent = data[0];
        setSelectedPhoto(mostRecent);
        setPhone(mostRecent.phone || '');
        setTitle(mostRecent.title || '');
        setEvent(mostRecent.event || '');
        setLocation(mostRecent.location || '');
        setDate(mostRecent.date || '');
        setNotes(mostRecent.notes || '');
      }
    } catch (error) {
      console.error('Error loading records:', error);
      Alert.alert('Error', 'Failed to load profile records');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const onRefresh = () => {
    setRefreshing(true);
    loadRecords();
  };

  const handleSave = async () => {
    if (!selectedPhoto) {
      Alert.alert('Error', 'No record selected');
      return;
    }

    try {
      setSaving(true);
      const { error } = await supabase
        .from('people')
        .update({
          name: name.trim(),
          phone: phone.trim() || null,
          title: title.trim() || null,
          event: event.trim() || null,
          location: location.trim() || null,
          date: date || null,
          notes: notes.trim() || '',
        })
        .eq('id', selectedPhoto.id);

      if (error) throw error;

      Alert.alert('Saved', 'Profile updated successfully');
      navigation.goBack();
    } catch (error) {
      console.error('Error saving profile:', error);
      Alert.alert('Error', 'Failed to save profile');
    } finally {
      setSaving(false);
    }
  };

  // Helper function to extract file path from Supabase public URL
  const extractFilePathFromUrl = (url) => {
    if (!url) return null;
    // URL format: https://<project>.supabase.co/storage/v1/object/public/photos/<user_id>/<timestamp>.jpg
    const match = url.match(/\/photos\/(.+)$/);
    return match ? match[1] : null;
  };

  const handleDeletePhoto = async (recordId) => {
    Alert.alert(
      'Delete Photo',
      'Are you sure you want to delete this photo?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          onPress: async () => {
            try {
              // Find the record to get the photo URL
              const record = records.find(r => r.id === recordId);

              // Delete from storage first (if photo exists)
              if (record?.photo_url) {
                const fileName = extractFilePathFromUrl(record.photo_url);
                if (fileName) {
                  const { error: storageError } = await supabase.storage
                    .from('photos')
                    .remove([fileName]);

                  if (storageError) {
                    console.warn('Failed to delete storage file:', storageError);
                    // Continue with DB deletion anyway
                  }
                }
              }

              // Then delete from database
              const { error } = await supabase
                .from('people')
                .delete()
                .eq('id', recordId);

              if (error) throw error;

              Alert.alert('Deleted', 'Photo removed');
              loadRecords();
            } catch (error) {
              console.error('Error deleting record:', error);
              Alert.alert('Error', 'Failed to delete photo');
            }
          },
          style: 'destructive',
        },
      ]
    );
  };

  const handleDeleteContact = () => {
    Alert.alert(
      'Delete Contact',
      `Delete ${name} and all associated photos? This cannot be undone.`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          onPress: async () => {
            try {
              // Collect all photo file paths to delete from storage
              const filePaths = records
                .map(record => extractFilePathFromUrl(record.photo_url))
                .filter(Boolean);

              // Delete all photos from storage
              if (filePaths.length > 0) {
                const { error: storageError } = await supabase.storage
                  .from('photos')
                  .remove(filePaths);

                if (storageError) {
                  console.warn('Failed to delete some storage files:', storageError);
                  // Continue with DB deletion anyway
                }
              }

              // Then delete all records from database
              const { error } = await supabase
                .from('people')
                .delete()
                .eq('user_id', user.id)
                .eq('name', profileName);

              if (error) throw error;

              navigation.goBack();
            } catch (error) {
              console.error('Error deleting contact:', error);
              Alert.alert('Error', 'Failed to delete contact');
            }
          },
          style: 'destructive',
        },
      ]
    );
  };

  if (loading) {
    return (
      <SafeAreaView style={[styles.safeArea, { backgroundColor: colors.background }]}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.accent} />
        </View>
      </SafeAreaView>
    );
  }

  const recentDates = records.slice(0, 3).map(r => r.date).filter(Boolean);
  const recentLocations = records.slice(0, 3).map(r => r.location).filter(Boolean);
  const recentEvents = records.slice(0, 3).map(r => r.event).filter(Boolean);

  return (
    <SafeAreaView style={[styles.safeArea, { backgroundColor: colors.background }]}>
      {/* Header */}
      <View style={[styles.header, { borderBottomColor: colors.border }]}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={styles.headerButton}>
          <Ionicons name="chevron-back" size={24} color={colors.accent} />
        </TouchableOpacity>
        <TouchableOpacity onPress={handleSave} disabled={saving} style={styles.headerButton}>
          {saving ? (
            <ActivityIndicator size="small" color={colors.accent} />
          ) : (
            <Text style={[styles.saveText, { color: colors.accent }]}>Save</Text>
          )}
        </TouchableOpacity>
      </View>

      <ScrollView
        style={[styles.container, { backgroundColor: colors.background }]}
        showsVerticalScrollIndicator={false}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={colors.accent} />}
      >
        {/* Hero section */}
        <View style={styles.heroSection}>
          {selectedPhoto?.photo_url ? (
            <Image source={{ uri: selectedPhoto.photo_url }} style={[styles.heroPhoto, { backgroundColor: colors.inputBg }]} />
          ) : (
            <View style={[styles.heroPhoto, styles.heroPhotoPlaceholder, { backgroundColor: colors.accentLight }]}>
              <Ionicons name="person" size={48} color={colors.accent} />
            </View>
          )}
          <Text style={[styles.heroName, { color: colors.text }]}>{name}</Text>
          {title ? <Text style={[styles.heroTitle, { color: colors.textSecondary }]}>{title}</Text> : null}
          {selectedPhoto?.created_at && (
            <Text style={[styles.heroDate, { color: colors.textTertiary }]}>
              {new Date(selectedPhoto.created_at).toLocaleDateString()}
            </Text>
          )}
        </View>

        {/* Photo gallery */}
        {records.length > 1 && (
          <View style={[styles.gallerySection, { borderBottomColor: colors.border }]}>
            <Text style={[styles.sectionHeader, { color: colors.textTertiary }]}>Photos ({records.length})</Text>
            <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.gallery}>
              {records.map((record) => (
                <TouchableOpacity
                  key={record.id}
                  onPress={() => setSelectedPhoto(record)}
                  style={[
                    styles.galleryItem,
                    selectedPhoto?.id === record.id && [styles.galleryItemSelected, { borderColor: colors.accent }],
                  ]}
                >
                  <Image source={{ uri: record.photo_url }} style={styles.galleryPhoto} />
                </TouchableOpacity>
              ))}
            </ScrollView>
          </View>
        )}

        {/* Info fields */}
        <View style={[styles.infoSection, { borderBottomColor: colors.border }]}>
          <Text style={[styles.sectionHeader, { color: colors.textTertiary }]}>Information</Text>

          <View style={styles.fieldRow}>
            <Text style={[styles.fieldLabel, { color: colors.textSecondary }]}>Name</Text>
            <TextInput
              style={[styles.fieldValue, { color: colors.text }]}
              value={name}
              onChangeText={setName}
              placeholder="Name"
              placeholderTextColor={colors.placeholder}
            />
          </View>
          <View style={[styles.fieldSeparator, { backgroundColor: colors.border }]} />

          <View style={styles.fieldRow}>
            <Text style={[styles.fieldLabel, { color: colors.textSecondary }]}>Phone</Text>
            <TextInput
              style={[styles.fieldValue, { color: colors.text }]}
              value={phone}
              onChangeText={setPhone}
              placeholder="Add phone"
              placeholderTextColor={colors.placeholder}
              keyboardType="phone-pad"
            />
          </View>
          <View style={[styles.fieldSeparator, { backgroundColor: colors.border }]} />

          <View style={styles.fieldRow}>
            <Text style={[styles.fieldLabel, { color: colors.textSecondary }]}>Relation</Text>
            <TextInput
              style={[styles.fieldValue, { color: colors.text }]}
              value={title}
              onChangeText={setTitle}
              placeholder="Add relation"
              placeholderTextColor={colors.placeholder}
            />
          </View>
          <View style={[styles.fieldSeparator, { backgroundColor: colors.border }]} />

          <View style={styles.fieldRow}>
            <Text style={[styles.fieldLabel, { color: colors.textSecondary }]}>Occasion</Text>
            <TextInput
              style={[styles.fieldValue, { color: colors.text }]}
              value={event}
              onChangeText={setEvent}
              placeholder="Add event"
              placeholderTextColor={colors.placeholder}
            />
          </View>
          <View style={[styles.fieldSeparator, { backgroundColor: colors.border }]} />

          <View style={styles.fieldRow}>
            <Text style={[styles.fieldLabel, { color: colors.textSecondary }]}>Location</Text>
            <TextInput
              style={[styles.fieldValue, { color: colors.text }]}
              value={location}
              onChangeText={setLocation}
              placeholder="Add location"
              placeholderTextColor={colors.placeholder}
            />
          </View>
          <View style={[styles.fieldSeparator, { backgroundColor: colors.border }]} />

          <View style={styles.fieldRow}>
            <Text style={[styles.fieldLabel, { color: colors.textSecondary }]}>Date</Text>
            <TextInput
              style={[styles.fieldValue, { color: colors.text }]}
              value={date}
              onChangeText={setDate}
              placeholder="YYYY-MM-DD"
              placeholderTextColor={colors.placeholder}
            />
          </View>
        </View>

        {/* Notes */}
        <View style={[styles.infoSection, { borderBottomColor: colors.border }]}>
          <Text style={[styles.sectionHeader, { color: colors.textTertiary }]}>Notes</Text>
          <View style={styles.notesContainer}>
            <TextInput
              style={[styles.notesInput, { color: colors.text, backgroundColor: colors.inputBg }]}
              value={notes}
              onChangeText={setNotes}
              placeholder="Add notes..."
              placeholderTextColor={colors.placeholder}
              multiline
              textAlignVertical="top"
            />
          </View>
        </View>

        {/* Recent history */}
        {(recentDates.length > 0 || recentLocations.length > 0 || recentEvents.length > 0) && (
          <View style={[styles.infoSection, { borderBottomColor: colors.border }]}>
            <Text style={[styles.sectionHeader, { color: colors.textTertiary }]}>Recent History</Text>

            {recentDates.length > 0 && (
              <View style={styles.historyRow}>
                <Ionicons name="calendar-outline" size={16} color={colors.accent} style={styles.historyIcon} />
                <Text style={[styles.historyText, { color: colors.text }]}>{recentDates.join(', ')}</Text>
              </View>
            )}

            {recentLocations.length > 0 && (
              <View style={styles.historyRow}>
                <Ionicons name="location-outline" size={16} color={colors.accent} style={styles.historyIcon} />
                <Text style={[styles.historyText, { color: colors.text }]}>{recentLocations.join(', ')}</Text>
              </View>
            )}

            {recentEvents.length > 0 && (
              <View style={styles.historyRow}>
                <Ionicons name="flag-outline" size={16} color={colors.accent} style={styles.historyIcon} />
                <Text style={[styles.historyText, { color: colors.text }]}>{recentEvents.join(', ')}</Text>
              </View>
            )}
          </View>
        )}

        {/* Facial details */}
        {showFacialDetails && selectedPhoto?.photo_url && !selectedPhoto?.facial_details && (
          <View style={[styles.infoSection, { borderBottomColor: colors.border }]}>
            <Text style={[styles.sectionHeader, { color: colors.textTertiary }]}>Facial Analysis</Text>
            <View style={styles.analyzingContainer}>
              <ActivityIndicator size="small" color={colors.accent} />
              <Text style={[styles.analyzingText, { color: colors.textSecondary }]}>Analyzing facial features...</Text>
            </View>
          </View>
        )}
        {showFacialDetails && selectedPhoto?.facial_details && (
          <View style={[styles.infoSection, { borderBottomColor: colors.border }]}>
            <Text style={[styles.sectionHeader, { color: colors.textTertiary }]}>Facial Analysis</Text>
            {(() => {
              try {
                const details = typeof selectedPhoto.facial_details === 'string'
                  ? JSON.parse(selectedPhoto.facial_details)
                  : selectedPhoto.facial_details;

                const d = details?.data || details;

                if (!d || details?.success === false) {
                  return (
                    <Text style={[styles.historyText, { color: colors.text, paddingHorizontal: 20 }]}>
                      {details?.error || 'No facial analysis data available'}
                    </Text>
                  );
                }

                const Row = ({ label, value }) => {
                  if (value === undefined || value === null || value === '') return null;
                  const displayVal = typeof value === 'boolean' ? (value ? 'Yes' : 'No') : String(value);
                  return (
                    <>
                      <View style={styles.fieldRow}>
                        <Text style={[styles.fieldLabel, { color: colors.textSecondary }]}>{label}</Text>
                        <Text style={[styles.fieldValueText, { color: colors.text }]}>{displayVal}</Text>
                      </View>
                      <View style={[styles.fieldSeparator, { backgroundColor: colors.border }]} />
                    </>
                  );
                };

                const SectionLabel = ({ children }) => (
                  <Text style={[styles.facialSubheader, { color: colors.accent }]}>{children}</Text>
                );

                return (
                  <>
                    <SectionLabel>Demographics</SectionLabel>
                    <Row label="Gender" value={d.gender} />
                    <Row label="Age Range" value={d.age_range} />
                    <Row label="Age Estimate" value={d.age_estimate} />
                    <Row label="Ethnicity" value={d.ethnicity} />

                    <SectionLabel>Emotion</SectionLabel>
                    <Row label="Primary Emotion" value={d.primary_emotion} />
                    <Row label="Secondary Emotion" value={d.secondary_emotion} />
                    <Row label="Mood" value={d.mood} />
                    <Row label="Smiling" value={d.smiling} />

                    <SectionLabel>Face Structure</SectionLabel>
                    <Row label="Face Shape" value={d.face_shape} />
                    <Row label="Jawline" value={d.jawline_type} />
                    <Row label="Chin" value={d.chin_type} />
                    <Row label="Cheekbones" value={d.cheekbone_prominence} />
                    <Row label="Forehead" value={d.forehead_width} />

                    <SectionLabel>Eyes</SectionLabel>
                    <Row label="Eye Shape" value={d.eye_shape} />
                    <Row label="Eye Color" value={typeof d.eye_color === 'string' ? d.eye_color : d.eye_color?.name} />
                    <Row label="Eye Depth" value={d.eye_depth} />
                    <Row label="Eye Spacing" value={d.eye_spacing} />
                    <Row label="Eye Size" value={d.eye_size} />

                    <SectionLabel>Eyebrows</SectionLabel>
                    <Row label="Shape" value={d.eyebrow_shape} />
                    <Row label="Arch" value={d.eyebrow_arch_height} />
                    <Row label="Thickness" value={d.eyebrow_thickness} />
                    <Row label="Arched (CelebA)" value={d.arched_eyebrows} />
                    <Row label="Bushy (CelebA)" value={d.bushy_eyebrows} />

                    <SectionLabel>Nose</SectionLabel>
                    <Row label="Shape" value={d.nose_shape} />
                    <Row label="Bridge" value={d.nose_bridge} />
                    <Row label="Tip" value={d.nose_tip_shape} />
                    <Row label="Nostril Width" value={d.nostril_width} />

                    <SectionLabel>Lips & Mouth</SectionLabel>
                    <Row label="Lip Fullness" value={d.lip_fullness} />
                    <Row label="Lip Balance" value={d.lip_balance} />
                    <Row label="Mouth Width" value={d.mouth_width} />
                    <Row label="Cupid's Bow" value={d.cupids_bow} />
                    <Row label="Lip Color" value={d.lip_color?.shade} />

                    <SectionLabel>Hair</SectionLabel>
                    <Row label="Hair Color" value={d.hair_color?.name || d.hair_color_celeba} />
                    <Row label="Hair Texture" value={d.hair_texture || d.hair_texture_celeba} />
                    <Row label="Hair Length" value={d.hair_length} />
                    <Row label="Bangs" value={d.has_bangs} />
                    <Row label="Bald" value={d.is_bald} />
                    <Row label="Receding Hairline" value={d.receding_hairline} />

                    <SectionLabel>Facial Hair</SectionLabel>
                    <Row label="Has Beard" value={d.has_beard} />
                    {d.facial_hair && (
                      <>
                        <Row label="Goatee" value={d.facial_hair.goatee} />
                        <Row label="Mustache" value={d.facial_hair.mustache} />
                        <Row label="Sideburns" value={d.facial_hair.sideburns} />
                      </>
                    )}

                    <SectionLabel>Skin</SectionLabel>
                    <Row label="Skin Tone" value={d.skin_tone?.fitzpatrick} />
                    <Row label="Skin Hex" value={d.skin_tone?.hex_color} />
                    <Row label="Undertone" value={d.skin_undertone} />
                    <Row label="Wrinkles" value={d.wrinkle_level} />
                    <Row label="Freckles/Moles" value={d.freckles_or_moles} />
                    <Row label="Pale Skin" value={d.pale_skin} />

                    <SectionLabel>Accessories</SectionLabel>
                    <Row label="Glasses" value={d.wearing_glasses || d.glasses_detected} />
                    <Row label="Hat" value={d.wearing_hat || d.hat_detected} />
                    <Row label="Earrings" value={d.wearing_earrings || d.earring_detected} />
                    <Row label="Necklace" value={d.wearing_necklace || d.necklace_detected} />
                    <Row label="Necktie" value={d.wearing_necktie} />
                    <Row label="Heavy Makeup" value={d.heavy_makeup} />
                    <Row label="Lipstick" value={d.wearing_lipstick} />

                    <SectionLabel>Analysis Models</SectionLabel>
                    <Row label="Landmarks" value="MediaPipe Face Landmarker — 478 3D landmarks + 52 blendshapes (Google)" />
                    <Row label="Age" value="dima806/fairface_age_image_detection — ViT, ~59% top-1 on FairFace buckets" />
                    <Row label="Gender" value="dima806/fairface_gender_image_detection — ViT, ~93.4% accuracy" />
                    <Row label="Ethnicity" value="cledoux42/Ethnicity_Test_v003 — ViT, 79.6% accuracy, macro-F1 0.797" />
                    <Row label="Attributes" value="openai/clip-vit-base-patch32 — zero-shot CLIP for ~30 facial attributes" />
                    <Row label="Face Parsing" value="matei-dorian/segformer-b5-finetuned-human-parsing — mIoU 0.6258, Face IoU 0.829, Hair IoU 0.817" />
                    <Row label="Emotion" value="HSEmotion (EfficientNet-B0) — 8-class, ~66.5% on AffectNet-8" />
                    <Row label="Color" value="Pixel-level LAB/HSV/K-means analysis (no AI model)" />
                  </>
                );
              } catch (error) {
                return <Text style={[styles.historyText, { color: colors.text, paddingHorizontal: 20 }]}>Error parsing facial data</Text>;
              }
            })()}
          </View>
        )}

        {/* Delete photo button (only when multiple photos) */}
        {selectedPhoto && records.length > 1 && (
          <TouchableOpacity
            onPress={() => handleDeletePhoto(selectedPhoto.id)}
            style={[styles.deletePhotoButton, { borderColor: colors.destructiveBorder, backgroundColor: colors.destructiveBg }]}
          >
            <Ionicons name="image-outline" size={16} color={colors.destructive} />
            <Text style={[styles.deleteText, { color: colors.destructive }]}>Delete This Photo</Text>
          </TouchableOpacity>
        )}

        {/* Delete entire contact */}
        <TouchableOpacity
          onPress={handleDeleteContact}
          style={[styles.deleteContactButton, { borderColor: colors.destructiveBorder, backgroundColor: colors.destructiveCardBg }]}
        >
          <Ionicons name="trash-outline" size={16} color={colors.destructive} />
          <Text style={[styles.deleteText, { color: colors.destructive }]}>Delete Contact</Text>
        </TouchableOpacity>

        <View style={styles.spacer} />
      </ScrollView>
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

  // Header
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 8,
    paddingVertical: 10,
    borderBottomWidth: 1,
  },
  headerButton: {
    padding: 8,
  },
  saveText: {
    fontSize: 16,
    fontWeight: '600',
  },

  // Hero
  heroSection: {
    alignItems: 'center',
    paddingVertical: 28,
  },
  heroPhoto: {
    width: 140,
    height: 140,
    borderRadius: 70,
    marginBottom: 16,
  },
  heroPhotoPlaceholder: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  heroName: {
    fontSize: 24,
    fontWeight: '700',
  },
  heroTitle: {
    fontSize: 16,
    marginTop: 4,
  },
  heroDate: {
    fontSize: 13,
    marginTop: 4,
  },

  // Gallery
  gallerySection: {
    paddingBottom: 20,
    borderBottomWidth: 1,
  },
  sectionHeader: {
    fontSize: 13,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    paddingHorizontal: 20,
    marginBottom: 12,
  },
  gallery: {
    paddingHorizontal: 20,
  },
  galleryItem: {
    marginRight: 10,
    borderRadius: 10,
    overflow: 'hidden',
    borderWidth: 2,
    borderColor: 'transparent',
  },
  galleryItemSelected: {},
  galleryPhoto: {
    width: 70,
    height: 70,
  },

  // Info fields
  infoSection: {
    paddingTop: 20,
    paddingBottom: 8,
    borderBottomWidth: 1,
  },
  fieldRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 12,
  },
  fieldLabel: {
    width: 90,
    fontSize: 14,
  },
  fieldValue: {
    flex: 1,
    fontSize: 15,
    textAlign: 'right',
  },
  fieldValueText: {
    flex: 1,
    fontSize: 15,
    textAlign: 'right',
  },
  fieldSeparator: {
    height: 1,
    marginLeft: 20,
  },

  // Notes
  notesContainer: {
    paddingHorizontal: 20,
    paddingBottom: 8,
  },
  notesInput: {
    fontSize: 15,
    borderRadius: 10,
    paddingHorizontal: 14,
    paddingTop: 12,
    paddingBottom: 12,
    minHeight: 100,
  },

  // Facial details
  analyzingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 16,
  },
  analyzingText: {
    marginLeft: 10,
    fontSize: 14,
  },
  facialSubheader: {
    fontSize: 13,
    fontWeight: '600',
    paddingHorizontal: 20,
    paddingTop: 12,
    paddingBottom: 4,
  },

  // History
  historyRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    paddingHorizontal: 20,
    paddingVertical: 8,
  },
  historyIcon: {
    marginRight: 10,
    marginTop: 2,
  },
  historyText: {
    flex: 1,
    fontSize: 14,
    lineHeight: 20,
  },

  // Delete
  deletePhotoButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginHorizontal: 20,
    marginTop: 24,
    paddingVertical: 14,
    borderRadius: 12,
    borderWidth: 1,
  },
  deleteContactButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginHorizontal: 20,
    marginTop: 12,
    paddingVertical: 14,
    borderRadius: 12,
    borderWidth: 1,
  },
  deleteText: {
    marginLeft: 8,
    fontSize: 15,
    fontWeight: '500',
  },

  spacer: {
    height: 40,
  },
});
