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

                const Row = ({ label, value, method }) => {
                  if (value === undefined || value === null || value === '') return null;
                  const displayVal = typeof value === 'boolean' ? (value ? 'Yes' : 'No') : String(value);
                  const methodLabel = method ? ` (${method})` : '';
                  return (
                    <>
                      <View style={styles.fieldRow}>
                        <Text style={[styles.fieldLabel, { color: colors.textSecondary }]}>
                          {label}
                          <Text style={{ fontSize: 12, color: colors.textTertiary }}>{methodLabel}</Text>
                        </Text>
                        <Text style={[styles.fieldValueText, { color: colors.text }]}>{displayVal}</Text>
                      </View>
                      <View style={[styles.fieldSeparator, { backgroundColor: colors.border }]} />
                    </>
                  );
                };

                const SectionLabel = ({ children }) => (
                  <Text style={[styles.facialSubheader, { color: colors.accent }]}>{children}</Text>
                );

                const ExpandableSection = ({ title, children }) => {
                  const [expanded, setExpanded] = useState(false);
                  return (
                    <View>
                      <TouchableOpacity
                        onPress={() => setExpanded(!expanded)}
                        style={[styles.fieldRow, { justifyContent: 'space-between', paddingVertical: 14 }]}
                      >
                        <Text style={[styles.facialSubheader, { color: colors.accent, marginLeft: 0, paddingLeft: 0 }]}>
                          {title}
                        </Text>
                        <Ionicons
                          name={expanded ? 'chevron-up' : 'chevron-down'}
                          size={20}
                          color={colors.accent}
                        />
                      </TouchableOpacity>
                      {expanded && children}
                    </View>
                  );
                };

                return (
                  <>
                    {/* ============ DEMOGRAPHICS ============ */}
                    <SectionLabel>Demographics</SectionLabel>
                    <Row label="Gender" value={d.gender} method="FairFace" />
                    <Row label="Gender Confidence" value={d.gender_confidence && `${(d.gender_confidence * 100).toFixed(1)}%`} method="FairFace" />
                    <Row label="Age Range" value={d.age_range} method="FairFace" />
                    <Row label="Age Estimate" value={d.age_estimate} method="FairFace" />
                    <Row label="Age Confidence" value={d.age_confidence && `${(d.age_confidence * 100).toFixed(1)}%`} method="FairFace" />
                    {d.age_distribution && Object.keys(d.age_distribution).length > 0 && (
                      <Row label="Age Distribution" value={Object.entries(d.age_distribution).map(([k, v]) => `${k}: ${(v * 100).toFixed(1)}%`).join(', ')} method="FairFace" />
                    )}
                    <Row label="Ethnicity" value={d.ethnicity} method="Ethnicity_Test_v003" />
                    <Row label="Ethnicity Confidence" value={d.ethnicity_confidence && `${(d.ethnicity_confidence * 100).toFixed(1)}%`} method="Ethnicity_Test_v003" />
                    {d.ethnicity_distribution && Object.keys(d.ethnicity_distribution).length > 0 && (
                      <Row label="Ethnicity Distribution" value={Object.entries(d.ethnicity_distribution).filter(([, v]) => v > 0).map(([k, v]) => `${k}: ${(v * 100).toFixed(1)}%`).join(', ')} method="Ethnicity_Test_v003" />
                    )}

                    {/* ============ EMOTION ============ */}
                    <SectionLabel>Emotion</SectionLabel>
                    <Row label="Primary Emotion" value={d.primary_emotion} method="HSEmotion" />
                    <Row label="Emotion Confidence" value={d.emotion_confidence && `${(d.emotion_confidence * 100).toFixed(1)}%`} method="HSEmotion" />
                    <Row label="Secondary Emotion" value={d.secondary_emotion} method="HSEmotion" />
                    {d.emotion_scores && Object.keys(d.emotion_scores).length > 0 && (
                      <Row label="Emotion Scores" value={Object.entries(d.emotion_scores).map(([k, v]) => `${k}: ${(v * 100).toFixed(1)}%`).join(', ')} method="HSEmotion" />
                    )}
                    <Row label="Valence" value={d.valence !== undefined ? `${(d.valence * 100).toFixed(1)}%` : undefined} method="HSEmotion" />
                    <Row label="Arousal" value={d.arousal !== undefined ? `${(d.arousal * 100).toFixed(1)}%` : undefined} method="HSEmotion" />
                    <Row label="Mood" value={d.mood} method="HSEmotion" />

                    {/* ============ FACE STRUCTURE ============ */}
                    <SectionLabel>Face Structure</SectionLabel>
                    <Row label="Face Shape" value={d.face_shape} method="MediaPipe" />
                    {d.face_shape_metrics && (
                      <>
                        <Row label="Width-Height Ratio" value={d.face_shape_metrics.width_height_ratio} method="MediaPipe" />
                        <Row label="Jaw-to-Face Ratio" value={d.face_shape_metrics.jaw_to_face_ratio} method="MediaPipe" />
                        <Row label="Forehead-to-Jaw Ratio" value={d.face_shape_metrics.forehead_to_jaw_ratio} method="MediaPipe" />
                        <Row label="Cheekbone-to-Jaw Ratio" value={d.face_shape_metrics.cheekbone_to_jaw_ratio} method="MediaPipe" />
                      </>
                    )}
                    <Row label="Jawline Type" value={d.jawline_type} method="MediaPipe" />
                    <Row label="Jawline Angle" value={d.jawline_angle && `${d.jawline_angle.toFixed(1)}°`} method="MediaPipe" />
                    <Row label="Chin Type" value={d.chin_type} method="MediaPipe" />
                    <Row label="Cheekbone Prominence" value={d.cheekbone_prominence} method="MediaPipe" />
                    <Row label="Cheek Fullness" value={d.cheek_fullness} method="MediaPipe" />
                    <Row label="Forehead Width" value={d.forehead_width} method="MediaPipe" />
                    <Row label="Facial Asymmetry" value={d.facial_asymmetry_score && `${(d.facial_asymmetry_score * 100).toFixed(1)}%`} method="MediaPipe" />

                    {/* ============ HAIR ============ */}
                    <SectionLabel>Hair</SectionLabel>
                    <Row label="Hair Length" value={d.hair_length} method="SegFormer" />
                    <Row label="Hair Present" value={d.hair_present} method="SegFormer" />
                    <Row label="Has Bangs" value={d.has_bangs} method="CLIP" />
                    <Row label="Is Bald" value={d.is_bald} method="CLIP" />
                    <Row label="Receding Hairline" value={d.receding_hairline} method="CLIP" />
                    <Row label="Hair Texture (Landmark)" value={d.hair_texture_celeba} method="CLIP" />
                    {d.hair_texture && d.hair_texture !== d.hair_texture_celeba && (
                      <Row label="Hair Texture (Pixel)" value={d.hair_texture} method="ColorAnalyzer" />
                    )}
                    <Row label="Hair Color (CLIP)" value={d.hair_color_celeba} method="CLIP" />
                    {d.hair_color_scores && Object.keys(d.hair_color_scores).length > 0 && (
                      <Row label="Hair Color Scores" value={Object.entries(d.hair_color_scores).map(([k, v]) => `${k}: ${(v * 100).toFixed(1)}%`).join(', ')} method="CLIP" />
                    )}
                    {d.hair_color && (
                      <>
                        <Row label="Hair Color (Pixel)" value={d.hair_color.name} method="ColorAnalyzer" />
                        <Row label="Hair Hex" value={d.hair_color.hex} method="ColorAnalyzer" />
                      </>
                    )}

                    {/* ============ EYES ============ */}
                    <SectionLabel>Eyes</SectionLabel>
                    <Row label="Eye Shape" value={d.eye_shape} method="MediaPipe" />
                    <Row label="Eye Depth" value={d.eye_depth} method="MediaPipe" />
                    <Row label="Eye Spacing" value={d.eye_spacing} method="MediaPipe" />
                    <Row label="Eye Size" value={d.eye_size} method="MediaPipe" />
                    <Row label="Eyes Open" value={d.eyes_open} method="MediaPipe" />
                    <Row label="Narrow Eyes" value={d.narrow_eyes} method="CLIP" />
                    <Row label="Eye Color" value={d.eye_color} method="ColorAnalyzer" />

                    {/* ============ EYEBROWS ============ */}
                    <SectionLabel>Eyebrows</SectionLabel>
                    <Row label="Eyebrow Shape" value={d.eyebrow_shape} method="MediaPipe" />
                    <Row label="Eyebrow Arch Height" value={d.eyebrow_arch_height} method="MediaPipe" />
                    <Row label="Eyebrow Thickness" value={d.eyebrow_thickness} method="MediaPipe" />
                    <Row label="Possible Unibrow" value={d.possible_unibrow} method="MediaPipe" />
                    <Row label="Arched Eyebrows (CLIP)" value={d.arched_eyebrows} method="CLIP" />
                    <Row label="Bushy Eyebrows (CLIP)" value={d.bushy_eyebrows} method="CLIP" />

                    {/* ============ NOSE ============ */}
                    <SectionLabel>Nose</SectionLabel>
                    <Row label="Nose Shape" value={d.nose_shape} method="MediaPipe" />
                    <Row label="Nose Bridge" value={d.nose_bridge} method="MediaPipe" />
                    <Row label="Nose Tip Shape" value={d.nose_tip_shape} method="MediaPipe" />
                    <Row label="Nostril Width" value={d.nostril_width} method="MediaPipe" />
                    <Row label="Big Nose" value={d.big_nose} method="CLIP" />
                    <Row label="Pointy Nose" value={d.pointy_nose} method="CLIP" />

                    {/* ============ LIPS & MOUTH ============ */}
                    <SectionLabel>Lips & Mouth</SectionLabel>
                    <Row label="Lip Fullness" value={d.lip_fullness} method="MediaPipe" />
                    <Row label="Lip Balance" value={d.lip_balance} method="MediaPipe" />
                    <Row label="Mouth Width" value={d.mouth_width} method="MediaPipe" />
                    <Row label="Cupid's Bow" value={d.cupids_bow} method="MediaPipe" />
                    <Row label="Smile Asymmetry" value={d.smile_asymmetry && `${(d.smile_asymmetry * 100).toFixed(1)}%`} method="MediaPipe" />
                    <Row label="Possible Dimples" value={d.possible_dimples} method="MediaPipe" />
                    <Row label="Smiling" value={d.smiling_celeba} method="CLIP" />
                    <Row label="Mouth Open" value={d.mouth_open} method="CLIP" />
                    <Row label="Wearing Lipstick" value={d.wearing_lipstick} method="CLIP" />
                    <Row label="Big Lips" value={d.big_lips} method="CLIP" />
                    {d.lip_color && (
                      <>
                        <Row label="Lip Color Shade" value={d.lip_color.shade} method="ColorAnalyzer" />
                        <Row label="Lip Hex" value={d.lip_color.hex} method="ColorAnalyzer" />
                      </>
                    )}

                    {/* ============ SKIN ============ */}
                    <SectionLabel>Skin</SectionLabel>
                    {d.skin_tone && (
                      <>
                        <Row label="Skin Tone (Fitzpatrick)" value={d.skin_tone.fitzpatrick} method="ColorAnalyzer" />
                        <Row label="Skin Lightness (L*)" value={d.skin_tone.lab_lightness && `${d.skin_tone.lab_lightness.toFixed(1)}`} method="ColorAnalyzer" />
                        <Row label="Skin A* (Red-Green)" value={d.skin_tone.lab_a && `${d.skin_tone.lab_a.toFixed(1)}`} method="ColorAnalyzer" />
                        <Row label="Skin B* (Yellow-Blue)" value={d.skin_tone.lab_b && `${d.skin_tone.lab_b.toFixed(1)}`} method="ColorAnalyzer" />
                      </>
                    )}
                    <Row label="Skin Undertone" value={d.skin_undertone} method="ColorAnalyzer" />
                    <Row label="Wrinkle Level" value={d.wrinkle_level} method="SegFormer" />
                    <Row label="Skin Texture Score" value={d.skin_texture_score && `${d.skin_texture_score.toFixed(2)}`} method="SegFormer" />
                    <Row label="Skin Uniformity" value={d.skin_uniformity && `${d.skin_uniformity.toFixed(2)}`} method="SegFormer" />
                    <Row label="Freckles or Moles" value={d.freckles_or_moles} method="SegFormer" />
                    <Row label="Pale Skin" value={d.pale_skin} method="CLIP" />
                    <Row label="Rosy Cheeks" value={d.rosy_cheeks} method="CLIP" />
                    <Row label="Bags Under Eyes" value={d.bags_under_eyes} method="CLIP" />
                    <Row label="Chubby Face" value={d.chubby} method="CLIP" />
                    <Row label="Double Chin" value={d.double_chin} method="CLIP" />
                    <Row label="High Cheekbones" value={d.high_cheekbones} method="CLIP" />
                    <Row label="Oval Face" value={d.oval_face_celeba} method="CLIP" />
                    {d.skin_tone?.hex_color && (
                      <Row label="Skin Hex Color" value={d.skin_tone.hex_color} method="ColorAnalyzer" />
                    )}

                    {/* ============ ACCESSORIES ============ */}
                    <SectionLabel>Accessories</SectionLabel>
                    <Row label="Wearing Glasses" value={d.wearing_glasses || d.glasses_detected} method={d.glasses_detected ? 'SegFormer' : 'CLIP'} />
                    <Row label="Wearing Hat" value={d.wearing_hat || d.hat_detected} method={d.hat_detected ? 'SegFormer' : 'CLIP'} />
                    <Row label="Wearing Earrings" value={d.wearing_earrings} method="CLIP" />
                    <Row label="Wearing Necklace" value={d.wearing_necklace} method="CLIP" />
                    <Row label="Wearing Necktie" value={d.wearing_necktie} method="CLIP" />
                    <Row label="Heavy Makeup" value={d.heavy_makeup} method="CLIP" />

                    {/* ============ MISCELLANEOUS (EXPANDABLE) ============ */}
                    <ExpandableSection title="Miscellaneous Details">
                      <>
                        <Row label="Attractive" value={d.attractive} method="CLIP" />
                        <Row label="Young" value={d.young} method="CLIP" />
                        <Row label="Has Beard" value={d.has_beard} method="CLIP" />
                        <Row label="Mustache" value={d.mustache} method="CLIP" />
                        <Row label="Goatee" value={d.goatee} method="CLIP" />
                        <Row label="Sideburns" value={d.sideburns} method="CLIP" />
                        {d.facial_hair && (
                          <>
                            <Row label="Facial Hair - Full Beard" value={d.facial_hair.full_beard} method="CLIP" />
                            <Row label="Facial Hair - Goatee" value={d.facial_hair.goatee} method="CLIP" />
                            <Row label="Facial Hair - Mustache" value={d.facial_hair.mustache} method="CLIP" />
                            <Row label="Facial Hair - Sideburns" value={d.facial_hair.sideburns} method="CLIP" />
                            <Row label="Facial Hair - 5 O'Clock Shadow" value={d.facial_hair['5_o_clock_shadow']} method="CLIP" />
                          </>
                        )}
                        {d._celeba_raw && Object.keys(d._celeba_raw).length > 0 && (
                          <Row label="CelebA Raw Scores" value={JSON.stringify(d._celeba_raw, null, 2)} method="CLIP" />
                        )}
                      </>
                    </ExpandableSection>

                    {/* ============ ANALYSIS MODELS LEGEND ============ */}
                    <SectionLabel>Analysis Method Details</SectionLabel>
                    <Row label="MediaPipe" value="478 3D landmarks + 52 blendshapes (Google)" />
                    <Row label="CLIP" value="OpenAI zero-shot attribute classification (ViT-B/32)" />
                    <Row label="FairFace" value="Gender & age classification (ViT, 93.4% / 59% accuracy)" />
                    <Row label="Ethnicity_Test_v003" value="5-class ethnicity classification (ViT, 79.6% accuracy)" />
                    <Row label="SegFormer" value="Human parsing segmentation (SegFormer-B5, mIoU 0.626)" />
                    <Row label="HSEmotion" value="8-class emotion recognition (EfficientNet-B0)" />
                    <Row label="ColorAnalyzer" value="Pixel-level LAB/HSV analysis (no AI model)" />
                  </>
                );
              } catch (error) {
                console.error('Error rendering facial data:', error);
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
