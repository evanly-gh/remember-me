import { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Image,
  TextInput,
  Alert,
  ActivityIndicator,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  BackHandler,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system';
import * as Location from 'expo-location';
import { Ionicons } from '@expo/vector-icons';
import { supabase } from '../lib/supabase';
import { useAuth } from '../context/AuthContext';
import { useTheme } from '../context/ThemeContext';
import { generateEmbedding, buildSearchableText } from '../lib/embeddings';
import { findSimilarFaces } from '../lib/face_matching';

const DETAIL_FIELDS = [
  { key: 'title', label: 'Relation', placeholder: 'e.g., Friend, Colleague', icon: 'person-outline' },
  { key: 'event', label: 'Occasion', placeholder: 'e.g., Conference, Party', icon: 'calendar-outline' },
  { key: 'location', label: 'Location', placeholder: 'City, State', icon: 'location-outline' },
  { key: 'date', label: 'Date', placeholder: 'YYYY-MM-DD', icon: 'time-outline' },
];

export default function RecordScreen({ navigation }) {
  const [permission, requestPermission] = useCameraPermissions();
  const [capturedPhoto, setCapturedPhoto] = useState(null);
  const [showCamera, setShowCamera] = useState(false);
  const [name, setName] = useState('');
  const [phone, setPhone] = useState('');
  const [title, setTitle] = useState('');
  const [event, setEvent] = useState('');
  const [location, setLocation] = useState('');
  const [date, setDate] = useState('');
  const [notes, setNotes] = useState('');
  const [loading, setLoading] = useState(false);
  const [cameraFacing, setCameraFacing] = useState('back');
  const [expandedChip, setExpandedChip] = useState(null);
  const [photoBase64, setPhotoBase64] = useState(null);
  // Discrete zoom level (UI multiplier). The expo-camera `zoom` prop
  // takes a normalised 0-1 value, not a literal multiplier — we map
  // these UI pills to (0, 0.25, 0.5, 0.75) so 1×/2×/3×/5× feel
  // roughly proportional. Devices vary in their max optical zoom so
  // the upper end is an approximation, not a calibrated zoom factor.
  const [zoomLevel, setZoomLevel] = useState(1);
  const cameraRef = useRef(null);
  const chipInputRef = useRef(null);
  const { user } = useAuth();
  const { colors } = useTheme();

  const detailValues = { title, event, location, date };
  const detailSetters = {
    title: setTitle,
    event: setEvent,
    location: setLocation,
    date: setDate,
  };

  // Intercept hardware/gesture back when camera is open
  useEffect(() => {
    if (!showCamera) return;
    const handler = BackHandler.addEventListener('hardwareBackPress', () => {
      setShowCamera(false);
      return true;
    });
    return () => handler.remove();
  }, [showCamera]);

  useEffect(() => {
    setDate(new Date().toISOString().split('T')[0]);
  }, []);

  useEffect(() => {
    getLocation();
  }, []);

  // Auto-focus when chip expands
  useEffect(() => {
    if (expandedChip && chipInputRef.current) {
      chipInputRef.current.focus();
    }
  }, [expandedChip]);

  const getLocation = async () => {
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') return;

      const currentLocation = await Location.getCurrentPositionAsync();
      const [locationData] = await Location.reverseGeocodeAsync({
        latitude: currentLocation.coords.latitude,
        longitude: currentLocation.coords.longitude,
      });

      const address = `${locationData.city || ''}, ${locationData.region || ''}`.replace(/^, |, $/g, '');
      setLocation(address || '');
    } catch (error) {
      console.log('Could not fetch location:', error.message);
    }
  };

  // Tapping the photo placeholder goes straight to the camera.
  // The "choose from library" path lives on a small paperclip button
  // next to the photo circle instead of in a popup, so it's one tap
  // not two.
  const handlePhotoPress = () => openCamera();

  const openCamera = async () => {
    if (!permission?.granted) {
      const result = await requestPermission();
      if (!result.granted) {
        Alert.alert('Permission Required', 'Camera access is needed to take photos.');
        return;
      }
    }
    setZoomLevel(1);   // reset zoom each time the camera opens
    setShowCamera(true);
  };

  const pickFromLibrary = async () => {
    try {
      // Recent expo-image-picker versions deprecated `MediaTypeOptions`
      // in favour of an array-of-strings shorthand. Use the new shape
      // and fall back to the old one for older SDKs.
      const mediaTypes = ImagePicker.MediaType
        ? ['images']
        : ImagePicker.MediaTypeOptions?.Images;

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes,
        base64: true,
        quality: 0.8,
        allowsEditing: false,
      });

      if (result.canceled || !result.assets?.length) return;

      const asset = result.assets[0];
      let base64 = asset.base64;

      // Some platforms return assets without base64 even when
      // requested (notably Android URIs that point at content://).
      // Fall back to reading the file ourselves.
      if (!base64 && asset.uri) {
        try {
          base64 = await FileSystem.readAsStringAsync(asset.uri, {
            encoding: FileSystem.EncodingType.Base64,
          });
        } catch (readError) {
          console.warn('Failed to read base64 from picked image:', readError.message);
        }
      }

      setCapturedPhoto(asset.uri);
      setPhotoBase64(base64 || null);
    } catch (error) {
      console.error('Image picker failed:', error);
      Alert.alert('Error', 'Could not load that photo. Try a different one.');
    }
  };

  const takePicture = async () => {
    if (cameraRef.current) {
      const photo = await cameraRef.current.takePictureAsync({ base64: true });
      setCapturedPhoto(photo.uri);
      setPhotoBase64(photo.base64);
      setShowCamera(false);
    }
  };

  // Discrete zoom levels users can pick from. The internal expo-camera
  // `zoom` prop is 0..1; we map the UI pills to a non-linear scale so
  // each step actually looks different on-device. (1× = no zoom.)
  const ZOOM_LEVELS = [
    { label: '1×', value: 0.0 },
    { label: '2×', value: 0.25 },
    { label: '3×', value: 0.5 },
    { label: '5×', value: 0.75 },
  ];
  const currentZoom = ZOOM_LEVELS.find((z) => z.label === `${zoomLevel}×`)?.value ?? 0;

  const handleSave = async () => {
    if (!user) {
      Alert.alert('Error', 'You must be logged in');
      return;
    }

    if (!name.trim()) {
      Alert.alert('Name Required', 'Please enter a name for this contact.');
      return;
    }

    setLoading(true);

    try {
      let photoUrl = null;

      if (capturedPhoto) {
        const response = await fetch(capturedPhoto);
        const arrayBuffer = await response.arrayBuffer();
        const blob = new Uint8Array(arrayBuffer);

        const timestamp = Date.now();
        const fileName = `${user.id}/${timestamp}.jpg`;

        const { error: uploadError } = await supabase.storage
          .from('photos')
          .upload(fileName, blob, { contentType: 'image/jpeg' });

        if (uploadError) throw uploadError;

        const { data: publicUrlData } = supabase.storage
          .from('photos')
          .getPublicUrl(fileName);

        photoUrl = publicUrlData.publicUrl;
      }

      const { data: insertedData, error: insertError } = await supabase
        .from('people')
        .insert({
          user_id: user.id,
          name: name.trim(),
          photo_url: photoUrl,
          phone: phone.trim() || null,
          title: title.trim() || null,
          event: event.trim() || null,
          location: location.trim() || null,
          date: date || null,
          notes: notes.trim() || '',
        })
        .select('id')
        .single();

      if (insertError) throw insertError;

      // Fire off face analysis asynchronously after the contact is saved
      if (photoBase64 && process.env.EXPO_PUBLIC_FACE_ANALYSIS_URL && insertedData?.id) {
        const recordId = insertedData.id;
        const imageData = photoBase64;
        const contactData = {
          name: name.trim(),
          title: title.trim(),
          event: event.trim(),
          location: location.trim(),
          notes: notes.trim(),
        };

        fetch(process.env.EXPO_PUBLIC_FACE_ANALYSIS_URL, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'ngrok-skip-browser-warning': 'true',
          },
          body: JSON.stringify({ image: imageData }),
        })
          .then((res) => res.ok ? res.json() : null)
          .then(async (result) => {
            if (!result) return;

            // 1. Persist the full facial_details JSON plus the ArcFace
            //    face_embedding column (used for cross-photo identity
            //    matching). The embedding lives in its own pgvector
            //    column so we can run nearest-neighbor queries on it.
            const faceEmbedding = result?.data?.face_embedding || null;
            const update = { facial_details: result };
            if (Array.isArray(faceEmbedding) && faceEmbedding.length === 512) {
              update.face_embedding = faceEmbedding;
            }
            await supabase.from('people').update(update).eq('id', recordId);

            // 2. Generate and store the text embedding for semantic search.
            try {
              const searchableText = buildSearchableText({
                ...contactData,
                facial_details: result
              });
              const embedding = await generateEmbedding(searchableText);
              if (embedding) {
                await supabase
                  .from('people')
                  .update({ embedding, searchable_text: searchableText })
                  .eq('id', recordId);
              }
            } catch (embeddingError) {
              console.warn('Embedding generation failed:', embeddingError.message);
            }

            // 3. Face matching: see if any existing contact looks like
            //    this person. Excludes the row we just created. If we
            //    find a confident match with a *different* name, prompt
            //    the user to merge into that profile.
            if (Array.isArray(faceEmbedding) && faceEmbedding.length === 512) {
              try {
                const matches = await findSimilarFaces(faceEmbedding, user.id);
                const candidate = (matches || []).find(
                  (m) => m.id !== recordId && m.name !== contactData.name
                );
                if (candidate) {
                  Alert.alert(
                    'Look familiar?',
                    `This face looks like ${candidate.name} ` +
                    `(${(candidate.similarity * 100).toFixed(1)}% match). ` +
                    `Add this photo to that profile?`,
                    [
                      { text: 'No, keep separate', style: 'cancel' },
                      {
                        text: `Add to ${candidate.name}`,
                        onPress: async () => {
                          // Profiles are grouped by name in this app, so
                          // renaming the new row is the merge operation.
                          await supabase
                            .from('people')
                            .update({ name: candidate.name })
                            .eq('id', recordId);
                        },
                      },
                    ]
                  );
                }
              } catch (matchError) {
                console.warn('Face matching failed:', matchError.message || matchError);
              }
            }
          })
          .catch((err) => {
            console.warn('Face analysis failed:', err.message || err);
          });
      }

      Alert.alert('Saved', `${name.trim()} has been added to your contacts.`, [
        {
          text: 'OK',
          onPress: () => navigation.navigate('Contacts'),
        },
      ]);
      setCapturedPhoto(null);
      setPhotoBase64(null);
      setName('');
      setPhone('');
      setTitle('');
      setEvent('');
      setLocation('');
      setNotes('');
      setDate(new Date().toISOString().split('T')[0]);
      setExpandedChip(null);
      getLocation();
    } catch (error) {
      Alert.alert('Error', error.message || 'Failed to save contact');
    } finally {
      setLoading(false);
    }
  };

  const toggleChip = (key) => {
    setExpandedChip(prev => prev === key ? null : key);
  };

  // Camera overlay
  if (showCamera) {
    return (
      <View style={styles.cameraContainer}>
        <CameraView
          style={StyleSheet.absoluteFill}
          facing={cameraFacing}
          zoom={currentZoom}
          ref={cameraRef}
        />
        <SafeAreaView style={styles.cameraOverlay} pointerEvents="box-none">
          <View style={styles.cameraTopRow}>
            <TouchableOpacity
              style={styles.cameraCloseButton}
              onPress={() => setShowCamera(false)}
            >
              <Ionicons name="chevron-back" size={28} color="#fff" />
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.cameraFlipButton}
              onPress={() => setCameraFacing(f => f === 'front' ? 'back' : 'front')}
            >
              <Ionicons name="camera-reverse-outline" size={26} color="#fff" />
            </TouchableOpacity>
          </View>

          {/* Bottom controls cluster: zoom pills directly above the
              shutter so the user's thumb is already in the right zone
              when they switch between zooming and tapping. */}
          <View style={styles.cameraBottom}>
            <View style={styles.zoomPillRow}>
              {ZOOM_LEVELS.map((z) => {
                const label = z.label;
                const numericLabel = parseInt(label, 10);
                const active = zoomLevel === numericLabel;
                return (
                  <TouchableOpacity
                    key={label}
                    onPress={() => setZoomLevel(numericLabel)}
                    style={[
                      styles.zoomPill,
                      active && styles.zoomPillActive,
                    ]}
                  >
                    <Text
                      style={[
                        styles.zoomPillText,
                        active && styles.zoomPillTextActive,
                      ]}
                    >
                      {label}
                    </Text>
                  </TouchableOpacity>
                );
              })}
            </View>
            <TouchableOpacity style={styles.shutterButton} onPress={takePicture}>
              <View style={styles.shutterButtonInner} />
            </TouchableOpacity>
          </View>
        </SafeAreaView>
      </View>
    );
  }

  const expandedField = expandedChip && DETAIL_FIELDS.find(f => f.key === expandedChip);

  return (
    <SafeAreaView style={[styles.safeArea, { backgroundColor: colors.background }]}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.flex}
      >
        <ScrollView
          style={[styles.container, { backgroundColor: colors.background }]}
          contentContainerStyle={styles.scrollContent}
          keyboardShouldPersistTaps="handled"
        >
          <Text style={[styles.screenTitle, { color: colors.text }]}>New Contact</Text>

          {/* Photo placeholder + secondary "attach from library" button.
              The photo circle is the primary action (opens camera). The
              paperclip on the right is the secondary path for users who
              want to use an existing photo. */}
          <View style={styles.photoSection}>
            <TouchableOpacity
              style={[styles.photoPlaceholder, { backgroundColor: colors.accentLight }]}
              onPress={handlePhotoPress}
              activeOpacity={0.7}
            >
              {capturedPhoto ? (
                <>
                  <Image source={{ uri: capturedPhoto }} style={styles.photoImage} />
                  <TouchableOpacity
                    style={styles.retakeButton}
                    onPress={handlePhotoPress}
                  >
                    <Ionicons name="camera" size={16} color="#fff" />
                  </TouchableOpacity>
                </>
              ) : (
                <Ionicons name="camera" size={36} color={colors.accent} />
              )}
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.attachButton, { backgroundColor: colors.accentLight }]}
              onPress={pickFromLibrary}
              activeOpacity={0.7}
              accessibilityLabel="Attach photo from library"
            >
              <Ionicons name="attach" size={24} color={colors.accent} />
            </TouchableOpacity>
          </View>

          {/* Primary fields */}
          <View style={styles.formSection}>
            <TextInput
              style={[styles.nameInput, { color: colors.text }]}
              placeholder="Name"
              placeholderTextColor={colors.placeholder}
              value={name}
              onChangeText={setName}
              editable={!loading}
            />
            <View style={[styles.separator, { backgroundColor: colors.border }]} />
            <TextInput
              style={[styles.fieldInput, { color: colors.text }]}
              placeholder="Phone"
              placeholderTextColor={colors.placeholder}
              value={phone}
              onChangeText={setPhone}
              keyboardType="phone-pad"
              editable={!loading}
            />
          </View>

          {/* Detail chips */}
          <View style={styles.chipsSection}>
            <View style={styles.chipsRow}>
              {DETAIL_FIELDS.map((field) => {
                const value = detailValues[field.key];
                const isFilled = !!value;
                const isExpanded = expandedChip === field.key;
                return (
                  <TouchableOpacity
                    key={field.key}
                    style={[
                      styles.chip,
                      { borderColor: colors.border },
                      isFilled && { backgroundColor: colors.accentLight, borderColor: colors.accentLight },
                      isExpanded && { backgroundColor: colors.accent, borderColor: colors.accent },
                    ]}
                    onPress={() => toggleChip(field.key)}
                    activeOpacity={0.7}
                  >
                    <Ionicons
                      name={field.icon}
                      size={14}
                      color={isExpanded ? '#fff' : isFilled ? colors.accent : colors.textTertiary}
                      style={styles.chipIcon}
                    />
                    <Text
                      style={[
                        styles.chipText,
                        { color: isExpanded ? '#fff' : isFilled ? colors.accent : colors.textTertiary },
                      ]}
                      numberOfLines={1}
                    >
                      {isFilled ? value : field.label}
                    </Text>
                  </TouchableOpacity>
                );
              })}
            </View>

            {/* Expanded chip input */}
            {expandedField && (
              <View style={[styles.chipInputContainer, { backgroundColor: colors.inputBg, borderColor: colors.border }]}>
                <TextInput
                  ref={chipInputRef}
                  style={[styles.chipInput, { color: colors.text }]}
                  placeholder={expandedField.placeholder}
                  placeholderTextColor={colors.placeholder}
                  value={detailValues[expandedField.key]}
                  onChangeText={detailSetters[expandedField.key]}
                  onSubmitEditing={() => setExpandedChip(null)}
                  returnKeyType="done"
                  editable={!loading}
                />
                <TouchableOpacity onPress={() => setExpandedChip(null)} style={styles.chipInputDone}>
                  <Ionicons name="checkmark-circle" size={22} color={colors.accent} />
                </TouchableOpacity>
              </View>
            )}
          </View>

          {/* Notes */}
          <View style={styles.notesSection}>
            <TextInput
              style={[
                styles.notesInput,
                { color: colors.text, backgroundColor: colors.inputBg, borderColor: colors.border },
              ]}
              placeholder="Add notes..."
              placeholderTextColor={colors.placeholder}
              value={notes}
              onChangeText={setNotes}
              multiline
              textAlignVertical="top"
              editable={!loading}
            />
          </View>

          {/* Save button */}
          <TouchableOpacity
            style={[styles.saveButton, loading && styles.saveButtonDisabled]}
            onPress={handleSave}
            disabled={loading}
          >
            {loading ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.saveButtonText}>Save Contact</Text>
            )}
          </TouchableOpacity>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  flex: {
    flex: 1,
  },
  safeArea: {
    flex: 1,
  },
  container: {
    flex: 1,
  },
  scrollContent: {
    paddingBottom: 40,
  },
  screenTitle: {
    fontSize: 28,
    fontWeight: '700',
    paddingHorizontal: 20,
    paddingTop: 10,
    paddingBottom: 20,
  },

  // Photo
  photoSection: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingBottom: 30,
    gap: 12,
  },
  photoPlaceholder: {
    width: 160,
    height: 160,
    borderRadius: 80,
    justifyContent: 'center',
    alignItems: 'center',
    overflow: 'hidden',
  },
  attachButton: {
    width: 48,
    height: 48,
    borderRadius: 24,
    justifyContent: 'center',
    alignItems: 'center',
  },
  photoImage: {
    width: 160,
    height: 160,
    borderRadius: 80,
    resizeMode: 'cover',
  },
  retakeButton: {
    position: 'absolute',
    bottom: 4,
    right: 4,
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },

  // Primary form
  formSection: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  nameInput: {
    fontSize: 20,
    fontWeight: '600',
    paddingVertical: 14,
  },
  fieldInput: {
    fontSize: 16,
    paddingVertical: 14,
  },
  separator: {
    height: 1,
  },

  // Detail chips
  chipsSection: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  chipsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  chip: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
    borderWidth: 1,
  },
  chipIcon: {
    marginRight: 5,
  },
  chipText: {
    fontSize: 13,
    fontWeight: '500',
    maxWidth: 120,
  },
  chipInputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 12,
    borderRadius: 10,
    borderWidth: 1,
    paddingHorizontal: 14,
  },
  chipInput: {
    flex: 1,
    fontSize: 15,
    paddingVertical: 12,
  },
  chipInputDone: {
    padding: 4,
    marginLeft: 8,
  },

  // Notes
  notesSection: {
    paddingHorizontal: 20,
    marginBottom: 24,
  },
  notesInput: {
    fontSize: 15,
    borderRadius: 10,
    borderWidth: 1,
    paddingHorizontal: 14,
    paddingTop: 12,
    paddingBottom: 12,
    minHeight: 100,
  },

  // Save button
  saveButton: {
    marginHorizontal: 20,
    backgroundColor: '#8B5CF6',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  saveButtonDisabled: {
    opacity: 0.6,
  },
  saveButtonText: {
    color: '#fff',
    fontSize: 17,
    fontWeight: '600',
  },

  // Camera
  cameraContainer: {
    flex: 1,
    backgroundColor: '#000',
  },
  camera: {
    flex: 1,
  },
  cameraOverlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'space-between',
  },
  cameraTopRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  cameraCloseButton: {
    padding: 16,
  },
  cameraFlipButton: {
    padding: 16,
  },
  cameraBottom: {
    alignItems: 'center',
    paddingBottom: 40,
  },
  zoomPillRow: {
    flexDirection: 'row',
    backgroundColor: 'rgba(0, 0, 0, 0.45)',
    borderRadius: 24,
    padding: 4,
    marginBottom: 18,
    alignSelf: 'center',
  },
  zoomPill: {
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 20,
    minWidth: 44,
    alignItems: 'center',
  },
  zoomPillActive: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
  },
  zoomPillText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
    fontVariant: ['tabular-nums'],
  },
  zoomPillTextActive: {
    color: '#000',
  },
  shutterButton: {
    width: 70,
    height: 70,
    borderRadius: 35,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 4,
    borderColor: '#fff',
  },
  shutterButtonInner: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: '#fff',
  },
});
