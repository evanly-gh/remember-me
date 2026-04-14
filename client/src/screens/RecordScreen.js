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
import * as Location from 'expo-location';
import { Ionicons } from '@expo/vector-icons';
import { supabase } from '../lib/supabase';
import { useAuth } from '../context/AuthContext';

export default function RecordScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [capturedPhoto, setCapturedPhoto] = useState(null);
  const [showCamera, setShowCamera] = useState(false);
  const [name, setName] = useState('');
  const [title, setTitle] = useState('');
  const [event, setEvent] = useState('');
  const [location, setLocation] = useState('');
  const [date, setDate] = useState('');
  const [loading, setLoading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [cameraFacing, setCameraFacing] = useState('back');
  const cameraRef = useRef(null);
  const { user } = useAuth();

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

  const handlePhotoPress = async () => {
    if (!permission?.granted) {
      const result = await requestPermission();
      if (!result.granted) {
        Alert.alert('Permission Required', 'Camera access is needed to take photos.');
        return;
      }
    }
    setShowCamera(true);
  };

  const takePicture = async () => {
    if (cameraRef.current) {
      const photo = await cameraRef.current.takePictureAsync({ base64: true });
      setCapturedPhoto(photo.uri);
      setShowCamera(false);
      analyzeFace(photo.base64);
    }
  };

  const analyzeFace = async (base64Image) => {
    if (!process.env.EXPO_PUBLIC_FACE_ANALYSIS_URL) return;

    setIsAnalyzing(true);
    setAnalysis(null);
    try {
      const response = await fetch(process.env.EXPO_PUBLIC_FACE_ANALYSIS_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true',
        },
        body: JSON.stringify({ image: base64Image }),
      });

      if (!response.ok) {
        setAnalysis(null);
        return;
      }

      const result = await response.json();
      setAnalysis(result || null);
    } catch (error) {
      console.warn('Face analysis failed:', error.message || error);
      setAnalysis(null);
    } finally {
      setIsAnalyzing(false);
    }
  };

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

      const { error: insertError } = await supabase
        .from('people')
        .insert({
          user_id: user.id,
          name: name.trim(),
          photo_url: photoUrl,
          title: title.trim() || null,
          event: event.trim() || null,
          location: location.trim() || null,
          date: date || null,
          notes: '',
          facial_details: analysis || null,
        });

      if (insertError) throw insertError;

      Alert.alert('Saved', `${name.trim()} has been added to your contacts.`);
      setCapturedPhoto(null);
      setName('');
      setTitle('');
      setEvent('');
      setLocation('');
      setDate(new Date().toISOString().split('T')[0]);
      setAnalysis(null);
      getLocation();
    } catch (error) {
      Alert.alert('Error', error.message || 'Failed to save contact');
    } finally {
      setLoading(false);
    }
  };

  // Camera overlay
  if (showCamera) {
    return (
      <View style={styles.cameraContainer}>
        <CameraView style={styles.camera} facing={cameraFacing} ref={cameraRef}>
          <SafeAreaView style={styles.cameraOverlay}>
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
            <View style={styles.cameraBottom}>
              <TouchableOpacity style={styles.shutterButton} onPress={takePicture}>
                <View style={styles.shutterButtonInner} />
              </TouchableOpacity>
            </View>
          </SafeAreaView>
        </CameraView>
      </View>
    );
  }

  return (
    <SafeAreaView style={styles.safeArea}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.flex}
      >
        <ScrollView
          style={styles.container}
          contentContainerStyle={styles.scrollContent}
          keyboardShouldPersistTaps="handled"
        >
          <Text style={styles.screenTitle}>New Contact</Text>

          {/* Photo placeholder */}
          <View style={styles.photoSection}>
            <TouchableOpacity
              style={styles.photoPlaceholder}
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
                <Ionicons name="camera" size={36} color="#8B5CF6" />
              )}
            </TouchableOpacity>
            {isAnalyzing && (
              <Text style={styles.analyzingText}>Analyzing...</Text>
            )}
          </View>

          {/* Form fields */}
          <View style={styles.formSection}>
            <TextInput
              style={styles.nameInput}
              placeholder="Name"
              placeholderTextColor="#9CA3AF"
              value={name}
              onChangeText={setName}
              editable={!loading}
            />
            <View style={styles.separator} />

            <TextInput
              style={styles.fieldInput}
              placeholder="Title (e.g., Friend, Colleague)"
              placeholderTextColor="#9CA3AF"
              value={title}
              onChangeText={setTitle}
              editable={!loading}
            />
            <View style={styles.separator} />

            <TextInput
              style={styles.fieldInput}
              placeholder="Event"
              placeholderTextColor="#9CA3AF"
              value={event}
              onChangeText={setEvent}
              editable={!loading}
            />
            <View style={styles.separator} />

            <TextInput
              style={styles.fieldInput}
              placeholder="Location"
              placeholderTextColor="#9CA3AF"
              value={location}
              onChangeText={setLocation}
              editable={!loading}
            />
            <View style={styles.separator} />

            <TextInput
              style={styles.fieldInput}
              placeholder="Date"
              placeholderTextColor="#9CA3AF"
              value={date}
              onChangeText={setDate}
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
    backgroundColor: '#fff',
  },
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  scrollContent: {
    paddingBottom: 40,
  },
  screenTitle: {
    fontSize: 28,
    fontWeight: '700',
    color: '#1F2937',
    paddingHorizontal: 20,
    paddingTop: 10,
    paddingBottom: 20,
  },

  // Photo
  photoSection: {
    alignItems: 'center',
    paddingBottom: 30,
  },
  photoPlaceholder: {
    width: 160,
    height: 160,
    borderRadius: 80,
    backgroundColor: '#EDE9FE',
    justifyContent: 'center',
    alignItems: 'center',
    overflow: 'hidden',
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
  analyzingText: {
    marginTop: 8,
    fontSize: 12,
    color: '#9CA3AF',
  },

  // Form
  formSection: {
    paddingHorizontal: 20,
    marginBottom: 30,
  },
  nameInput: {
    fontSize: 20,
    fontWeight: '600',
    color: '#1F2937',
    paddingVertical: 14,
  },
  fieldInput: {
    fontSize: 16,
    color: '#1F2937',
    paddingVertical: 14,
  },
  separator: {
    height: 1,
    backgroundColor: '#F3F4F6',
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
    flex: 1,
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
