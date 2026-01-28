import React, { useState, useRef, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Image, TextInput, Alert, ActivityIndicator, ScrollView } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
// Added: Location service for reverse geocoding to auto-populate location
import * as Location from 'expo-location';
import { supabase } from '../lib/supabase';
import { useAuth } from '../context/AuthContext';

export default function RecordScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [capturedPhoto, setCapturedPhoto] = useState(null);
  const [photoText, setPhotoText] = useState('');
  // Added: State for event field (editable user input)
  const [event, setEvent] = useState('');
  // Added: State for location field (auto-populated via reverse geocoding)
  const [location, setLocation] = useState('');
  // Added: State for date field (auto-populated with today's date)
  const [date, setDate] = useState('');
  const [loading, setLoading] = useState(false);
  // Analysis state: whether remote face analysis is running and its result
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const cameraRef = useRef(null);
  const { user } = useAuth();

  // Added: Auto-populate date field with today's date when component mounts
  useEffect(() => {
    const today = new Date().toISOString().split('T')[0];
    setDate(today);
  }, []);

  // Added: Auto-populate location field using reverse geocoding when component mounts
  useEffect(() => {
    getLocation();
  }, []);

  // Added: Function to get current device location and reverse-geocode to city/region
  const getLocation = async () => {
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        console.log('Location permission denied');
        return;
      }

      const currentLocation = await Location.getCurrentPositionAsync();
      const [locationData] = await Location.reverseGeocodeAsync({
        latitude: currentLocation.coords.latitude,
        longitude: currentLocation.coords.longitude,
      });

      const address = `${locationData.city || ''}, ${locationData.region || ''}`.replace(/^, |, $/g, '');
      setLocation(address || 'Location unavailable');
    } catch (error) {
      console.log('Could not fetch location:', error.message);
      setLocation('');
    }
  }

  if (!permission) {
    return <View style={styles.container} />;
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.message}>We need your permission to show the camera</Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // Take a photo from the camera. set captured URI and trigger optional analysis.
  const takePicture = async () => {
    if (cameraRef.current) {
      const photo = await cameraRef.current.takePictureAsync({
        base64: true,
      });
      setCapturedPhoto(photo.uri);
      // Run face analysis asynchronously (won't block UI)
      analyzeFace(photo.base64);
    }
  };

  // Added: Analyze face using remote API (optional - won't block app if unavailable)
  const analyzeFace = async (base64Image) => {
    // Skip analysis if URL not configured
    if (!process.env.EXPO_PUBLIC_FACE_ANALYSIS_URL) {
      console.log('Face analysis URL not configured, skipping analysis');
      return;
    }

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
        console.warn(`Face analysis returned status ${response.status}`);
        setAnalysis(null);
        return;
      }

      // Expect JSON result from analysis service
      const result = await response.json();
      setAnalysis(result || null);
    } catch (error) {
      // Log error but don't crash app - face analysis is optional
      console.warn('Face analysis failed (app will continue):', error.message || error);
      setAnalysis(null);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Added: Reset all form fields when user wants to retake photo
  const handleRetake = () => {
    setCapturedPhoto(null);
    setPhotoText('');
    // Added: Reset event and location fields
    setEvent('');
    setLocation('');
    // Added: Reset date to today
    setDate(new Date().toISOString().split('T')[0]);
  };

  // Added: Enhanced submit handler that saves photo and all metadata to Supabase
  const handleSubmit = async () => {
    if (!user) {
      Alert.alert('Error', 'You must be logged in to submit');
      return;
    }

    if (!photoText.trim()) {
      Alert.alert('Error', 'Please add a name for this person');
      return;
    }

    setLoading(true);

    try {
      // 1. Read the photo file and convert to blob
      const response = await fetch(capturedPhoto);
      const arrayBuffer = await response.arrayBuffer();
      const blob = new Uint8Array(arrayBuffer);

      // 2. Create a unique filename with user ID and timestamp (ensures isolation by user)
      const timestamp = Date.now();
      const fileName = `${user.id}/${timestamp}.jpg`;

      // 3. Upload photo to Supabase Storage bucket with RLS protection
      const { data, error: uploadError } = await supabase.storage
        .from('photos')
        .upload(fileName, blob, {
          contentType: 'image/jpeg',
        });

      if (uploadError) throw uploadError;

      // 4. Get the public URL for the uploaded photo
      const { data: publicUrlData } = supabase.storage
        .from('photos')
        .getPublicUrl(fileName);

      const photoUrl = publicUrlData.publicUrl;

      // 5. Save complete record with all metadata to people table
      // Added: event, location, and date fields are now persisted
      const { error: insertError } = await supabase
        .from('people')
        .insert({
          user_id: user.id,
          name: photoText.trim(),
          photo_url: photoUrl,
          // Added: Store event metadata
          event: event.trim() || null,
          // Added: Store location metadata (auto-populated)
          location: location.trim() || null,
          // Added: Store date metadata (auto-populated)
          date: date || null,
          notes: '',
        });

      if (insertError) throw insertError;

      // 6. Success - clear form and reset fields
      Alert.alert('Success', 'Person added to your list!');
      setCapturedPhoto(null);
      setPhotoText('');
      // Added: Clear all metadata fields
      setEvent('');
      setLocation('');
      setDate(new Date().toISOString().split('T')[0]);
    } catch (error) {
      Alert.alert('Error', error.message || 'Failed to submit');
    } finally {
      setLoading(false);
    }
  };

  // Preview mode after taking a photo: show image, optional analysis overlay, inputs and actions
  if (capturedPhoto) {
    return (
      <View style={styles.container}>
        <View style={styles.previewContainer}>
          {/* Use contain so image is not overly zoomed/cropped */}
          <Image source={{ uri: capturedPhoto }} style={styles.previewImage} />

          {/* Show simple analyzing overlay while remote analysis runs */}
          {isAnalyzing && (
            <View style={styles.analysisOverlay}>
              <Text style={styles.analysisText}>Analyzing face...</Text>
            </View>
          )}

          {/* Display analysis results when available */}
          {analysis && analysis.available && (
            <View style={styles.analysisOverlay}>
              <Text style={styles.analysisText}>
                {analysis.face_count > 0 
                  ? `Detected ${analysis.face_count} face(s)` 
                  : 'No faces detected'}
              </Text>
              {analysis.faces && analysis.faces.map((face, i) => (
                <Text key={i} style={styles.analysisSubtext}>
                  Face {i+1}: {face.primary_emotion} {face.smiling ? '😊' : ''}
                </Text>
              ))}
            </View>
          )}

          {/* Inputs: made smaller and single-line so buttons remain visible and scrollable */}
          <View style={styles.inputContainer}>
            <TextInput
              style={styles.textInput}
              placeholder="Name of person..."
              value={photoText}
              onChangeText={setPhotoText}
              multiline={false}
              editable={!loading}
            />
            <TextInput
              style={styles.textInput}
              placeholder="Event (e.g., Party, Conference)"
              value={event}
              onChangeText={setEvent}
              multiline={false}
              editable={!loading}
            />
            <TextInput
              style={styles.textInput}
              placeholder="Location"
              value={location}
              onChangeText={setLocation}
              multiline={false}
              editable={!loading}
            />
            <TextInput
              style={styles.textInput}
              placeholder="Date (YYYY-MM-DD)"
              value={date}
              onChangeText={setDate}
              multiline={false}
              editable={!loading}
            />
            <View style={styles.buttonRow}>
              <TouchableOpacity
                style={[styles.button, styles.retakeButton]}
                onPress={handleRetake}
                disabled={loading}
              >
                <Text style={styles.buttonText}>Retake</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.button, styles.submitButton, loading && styles.buttonDisabled]}
                onPress={handleSubmit}
                disabled={loading}
              >
                {loading ? (
                  <ActivityIndicator color="#fff" />
                ) : (
                  <Text style={styles.buttonText}>Submit</Text>
                )}
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* cameraWrapper provides a small padding/border so camera preview isn't zoomed/cropped */}
      <View style={styles.cameraWrapper}>
        <CameraView style={styles.cameraInner} facing="back" ref={cameraRef}>
          <View style={styles.buttonContainer}>
            <TouchableOpacity style={styles.shutterButton} onPress={takePicture}>
              <View style={styles.shutterButtonInner} />
            </TouchableOpacity>
          </View>
        </CameraView>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000',
  },
  message: {
    textAlign: 'center',
    paddingBottom: 20,
    color: '#fff',
    fontSize: 16,
  },
  camera: {
    flex: 1,
    width: '100%',
  },
  // Wrapper provides a small margin so camera input isn't edge-to-edge (reduces zoomed-in feeling)
  cameraWrapper: {
    flex: 1,
    width: '100%',
    padding: 8,
    backgroundColor: '#000',
  },
  // Camera inner should fill wrapper and keep aspect
  cameraInner: {
    flex: 1,
    borderRadius: 8,
    overflow: 'hidden',
  },
  buttonContainer: {
    flex: 1,
    justifyContent: 'flex-end',
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
  previewContainer: {
    flex: 1,
    width: '100%',
    backgroundColor: '#fff',
  },
  previewImage: {
    width: '100%',
    height: '60%',
    // Use contain to avoid cropping / zooming in on face
    resizeMode: 'contain',
    backgroundColor: '#000',
  },
  // Make inputs compact and allow scrolling so buttons remain visible
  inputContainer: {
    flex: 1,
    padding: 12,
    justifyContent: 'flex-start',
  },
  // Reduced input height so submit/retake buttons fit on screen
  textInput: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 16,
    minHeight: 44,
    textAlignVertical: 'center',
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 10,
  },
  button: {
    flex: 1,
    paddingVertical: 15,
    borderRadius: 8,
    alignItems: 'center',
  },
  retakeButton: {
    backgroundColor: '#8E8E93',
  },
  submitButton: {
    backgroundColor: '#007AFF',
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  buttonText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#fff',
  },
});
