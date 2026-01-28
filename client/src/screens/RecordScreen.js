import React, { useState, useRef } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Image, TextInput, Alert, ActivityIndicator } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { supabase } from '../lib/supabase';
import { useAuth } from '../context/AuthContext';

export default function RecordScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [capturedPhoto, setCapturedPhoto] = useState(null);
  const [photoText, setPhotoText] = useState('');
  const [loading, setLoading] = useState(false);
  const cameraRef = useRef(null);
  const { user } = useAuth();

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

  const takePicture = async () => {
    if (cameraRef.current) {
      const photo = await cameraRef.current.takePictureAsync();
      setCapturedPhoto(photo.uri);
    }
  };

  const handleRetake = () => {
    setCapturedPhoto(null);
    setPhotoText('');
  };

  const handleSubmit = async () => {
    console.log('User object:', user);
    console.log('User ID:', user?.id);
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
      // 1. Read the photo file
      const response = await fetch(capturedPhoto);
      const arrayBuffer = await response.arrayBuffer();
      const blob = new Uint8Array(arrayBuffer);

      // 2. Create a unique filename with user ID
      const timestamp = Date.now();
      const fileName = `${user.id}/${timestamp}.jpg`;

      // 3. Upload to Supabase Storage bucket
      console.log('Uploading file:', fileName);
      console.log('Blob type:', typeof blob, 'Blob length:', blob.length);
      const { data, error: uploadError } = await supabase.storage
        .from('photos')
        .upload(fileName, blob, {
          contentType: 'image/jpeg',
        });

      console.log('Upload response:', { data, uploadError });
      if (uploadError) {
        console.log('Upload error details:', JSON.stringify(uploadError, null, 2));
        throw uploadError;
      }

      // 4. Get the public URL
      const { data: publicUrlData } = supabase.storage
        .from('photos')
        .getPublicUrl(fileName);

      const photoUrl = publicUrlData.publicUrl;
      console.log('Photo URL:', photoUrl);

      // 5. Save record to people table
      console.log('Attempting to insert with user_id:', user.id);
      const { error: insertError } = await supabase
        .from('people')
        .insert({
          user_id: user.id,
          name: photoText.trim(),
          photo_url: photoUrl,
          notes: '',
        });

      console.log('Insert error:', insertError);
      if (insertError) {
        console.log('Full error details:', JSON.stringify(insertError, null, 2));
        throw insertError;
      }

      // 6. Success!
      Alert.alert('Success', 'Person added to your list!');
      setCapturedPhoto(null);
      setPhotoText('');
    } catch (error) {
      Alert.alert('Error', error.message || 'Failed to submit');
    } finally {
      setLoading(false);
    }
  };

  if (capturedPhoto) {
    return (
      <View style={styles.container}>
        <View style={styles.previewContainer}>
          <Image source={{ uri: capturedPhoto }} style={styles.previewImage} />
          <View style={styles.inputContainer}>
            <TextInput
              style={styles.textInput}
              placeholder="Name of person..."
              value={photoText}
              onChangeText={setPhotoText}
              multiline
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
      <CameraView style={styles.camera} facing="back" ref={cameraRef}>
        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.shutterButton} onPress={takePicture}>
            <View style={styles.shutterButtonInner} />
          </TouchableOpacity>
        </View>
      </CameraView>
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
    resizeMode: 'cover',
    backgroundColor: '#000',
  },
  inputContainer: {
    flex: 1,
    padding: 20,
    justifyContent: 'space-between',
  },
  textInput: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 15,
    fontSize: 16,
    minHeight: 100,
    textAlignVertical: 'top',
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
