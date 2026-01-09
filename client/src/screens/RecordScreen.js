import React, { useState, useRef } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Image, TextInput } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';

export default function RecordScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [capturedPhoto, setCapturedPhoto] = useState(null);
  const [photoText, setPhotoText] = useState('');
  const cameraRef = useRef(null);

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

  const handleSubmit = () => {
    // Submit functionality will be implemented later
  };

  if (capturedPhoto) {
    return (
      <View style={styles.container}>
        <View style={styles.previewContainer}>
          <Image source={{ uri: capturedPhoto }} style={styles.previewImage} />
          <View style={styles.inputContainer}>
            <TextInput
              style={styles.textInput}
              placeholder="Add a description..."
              value={photoText}
              onChangeText={setPhotoText}
              multiline
            />
            <View style={styles.buttonRow}>
              <TouchableOpacity style={[styles.button, styles.retakeButton]} onPress={handleRetake}>
                <Text style={styles.buttonText}>Retake</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[styles.button, styles.submitButton]} onPress={handleSubmit}>
                <Text style={styles.buttonText}>Submit</Text>
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
  buttonText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#fff',
  },
});
