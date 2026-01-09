import React, { useState } from 'react';
import { View, TextInput, StyleSheet, Text, ScrollView } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

const FAKE_PEOPLE = [
  {
    id: 1,
    name: 'John Smith',
    description: 'Software engineer at Tech Corp, loves hiking and photography',
  },
  {
    id: 2,
    name: 'John Smith',
    description: 'Software engineer at Tech Corp, loves hiking and photography',
  },
  {
    id: 3,
    name: 'John Smith',
    description: 'Software engineer at Tech Corp, loves hiking and photography',
  },
  {
    id: 4,
    name: 'John Smith',
    description: 'Software engineer at Tech Corp, loves hiking and photography',
  },
  {
    id: 5,
    name: 'John Smith',
    description: 'Software engineer at Tech Corp, loves hiking and photography',
  },
  {
    id: 6,
    name: 'John Smith',
    description: 'Software engineer at Tech Corp, loves hiking and photography',
  },
  {
    id: 7,
    name: 'John Smith',
    description: 'Software engineer at Tech Corp, loves hiking and photography',
  },
  {
    id: 8,
    name: 'John Smith',
    description: 'Software engineer at Tech Corp, loves hiking and photography',
  },
];

export default function LookupScreen() {
  const [searchQuery, setSearchQuery] = useState('');

  return (
    <View style={styles.container}>
      <View style={styles.searchContainer}>
        <TextInput
          style={styles.searchBar}
          placeholder="Search..."
          value={searchQuery}
          onChangeText={setSearchQuery}
        />
      </View>
      <ScrollView style={styles.content}>
        {FAKE_PEOPLE.map((person) => (
          <View key={person.id} style={styles.card}>
            <View style={styles.cardImagePlaceholder}>
              <Ionicons name="person-outline" size={40} color="#999" />
            </View>
            <View style={styles.cardContent}>
              <Text style={styles.cardName}>{person.name}</Text>
              <Text style={styles.cardDescription}>{person.description}</Text>
            </View>
          </View>
        ))}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  searchContainer: {
    padding: 10,
    backgroundColor: '#f5f5f5',
  },
  searchBar: {
    height: 40,
    backgroundColor: '#fff',
    borderRadius: 8,
    paddingHorizontal: 15,
    fontSize: 16,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  content: {
    flex: 1,
  },
  card: {
    flexDirection: 'row',
    padding: 20,
    marginHorizontal: 10,
    marginTop: 10,
    minHeight: 100,
    backgroundColor: '#fff',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e0e0e0',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  cardImagePlaceholder: {
    width: 70,
    height: 70,
    borderRadius: 35,
    backgroundColor: '#f0f0f0',
    marginRight: 15,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#ddd',
  },
  cardContent: {
    flex: 1,
    justifyContent: 'center',
  },
  cardName: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 5,
    color: '#333',
  },
  cardDescription: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
  },
});
