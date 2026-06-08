#!/usr/bin/env bash
# Insert hypervector embeddings and query by similarity
set -e

jouledb init mydata

# Schema: items table with an HDC vector column
jouledb exec mydata \
  "CREATE TABLE items(id u64 pk, label text, vec hdc[2048])"

# Insert (vector generation from text uses the built-in hash-based encoder)
jouledb exec mydata \
  "INSERT INTO items VALUES
     (1, 'cat',    hdc_encode('cat')),
     (2, 'kitten', hdc_encode('kitten')),
     (3, 'rocket', hdc_encode('rocket'))"

# Similarity query: find items closest to 'cat'
jouledb exec mydata \
  "SELECT id, label, hdc_sim(vec, hdc_encode('cat')) as sim
   FROM items
   ORDER BY sim DESC" --pretty
