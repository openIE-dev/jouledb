#!/usr/bin/env bash
# Embedded mode: create a DB, define a schema, insert + query
set -e

jouledb init mydata

jouledb exec mydata \
  "CREATE TABLE notes(id u64 pk, body text)"

jouledb exec mydata \
  "INSERT INTO notes VALUES (1, 'hello'), (2, 'world')"

jouledb exec mydata \
  "SELECT * FROM notes" --pretty
