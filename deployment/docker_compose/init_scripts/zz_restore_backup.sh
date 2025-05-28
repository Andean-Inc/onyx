#!/bin/sh
set -e

# Set the password for the andean_admin user
export PGPASSWORD='comparch-18447'

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
until pg_isready -U "${POSTGRES_USER:-postgres}" -q
do
  echo "PostgreSQL is unavailable - sleeping"
  sleep 1
done
echo "PostgreSQL is up - executing restore"

# Restore the database backup
# The -v flag provides verbose output, which can be helpful for debugging.
# The -d andean_db flag specifies the database to restore to.
# The -U andean_admin flag specifies the user to connect as.
# We are assuming the database_backup.dump file is in the same directory (/docker-entrypoint-initdb.d/)
pg_restore -U andean_admin -d andean_db -v /docker-entrypoint-initdb.d/database_backup.dump

# Unset the password
unset PGPASSWORD

echo "Database backup restored successfully."