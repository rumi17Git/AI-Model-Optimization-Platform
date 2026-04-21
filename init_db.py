#!/usr/bin/env python3
"""
Initialize the database with default user for simple mode
"""
import hashlib
from modules.database import get_db_manager

def init_database():
    """Initialize database with default user for no-auth mode"""
    print("Initializing database...")
    
    db = get_db_manager()
    
    # Create default user (no authentication needed)
    default_username = "default_user"
    default_password = "not_used"
    
    # Check if default user exists
    existing_user = db.get_user(default_username)
    
    if not existing_user:
        print("Creating default user for history storage...")
        hashed = hashlib.sha256(default_password.encode()).hexdigest()
        user_id = db.add_user(default_username, hashed)
        print(f"✅ Default user created with ID: {user_id}")
    else:
        print(f"✅ Default user already exists (ID: {existing_user.id})")
    
    print("\nDatabase initialization complete!")
    print("\n📊 Your optimization history will be stored locally")
    print("🚀 Run: streamlit run app_simple.py")

if __name__ == "__main__":
    init_database()
