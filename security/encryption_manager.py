"""
Encryption Manager for Real-time Streaming Analysis

Provides data encryption, secure key management, and cryptographic
operations for protecting sensitive clinical data.

Reference: Real-time Streaming Analysis - Task 4.2
"""

import os
import logging
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import base64
import json

logger = logging.getLogger(__name__)


class KeyManager:
    """
    Secure key management for encryption operations.
    
    Handles generation, storage, and rotation of encryption keys
    with secure key derivation and storage mechanisms.
    """
    
    def __init__(self, key_storage_path: str = ".keys"):
        """
        Initialize key manager.
        
        Args:
            key_storage_path: Path to store encryption keys
        """
        self.key_storage_path = key_storage_path
        self.master_key = None
        self.data_keys: Dict[str, bytes] = {}
        
        # Ensure key storage directory exists
        os.makedirs(key_storage_path, mode=0o700, exist_ok=True)
        
        # Initialize or load master key
        self._initialize_master_key()
    
    def _initialize_master_key(self):
        """Initialize or load master key."""
        master_key_file = os.path.join(self.key_storage_path, "master.key")
        
        if os.path.exists(master_key_file):
            # Load existing master key
            try:
                with open(master_key_file, 'rb') as f:
                    self.master_key = f.read()
                logger.info("Master key loaded from storage")
            except Exception as e:
                logger.error(f"Failed to load master key: {e}")
                raise
        else:
            # Generate new master key
            self.master_key = Fernet.generate_key()
            
            # Save master key securely
            try:
                with open(master_key_file, 'wb') as f:
                    f.write(self.master_key)
                os.chmod(master_key_file, 0o600)  # Read/write for owner only
                logger.info("New master key generated and saved")
            except Exception as e:
                logger.error(f"Failed to save master key: {e}")
                raise
    
    def derive_key(self, purpose: str, salt: Optional[bytes] = None) -> bytes:
        """
        Derive encryption key for specific purpose.
        
        Args:
            purpose: Purpose identifier for key derivation
            salt: Optional salt for key derivation
            
        Returns:
            Derived encryption key
        """
        if salt is None:
            salt = hashlib.sha256(purpose.encode()).digest()[:16]
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = kdf.derive(self.master_key)
        return key
    
    def get_data_key(self, key_id: str) -> bytes:
        """
        Get or generate data encryption key.
        
        Args:
            key_id: Unique identifier for the key
            
        Returns:
            Data encryption key
        """
        if key_id not in self.data_keys:
            self.data_keys[key_id] = self.derive_key(f"data_key_{key_id}")
        
        return self.data_keys[key_id]
    
    def rotate_key(self, key_id: str) -> bytes:
        """
        Rotate encryption key for specific purpose.
        
        Args:
            key_id: Key identifier to rotate
            
        Returns:
            New encryption key
        """
        # Generate new key with timestamp
        timestamp = datetime.now().isoformat()
        new_key = self.derive_key(f"data_key_{key_id}_{timestamp}")
        
        # Store old key for decryption of existing data
        old_key_id = f"{key_id}_old_{int(datetime.now().timestamp())}"
        if key_id in self.data_keys:
            self.data_keys[old_key_id] = self.data_keys[key_id]
        
        # Update current key
        self.data_keys[key_id] = new_key
        
        logger.info(f"Key rotated for {key_id}")
        return new_key


class DataEncryption:
    """
    Data encryption and decryption operations.
    
    Provides AES-256 encryption for data at rest and in transit
    with secure key management and integrity verification.
    """
    
    def __init__(self, key_manager: KeyManager):
        """
        Initialize data encryption.
        
        Args:
            key_manager: Key manager instance
        """
        self.key_manager = key_manager
    
    def encrypt_data(self, data: Union[str, bytes], key_id: str = "default") -> Dict[str, Any]:
        """
        Encrypt data with AES-256.
        
        Args:
            data: Data to encrypt (string or bytes)
            key_id: Key identifier for encryption
            
        Returns:
            Encrypted data package with metadata
        """
        try:
            # Convert string to bytes if necessary
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data
            
            # Get encryption key
            key = self.key_manager.get_data_key(key_id)
            
            # Generate random IV
            iv = secrets.token_bytes(16)
            
            # Create cipher
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
            encryptor = cipher.encryptor()
            
            # Pad data to block size
            padded_data = self._pad_data(data_bytes)
            
            # Encrypt data
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            
            # Calculate HMAC for integrity
            hmac_key = self.key_manager.derive_key(f"hmac_{key_id}")
            data_hmac = hmac.new(hmac_key, encrypted_data, hashlib.sha256).hexdigest()
            
            # Create encrypted package
            encrypted_package = {
                'encrypted_data': base64.b64encode(encrypted_data).decode('utf-8'),
                'iv': base64.b64encode(iv).decode('utf-8'),
                'key_id': key_id,
                'hmac': data_hmac,
                'timestamp': datetime.now().isoformat(),
                'algorithm': 'AES-256-CBC'
            }
            
            return encrypted_package
        
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_data(self, encrypted_package: Dict[str, Any]) -> bytes:
        """
        Decrypt data from encrypted package.
        
        Args:
            encrypted_package: Encrypted data package
            
        Returns:
            Decrypted data as bytes
        """
        try:
            # Extract package components
            encrypted_data = base64.b64decode(encrypted_package['encrypted_data'])
            iv = base64.b64decode(encrypted_package['iv'])
            key_id = encrypted_package['key_id']
            expected_hmac = encrypted_package['hmac']
            
            # Get decryption key
            key = self.key_manager.get_data_key(key_id)
            
            # Verify HMAC integrity
            hmac_key = self.key_manager.derive_key(f"hmac_{key_id}")
            calculated_hmac = hmac.new(hmac_key, encrypted_data, hashlib.sha256).hexdigest()
            
            if not hmac.compare_digest(expected_hmac, calculated_hmac):
                raise ValueError("Data integrity check failed")
            
            # Create cipher
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
            decryptor = cipher.decryptor()
            
            # Decrypt data
            padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
            
            # Remove padding
            decrypted_data = self._unpad_data(padded_data)
            
            return decrypted_data
        
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def _pad_data(self, data: bytes) -> bytes:
        """Pad data to AES block size using PKCS7."""
        block_size = 16
        padding_length = block_size - (len(data) % block_size)
        padding = bytes([padding_length] * padding_length)
        return data + padding
    
    def _unpad_data(self, padded_data: bytes) -> bytes:
        """Remove PKCS7 padding from data."""
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]


class SecureStorage:
    """
    Secure storage for sensitive data.
    
    Provides encrypted storage with automatic key rotation
    and secure deletion capabilities.
    """
    
    def __init__(self, storage_path: str = ".secure_storage"):
        """
        Initialize secure storage.
        
        Args:
            storage_path: Path for secure storage
        """
        self.storage_path = storage_path
        self.key_manager = KeyManager()
        self.encryption = DataEncryption(self.key_manager)
        
        # Ensure storage directory exists
        os.makedirs(storage_path, mode=0o700, exist_ok=True)
    
    def store_data(self, data_id: str, data: Any, key_id: str = "default") -> bool:
        """
        Store data securely.
        
        Args:
            data_id: Unique identifier for the data
            data: Data to store
            key_id: Encryption key identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Serialize data
            serialized_data = json.dumps(data, default=str)
            
            # Encrypt data
            encrypted_package = self.encryption.encrypt_data(serialized_data, key_id)
            
            # Store encrypted package
            storage_file = os.path.join(self.storage_path, f"{data_id}.enc")
            with open(storage_file, 'w') as f:
                json.dump(encrypted_package, f)
            
            # Set secure file permissions
            os.chmod(storage_file, 0o600)
            
            logger.info(f"Data stored securely: {data_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to store data {data_id}: {e}")
            return False
    
    def retrieve_data(self, data_id: str) -> Optional[Any]:
        """
        Retrieve and decrypt stored data.
        
        Args:
            data_id: Unique identifier for the data
            
        Returns:
            Decrypted data or None if not found
        """
        try:
            storage_file = os.path.join(self.storage_path, f"{data_id}.enc")
            
            if not os.path.exists(storage_file):
                return None
            
            # Load encrypted package
            with open(storage_file, 'r') as f:
                encrypted_package = json.load(f)
            
            # Decrypt data
            decrypted_bytes = self.encryption.decrypt_data(encrypted_package)
            decrypted_str = decrypted_bytes.decode('utf-8')
            
            # Deserialize data
            data = json.loads(decrypted_str)
            
            return data
        
        except Exception as e:
            logger.error(f"Failed to retrieve data {data_id}: {e}")
            return None
    
    def delete_data(self, data_id: str) -> bool:
        """
        Securely delete stored data.
        
        Args:
            data_id: Unique identifier for the data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            storage_file = os.path.join(self.storage_path, f"{data_id}.enc")
            
            if os.path.exists(storage_file):
                # Overwrite file with random data before deletion
                file_size = os.path.getsize(storage_file)
                with open(storage_file, 'wb') as f:
                    f.write(secrets.token_bytes(file_size))
                
                # Delete file
                os.remove(storage_file)
                
                logger.info(f"Data securely deleted: {data_id}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete data {data_id}: {e}")
            return False
    
    def list_stored_data(self) -> List[str]:
        """
        List all stored data identifiers.
        
        Returns:
            List of data identifiers
        """
        try:
            data_ids = []
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.enc'):
                    data_id = filename[:-4]  # Remove .enc extension
                    data_ids.append(data_id)
            return data_ids
        
        except Exception as e:
            logger.error(f"Failed to list stored data: {e}")
            return []


class EncryptionManager:
    """
    Main encryption manager coordinating all cryptographic operations.
    
    Provides high-level interface for data encryption, secure storage,
    and key management operations.
    """
    
    def __init__(self, storage_path: str = ".secure_data"):
        """
        Initialize encryption manager.
        
        Args:
            storage_path: Path for secure data storage
        """
        self.key_manager = KeyManager()
        self.data_encryption = DataEncryption(self.key_manager)
        self.secure_storage = SecureStorage(storage_path)
        
        # Key rotation schedule
        self.key_rotation_interval = timedelta(days=30)
        self.last_key_rotation = datetime.now()
    
    def encrypt_session_data(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """
        Encrypt and store session data.
        
        Args:
            session_id: Session identifier
            session_data: Session data to encrypt
            
        Returns:
            True if successful, False otherwise
        """
        return self.secure_storage.store_data(
            f"session_{session_id}", 
            session_data, 
            key_id="session_data"
        )
    
    def decrypt_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Decrypt and retrieve session data.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Decrypted session data or None
        """
        return self.secure_storage.retrieve_data(f"session_{session_id}")
    
    def encrypt_patient_data(self, patient_id: str, patient_data: Dict[str, Any]) -> bool:
        """
        Encrypt and store patient data.
        
        Args:
            patient_id: Patient identifier
            patient_data: Patient data to encrypt
            
        Returns:
            True if successful, False otherwise
        """
        return self.secure_storage.store_data(
            f"patient_{patient_id}", 
            patient_data, 
            key_id="patient_data"
        )
    
    def decrypt_patient_data(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        Decrypt and retrieve patient data.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Decrypted patient data or None
        """
        return self.secure_storage.retrieve_data(f"patient_{patient_id}")
    
    def encrypt_clinical_notes(self, note_id: str, note_data: Dict[str, Any]) -> bool:
        """
        Encrypt and store clinical notes.
        
        Args:
            note_id: Note identifier
            note_data: Note data to encrypt
            
        Returns:
            True if successful, False otherwise
        """
        return self.secure_storage.store_data(
            f"note_{note_id}", 
            note_data, 
            key_id="clinical_notes"
        )
    
    def decrypt_clinical_notes(self, note_id: str) -> Optional[Dict[str, Any]]:
        """
        Decrypt and retrieve clinical notes.
        
        Args:
            note_id: Note identifier
            
        Returns:
            Decrypted note data or None
        """
        return self.secure_storage.retrieve_data(f"note_{note_id}")
    
    def secure_delete_session(self, session_id: str) -> bool:
        """
        Securely delete session data.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        return self.secure_storage.delete_data(f"session_{session_id}")
    
    def secure_delete_patient(self, patient_id: str) -> bool:
        """
        Securely delete patient data.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            True if successful, False otherwise
        """
        return self.secure_storage.delete_data(f"patient_{patient_id}")
    
    def rotate_keys_if_needed(self):
        """Rotate encryption keys if rotation interval has passed."""
        if datetime.now() - self.last_key_rotation > self.key_rotation_interval:
            self._rotate_all_keys()
            self.last_key_rotation = datetime.now()
    
    def _rotate_all_keys(self):
        """Rotate all encryption keys."""
        key_ids = ["session_data", "patient_data", "clinical_notes"]
        
        for key_id in key_ids:
            try:
                self.key_manager.rotate_key(key_id)
                logger.info(f"Key rotated: {key_id}")
            except Exception as e:
                logger.error(f"Failed to rotate key {key_id}: {e}")
    
    def get_encryption_status(self) -> Dict[str, Any]:
        """
        Get encryption system status.
        
        Returns:
            Encryption status information
        """
        stored_data = self.secure_storage.list_stored_data()
        
        return {
            'encryption_enabled': True,
            'algorithm': 'AES-256-CBC',
            'key_rotation_interval_days': self.key_rotation_interval.days,
            'last_key_rotation': self.last_key_rotation.isoformat(),
            'next_key_rotation': (self.last_key_rotation + self.key_rotation_interval).isoformat(),
            'stored_data_count': len(stored_data),
            'stored_sessions': len([d for d in stored_data if d.startswith('session_')]),
            'stored_patients': len([d for d in stored_data if d.startswith('patient_')]),
            'stored_notes': len([d for d in stored_data if d.startswith('note_')])
        }