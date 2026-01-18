"""
Authentication and Security for Clinical Dashboard

Provides user authentication, role-based access control, and session security
for the clinical dashboard interface.
"""

import hashlib
import hmac
import logging
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
import streamlit as st
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User role enumeration for access control."""
    ADMIN = "admin"
    CLINICIAN = "clinician"
    OBSERVER = "observer"
    TECHNICIAN = "technician"


class Permission(Enum):
    """Permission enumeration for fine-grained access control."""
    VIEW_SESSIONS = "view_sessions"
    MANAGE_SESSIONS = "manage_sessions"
    ACKNOWLEDGE_ALERTS = "acknowledge_alerts"
    VIEW_PATIENT_DATA = "view_patient_data"
    EXPORT_DATA = "export_data"
    SYSTEM_ADMIN = "system_admin"
    CLINICAL_NOTES = "clinical_notes"


@dataclass
class User:
    """User data structure."""
    user_id: str
    username: str
    email: str
    role: UserRole
    permissions: Set[Permission]
    created_at: datetime
    last_login: Optional[datetime] = None
    active: bool = True
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions
    
    def can_access_session(self, session_id: str) -> bool:
        """Check if user can access specific session."""
        # Admin and clinicians can access all sessions
        if self.role in [UserRole.ADMIN, UserRole.CLINICIAN]:
            return True
        
        # Observers can view but not manage
        if self.role == UserRole.OBSERVER:
            return self.has_permission(Permission.VIEW_SESSIONS)
        
        return False


class AuthenticationManager:
    """
    Manages user authentication and session security.
    
    Provides secure login, session management, and role-based access control
    for the clinical dashboard.
    """
    
    def __init__(self, session_timeout_minutes: int = 60):
        """
        Initialize authentication manager.
        
        Args:
            session_timeout_minutes: Session timeout in minutes
        """
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        
        # User database (in production, use proper database)
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        # Role permissions mapping
        self.role_permissions = {
            UserRole.ADMIN: {
                Permission.VIEW_SESSIONS,
                Permission.MANAGE_SESSIONS,
                Permission.ACKNOWLEDGE_ALERTS,
                Permission.VIEW_PATIENT_DATA,
                Permission.EXPORT_DATA,
                Permission.SYSTEM_ADMIN,
                Permission.CLINICAL_NOTES
            },
            UserRole.CLINICIAN: {
                Permission.VIEW_SESSIONS,
                Permission.MANAGE_SESSIONS,
                Permission.ACKNOWLEDGE_ALERTS,
                Permission.VIEW_PATIENT_DATA,
                Permission.CLINICAL_NOTES
            },
            UserRole.OBSERVER: {
                Permission.VIEW_SESSIONS,
                Permission.VIEW_PATIENT_DATA
            },
            UserRole.TECHNICIAN: {
                Permission.VIEW_SESSIONS,
                Permission.SYSTEM_ADMIN
            }
        }
        
        # Initialize default users
        self._create_default_users()
    
    def _create_default_users(self):
        """Create default users for development/demo."""
        default_users = [
            {
                'user_id': 'admin_001',
                'username': 'admin',
                'email': 'admin@clinic.local',
                'role': UserRole.ADMIN,
                'password': 'admin123'  # In production, use proper hashing
            },
            {
                'user_id': 'clinician_001',
                'username': 'dr_smith',
                'email': 'dr.smith@clinic.local',
                'role': UserRole.CLINICIAN,
                'password': 'clinic123'
            },
            {
                'user_id': 'observer_001',
                'username': 'observer',
                'email': 'observer@clinic.local',
                'role': UserRole.OBSERVER,
                'password': 'observe123'
            }
        ]
        
        for user_data in default_users:
            user = User(
                user_id=user_data['user_id'],
                username=user_data['username'],
                email=user_data['email'],
                role=user_data['role'],
                permissions=self.role_permissions[user_data['role']],
                created_at=datetime.now()
            )
            self.users[user_data['username']] = user
            
            # Store password hash (simplified for demo)
            self.users[user_data['username']].password_hash = self._hash_password(
                user_data['password']
            )
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256 (simplified for demo)."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate user credentials.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            User object if authentication successful, None otherwise
        """
        user = self.users.get(username)
        if not user or not user.active:
            return None
        
        password_hash = self._hash_password(password)
        if hasattr(user, 'password_hash') and user.password_hash == password_hash:
            user.last_login = datetime.now()
            logger.info(f"User {username} authenticated successfully")
            return user
        
        logger.warning(f"Authentication failed for user {username}")
        return None
    
    def create_session(self, user: User) -> str:
        """
        Create a new user session.
        
        Args:
            user: Authenticated user
            
        Returns:
            Session token
        """
        session_token = secrets.token_urlsafe(32)
        
        self.sessions[session_token] = {
            'user_id': user.user_id,
            'username': user.username,
            'role': user.role,
            'permissions': user.permissions,
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'active': True
        }
        
        logger.info(f"Session created for user {user.username}")
        return session_token
    
    def validate_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """
        Validate session token and check timeout.
        
        Args:
            session_token: Session token to validate
            
        Returns:
            Session data if valid, None otherwise
        """
        session = self.sessions.get(session_token)
        if not session or not session['active']:
            return None
        
        # Check timeout
        if datetime.now() - session['last_activity'] > self.session_timeout:
            self.invalidate_session(session_token)
            return None
        
        # Update last activity
        session['last_activity'] = datetime.now()
        return session
    
    def invalidate_session(self, session_token: str):
        """Invalidate a session."""
        if session_token in self.sessions:
            self.sessions[session_token]['active'] = False
            logger.info(f"Session invalidated: {session_token[:8]}...")
    
    def check_permission(self, session_token: str, permission: Permission) -> bool:
        """
        Check if session has specific permission.
        
        Args:
            session_token: Session token
            permission: Permission to check
            
        Returns:
            True if permission granted, False otherwise
        """
        session = self.validate_session(session_token)
        if not session:
            return False
        
        return permission in session['permissions']
    
    def get_user_from_session(self, session_token: str) -> Optional[User]:
        """
        Get user object from session token.
        
        Args:
            session_token: Session token
            
        Returns:
            User object if session valid, None otherwise
        """
        session = self.validate_session(session_token)
        if not session:
            return None
        
        return self.users.get(session['username'])


class StreamlitAuthenticator:
    """
    Streamlit-specific authentication integration.
    
    Provides authentication UI components and session management
    integrated with Streamlit's session state.
    """
    
    def __init__(self, auth_manager: AuthenticationManager):
        """
        Initialize Streamlit authenticator.
        
        Args:
            auth_manager: Authentication manager instance
        """
        self.auth_manager = auth_manager
    
    def login_form(self) -> bool:
        """
        Display login form and handle authentication.
        
        Returns:
            True if user is authenticated, False otherwise
        """
        # Check if already authenticated
        if self.is_authenticated():
            return True
        
        st.title("🏥 Clinical Dashboard Login")
        
        with st.form("login_form"):
            st.markdown("### Please enter your credentials")
            
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                login_button = st.form_submit_button("Login")
            
            if login_button:
                if username and password:
                    user = self.auth_manager.authenticate_user(username, password)
                    
                    if user:
                        session_token = self.auth_manager.create_session(user)
                        
                        # Store in Streamlit session state
                        st.session_state['authenticated'] = True
                        st.session_state['session_token'] = session_token
                        st.session_state['user'] = user
                        st.session_state['username'] = user.username
                        st.session_state['role'] = user.role.value
                        
                        st.success(f"Welcome, {user.username}!")
                        st.rerun()
                        return True
                    else:
                        st.error("Invalid username or password")
                else:
                    st.error("Please enter both username and password")
        
        # Display demo credentials
        with st.expander("Demo Credentials"):
            st.markdown("""
            **Admin**: admin / admin123
            
            **Clinician**: dr_smith / clinic123
            
            **Observer**: observer / observe123
            """)
        
        return False
    
    def is_authenticated(self) -> bool:
        """Check if current user is authenticated."""
        if not st.session_state.get('authenticated', False):
            return False
        
        session_token = st.session_state.get('session_token')
        if not session_token:
            return False
        
        # Validate session
        session = self.auth_manager.validate_session(session_token)
        if not session:
            self.logout()
            return False
        
        return True
    
    def logout(self):
        """Logout current user."""
        session_token = st.session_state.get('session_token')
        if session_token:
            self.auth_manager.invalidate_session(session_token)
        
        # Clear session state
        for key in ['authenticated', 'session_token', 'user', 'username', 'role']:
            if key in st.session_state:
                del st.session_state[key]
        
        st.rerun()
    
    def get_current_user(self) -> Optional[User]:
        """Get current authenticated user."""
        if not self.is_authenticated():
            return None
        
        session_token = st.session_state.get('session_token')
        return self.auth_manager.get_user_from_session(session_token)
    
    def check_permission(self, permission: Permission) -> bool:
        """Check if current user has specific permission."""
        session_token = st.session_state.get('session_token')
        if not session_token:
            return False
        
        return self.auth_manager.check_permission(session_token, permission)
    
    def require_permission(self, permission: Permission) -> bool:
        """
        Require specific permission, show error if not granted.
        
        Args:
            permission: Required permission
            
        Returns:
            True if permission granted, False otherwise
        """
        if self.check_permission(permission):
            return True
        
        st.error(f"Access denied. Required permission: {permission.value}")
        return False
    
    def display_user_info(self):
        """Display current user information in sidebar."""
        user = self.get_current_user()
        if not user:
            return
        
        with st.sidebar:
            st.markdown("---")
            st.markdown("### User Information")
            st.markdown(f"**User**: {user.username}")
            st.markdown(f"**Role**: {user.role.value.title()}")
            
            if st.button("Logout", key="logout_button"):
                self.logout()


# Global authentication manager instance
auth_manager = AuthenticationManager()
streamlit_auth = StreamlitAuthenticator(auth_manager)