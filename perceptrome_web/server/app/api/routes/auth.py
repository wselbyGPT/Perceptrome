from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.orm import Session

from ...deps import get_current_user, get_db
from ...models import User
from ...schemas import (
    AuthUserOut,
    ChangePasswordRequest,
    ForgotPasswordRequest,
    LoginRequest,
    MessageOut,
    RegisterRequest,
    ResendVerificationRequest,
    ResetPasswordRequest,
    SessionOut,
    UpdateProfileRequest,
    UserOut,
    VerifyEmailRequest,
)
from ...services import audit_service, auth_service, session_service, user_service

router = APIRouter(prefix='/api/auth', tags=['auth'])


@router.post('/register', response_model=UserOut)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    return user_service.user_out(auth_service.register_user(db, email=payload.email, password=payload.password, username=payload.username))


@router.get('/me', response_model=AuthUserOut)
def me(user: User = Depends(get_current_user)):
    return AuthUserOut(user=user_service.user_out(user))


@router.post('/login', response_model=AuthUserOut)
def login(payload: LoginRequest, request: Request, response: Response, db: Session = Depends(get_db)):
    user = auth_service.login_user(db, email=payload.email, password=payload.password, request=request)
    session_service.revoke_session_by_cookie(db, request.cookies.get(session_service.settings.session_cookie_name))
    raw_session = session_service.issue_session(db, user, request)
    session_service.set_session_cookie(response, raw_session)
    return AuthUserOut(user=user_service.user_out(user))


@router.post('/logout', response_model=MessageOut)
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    session_service.revoke_session_by_cookie(db, request.cookies.get(session_service.settings.session_cookie_name))
    session_service.clear_session_cookie(response)
    return MessageOut(message='Logged out')


@router.post('/verify-email', response_model=MessageOut)
def verify_email(payload: VerifyEmailRequest, db: Session = Depends(get_db)):
    return MessageOut(message=auth_service.verify_email_token(db, payload.token))


@router.post('/resend-verification', response_model=MessageOut)
def resend_verification(payload: ResendVerificationRequest, db: Session = Depends(get_db)):
    return MessageOut(message=auth_service.resend_verification(db, payload.email))


@router.post('/forgot-password', response_model=MessageOut)
def forgot_password(payload: ForgotPasswordRequest, db: Session = Depends(get_db)):
    return MessageOut(message=auth_service.forgot_password(db, payload.email))


@router.post('/reset-password', response_model=MessageOut)
def reset_password(payload: ResetPasswordRequest, request: Request, db: Session = Depends(get_db)):
    return MessageOut(message=auth_service.reset_password(db, token_value=payload.token, new_password=payload.new_password, request=request))


@router.post('/change-password', response_model=MessageOut)
def change_password(payload: ChangePasswordRequest, request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return MessageOut(message=auth_service.change_password(db, user=user, current_password=payload.current_password, new_password=payload.new_password, current_cookie=request.cookies.get(session_service.settings.session_cookie_name), request=request))


@router.get('/sessions', response_model=list[SessionOut])
def list_sessions(request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return session_service.list_sessions(db, user, current_cookie=request.cookies.get(session_service.settings.session_cookie_name))


@router.delete('/sessions/{session_id}', response_model=MessageOut)
def revoke_session(session_id: str, request: Request, response: Response, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    revoked_current = session_service.revoke_session_by_id(db, user, session_id, current_cookie=request.cookies.get(session_service.settings.session_cookie_name))
    audit_service.create_audit_event(db, action='auth.session_revoked', actor_user_id=user.id, target_user_id=user.id, request=request, metadata={'session_id': session_id, 'revoked_current_session': revoked_current})
    if revoked_current:
        session_service.clear_session_cookie(response)
    return MessageOut(message='Session revoked')


@router.post('/sessions/revoke-others', response_model=MessageOut)
def revoke_other_sessions(request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    revoked_count = session_service.revoke_other_sessions(db, user, current_cookie=request.cookies.get(session_service.settings.session_cookie_name))
    audit_service.create_audit_event(db, action='auth.other_sessions_revoked', actor_user_id=user.id, target_user_id=user.id, request=request, metadata={'revoked_session_count': revoked_count})
    return MessageOut(message=f'Revoked {revoked_count} other sessions')


@router.patch('/profile', response_model=AuthUserOut)
def update_profile(payload: UpdateProfileRequest, request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    updated_user = auth_service.update_profile(db, user=user, username=payload.username, request=request)
    return AuthUserOut(user=user_service.user_out(updated_user))
