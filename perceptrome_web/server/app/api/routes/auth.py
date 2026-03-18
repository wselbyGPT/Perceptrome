from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.orm import Session

from ...deps import get_current_user, get_db
from ...models import User
from ...schemas import ChangePasswordRequest, ForgotPasswordRequest, LoginRequest, MessageOut, RegisterRequest, ResendVerificationRequest, ResetPasswordRequest, UserOut, VerifyEmailRequest
from ...services import auth_service, session_service, user_service

router = APIRouter(prefix='/api/auth', tags=['auth'])


@router.post('/register', response_model=UserOut)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    return user_service.user_out(auth_service.register_user(db, email=payload.email, password=payload.password, username=payload.username))


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
def reset_password(payload: ResetPasswordRequest, db: Session = Depends(get_db)):
    return MessageOut(message=auth_service.reset_password(db, token_value=payload.token, new_password=payload.new_password))


@router.post('/login', response_model=UserOut)
def login(payload: LoginRequest, request: Request, response: Response, db: Session = Depends(get_db)):
    user = auth_service.login_user(db, email=payload.email, password=payload.password, request=request)
    session_service.revoke_session_by_cookie(db, request.cookies.get('perceptrome_session') or request.cookies.get(session_service.settings.session_cookie_name))
    raw_session = session_service.issue_session(db, user, request)
    session_service.set_session_cookie(response, raw_session)
    return user_service.user_out(user)


@router.post('/logout', response_model=MessageOut)
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    session_service.revoke_session_by_cookie(db, request.cookies.get(session_service.settings.session_cookie_name))
    session_service.clear_session_cookie(response)
    return MessageOut(message='Logged out')


@router.get('/me', response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user_service.user_out(user)


@router.post('/change-password', response_model=MessageOut)
def change_password(payload: ChangePasswordRequest, request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return MessageOut(message=auth_service.change_password(db, user=user, current_password=payload.current_password, new_password=payload.new_password, current_cookie=request.cookies.get(session_service.settings.session_cookie_name)))
