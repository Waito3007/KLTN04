# backend/core/oauth.py
import os
from dotenv import load_dotenv
from authlib.integrations.starlette_client import OAuth

load_dotenv()

oauth = OAuth()

oauth.register(
    name='github',
    client_id=os.getenv('GITHUB_CLIENT_ID'),
    client_secret=os.getenv('GITHUB_CLIENT_SECRET'),
     access_token_url='https://github.com/login/oauth/access_token',
    access_token_params=None,
    authorize_url='https://github.com/login/oauth/authorize',
    authorize_params=None,
    api_base_url='https://api.github.com/',  # <--- Cái này đang bị thiếu
    client_kwargs={'scope': 'read:user user:email'},
)
