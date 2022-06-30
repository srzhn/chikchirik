import os
import sys
from pathlib import Path
# from dotenv import load_dotenv
import json

# config_template = """
# # CLEARML-AGENT configuration file
# api {{
#     # Notice: 'host' is the api server (default port 8008), not the web server.
#     api_server: {api_server}
#     web_server: {web_server}
#     files_server: {files_server}
#     # Credentials are generated using the webapp, /profile
#     # Override with os environment: CLEARML_API_ACCESS_KEY / CLEARML_API_SECRET_KEY
#     credentials {{"access_key": "{access_key}", "secret_key": "{secret_key}"}}
# }}
# agent{{
#     # Set GIT user/pass credentials (if user/pass are set, GIT protocol will be set to https)
#     # leave blank for GIT SSH credentials (set force_git_ssh_protocol=true to force SSH protocol)
#     # **Notice**: GitHub personal token is equivalent to password, you can put it directly into `git_pass`
#     # To learn how to generate git token GitHub/Bitbucket/GitLab:
#     # https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token
#     # https://support.atlassian.com/bitbucket-cloud/docs/app-passwords/
#     # https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html
#     git_user=""
#     git_pass=""
#     # Limit credentials to a single domain, for example: github.com,
#     # all other domains will use public access (no user/pass). Default: always send user/pass for any VCS domain
#     git_host=""
#     # cached virtual environment folder
#     venvs_cache: {{
#         # maximum number of cached venvs
#         max_entries: 10
#         # minimum required free space to allow for cache entry, disable by passing 0 or negative value
#         free_space_threshold_gb: 2.0
#         # unmark to enable virtual environment caching
#         path: ~/.clearml/venvs-cache
#     }}
# }}
# """

config_template = """
# CLEARML-AGENT configuration file
api {
    # Notice: 'host' is the api server (default port 8008), not the web server.
    api_server: %s
    web_server: %s
    files_server: %s
    # Credentials are generated using the webapp, /profile
    # Override with os environment: CLEARML_API_ACCESS_KEY / CLEARML_API_SECRET_KEY
    credentials {"access_key": "%s", "secret_key": "%s"}
}
agent{
    # Set GIT user/pass credentials (if user/pass are set, GIT protocol will be set to https)
    # leave blank for GIT SSH credentials (set force_git_ssh_protocol=true to force SSH protocol)
    # **Notice**: GitHub personal token is equivalent to password, you can put it directly into `git_pass`
    # To learn how to generate git token GitHub/Bitbucket/GitLab:
    # https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token
    # https://support.atlassian.com/bitbucket-cloud/docs/app-passwords/
    # https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html
    git_user=""
    git_pass=""
    # Limit credentials to a single domain, for example: github.com,
    # all other domains will use public access (no user/pass). Default: always send user/pass for any VCS domain
    git_host=""
    # cached virtual environment folder
    venvs_cache: {
        # maximum number of cached venvs
        max_entries: 10
        # minimum required free space to allow for cache entry, disable by passing 0 or negative value
        free_space_threshold_gb: 2.0
        # unmark to enable virtual environment caching
        path: ~/.clearml/venvs-cache
    }
}
"""

# def fill_config(api_server, web_server, files_server,
#                 access_key, secret_key,
#                 template=config_template):
#     config = template.format(api_server=api_server, 
#                              web_server=web_server,
#                              files_server=files_server,
#                              access_key=access_key,
#                              secret_key=secret_key)
#     return config


def fill_config(template=config_template):
    # load_dotenv(dotenv_path=Path(".") / "config.env")
    # api_server = os.getenv("api_server") 
    # web_server = os.getenv("web_server") 
    # files_server = os.getenv("files_server") 
    # access_key = os.getenv("access_key") 
    # secret_key = os.getenv("secret_key") 

    env = json.load(open(Path('.', 'config.json'), 'r'))
    api_server = env.get("api_server") 
    web_server = env.get("web_server") 
    files_server = env.get("files_server") 
    access_key = env.get("access_key") 
    secret_key = env.get("secret_key") 
    

    # config = template.format(api_server=api_server, 
    #                          web_server=web_server,
    #                          files_server=files_server,
    #                          access_key=access_key,
    #                          secret_key=secret_key)

    config = template % (api_server, web_server, files_server, access_key, secret_key)
    return config

def touch_config(config):
    def save(config, path):
        with open(path, 'w') as f:
            f.write(config)

    if sys.platform=='linux':
        try:
            save(config, '~/clearml.conf')
        except:
            save(config, '/root/clearml.conf')
    elif 'win' in sys.platform:
        user = os.getlogin()
        path = Path('C:/Users', user, 'clearml.conf')        
        save(config, path)

if __name__=="__main__":
    config = fill_config(config_template)
    touch_config(config)
