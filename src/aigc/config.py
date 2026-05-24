from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-v4-pro"
    default_meeting_duration_minutes: int = 60
    default_meeting_start_hour: int = 10
    max_retry_count: int = 1


settings = Settings()
