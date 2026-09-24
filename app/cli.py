import argparse
import json
from pathlib import Path

from src.config import Settings
from src.inference import InferenceService
from src.logging_utils import configure_logging
from src.preprocessing import load_artifacts


def build_service(config_path: str) -> InferenceService:
    settings = Settings.from_file(config_path)
    service_config = settings.section("service")
    configure_logging(service_config["log_level"], settings.root_dir / service_config["log_file"])
    artifacts = load_artifacts(settings.path("model", "artifacts_dir"), settings.section("model"))
    return InferenceService(settings, artifacts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Olist delay inference")
    parser.add_argument("payload", type=Path, help="JSON file containing one request")
    parser.add_argument("--config", default="config/config.yaml")
    arguments = parser.parse_args()
    with arguments.payload.open("r", encoding="utf-8") as payload_file:
        payload = json.load(payload_file)
    print(json.dumps(build_service(arguments.config).predict(payload), indent=2))


if __name__ == "__main__":
    main()
