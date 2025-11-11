from pydantic import BaseModel
from typing import Literal
from pathlib import Path


camera_type = Literal["USB", "PiCamera"]


class CalibrationSettings(BaseModel):
    calibration_path: str | Path
    camera_type: camera_type


# Set and create a folder in the same folder
calibration_path = (Path(__file__).parent / "calibration_image").resolve()
print(f"Setting and Creating Calibration Path to {calibration_path}")
calibration_path.mkdir(exist_ok=True)


def get_basic_camera_info() -> CalibrationSettings:
    global calibration_path

    # Step 1: Confirm or update base calibration path
    while True:
        confirm = (
            input(
                f"The following is the current calibration path settings:\n"
                f"📁 Calibration Path: {calibration_path}\n"
                "Would you like to continue with these settings or provide a new base folder path?\n"
                "(yes to continue / no to provide a new path): "
            )
            .strip()
            .lower()
        )

        if confirm in ["yes", "y"]:
            break
        elif confirm in ["no", "n"]:
            filepath = input(
                "Enter the (absolute) directory where the images will be placed: "
            ).strip()
            try:
                calibration_path = Path(filepath)
                calibration_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Calibration Path Set: {calibration_path}")
                break
            except Exception as e:
                print(f"❌ Could not set the calibration path: {e}")
        else:
            print("Please enter 'yes' or 'no'.")

    # Step 2: Create or confirm camera folder
    while True:
        camera_name = input("Enter a name for the camera folder: ").strip().lower()
        camera_folder = calibration_path / camera_name

        if camera_folder.exists():
            confirm = (
                input(
                    f"The folder '{camera_folder}' already exists. Do you want to overwrite it? (yes/no): "
                )
                .strip()
                .lower()
            )
            if confirm in ["yes", "y"]:
                print("⚠️ Existing folder will be used.")
                break
            elif confirm in ["no", "n"]:
                print("Please enter a different name.")
                continue
        else:
            camera_folder.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created new camera folder: {camera_folder}")
            break

    # Step 3: Choose camera type
    options = ["USB", "PiCamera"]
    while True:
        camera = input(f"Choose a camera ({', '.join(options)}): ").strip()
        if camera in options:
            print(f"🎥 You selected: {camera}")
            break
        print("Invalid choice. Please select from the listed options.")

    # Step 4: Return configuration
    return CalibrationSettings(calibration_path=camera_folder, camera_type=camera)


if __name__ == "__main__":
    settings = get_basic_camera_info()
    print("Calibration Settings Set OK")
