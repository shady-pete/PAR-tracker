# Video Tracking Script

`tracker.py` is the primary script used to process video files.

## Command Line Arguments

| Argument       | Default                                  | Description                                             |
|----------------|------------------------------------------|---------------------------------------------------------|
| `--input`      | `../video.mp4`                           | Path to the input video file to process                 |
| `--output`     | ` ./output/output.mp4`                   | Path to the output video file                           |
| `--config`     | `./config/configuration_example.txt`     | Path to the camera configuration file                   |

## Notes

- Parameters such as `skip_frames` and `classification_interval` are hard-coded in the script. These can be adjusted by modifying the first few lines of `tracker.py`.
- To enable the masked approach, set the `USE_MASK` variable to `True` in the script (not raccomended).


## Usage

Basic usage:
```bash
python tracker.py --input <path_to_video> --output <path_to_output> --config <path_to_config>
