#!/usr/bin/env python3
"""
Collect training images from webcam with keyboard labelling.

Controls:
  SPACE  — save current frame to data/training_images/
  Q      — quit
"""
import time
from pathlib import Path

import cv2


def main(out_dir: str = "data/training_images", camera_id: int = 0) -> None:
    save_dir = Path(out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    count = len(list(save_dir.glob("*.jpg")))
    print(f"Saving to {save_dir}  (existing: {count} images)")
    print("SPACE=save  Q=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cv2.putText(
            frame, f"Saved: {count}  SPACE=capture  Q=quit",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        cv2.imshow("Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            fname = save_dir / f"{int(time.time() * 1000)}.jpg"
            cv2.imwrite(str(fname), frame)
            count += 1
            print(f"  Saved: {fname.name}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. {count} images in {save_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/training_images")
    p.add_argument("--camera", default=0, type=int)
    args = p.parse_args()
    main(args.out, args.camera)
